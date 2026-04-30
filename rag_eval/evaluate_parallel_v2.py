# -*- coding: utf-8 -*-
"""
Ragas Parallel Evaluation - 10 Workers
并行优化版本，大幅加速评估
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datasets import Dataset as HFDataset
from openai import OpenAI
import threading

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rag_project.pipeline import RAGPipeline
except ImportError:
    # 尝试直接导入
    sys.path.insert(0, str(PROJECT_ROOT / "rag_project"))
    from pipeline import RAGPipeline


# Configuration
API_KEY = "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 10
BATCH_SIZE = 50
TIMEOUT = 120

# Thread-safe logging and pipeline access
print_lock = threading.Lock()
pipeline_lock = threading.Lock()


def log(msg):
    """Thread-safe log"""
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def evaluate_one(item, pipeline, client):
    """Evaluate one question (both metrics)"""
    q_id = item.get('id', '')
    question = item['question']
    ground_truth = item['ground_truth']

    try:
        # Search - use lock to prevent CUDA multi-threading issues
        with pipeline_lock:
            results = pipeline.search(question, top_k=3)
        contexts = [doc['text'] for doc in results]

        if not contexts:
            return {
                'id': q_id,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'error': 'No contexts'
            }

        answer = contexts[0][:500]

        # Create dataset
        from ragas import evaluate
        from ragas.metrics import context_recall, context_precision
        from ragas.llms import llm_factory

        sample_ds = HFDataset.from_list([{
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }])

        # Create LLM
        llm = llm_factory(
            model=MODEL_NAME,
            provider="openai",
            client=client,
            max_tokens=8192,
            timeout=TIMEOUT
        )

        # Evaluate
        result = evaluate(
            dataset=sample_ds,
            metrics=[context_precision, context_recall],
            llm=llm,
            raise_exceptions=True
        )

        df = result.to_pandas()
        row = df.iloc[0]

        return {
            'id': q_id,
            'context_precision': float(row.get('context_precision', 0)),
            'context_recall': float(row.get('context_recall', 0))
        }

    except Exception as e:
        return {
            'id': q_id,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'error': str(e)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="rag_eval/evals/datasets/smart_testset_full.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="rag_eval/evals/experiments/ragas_parallel_results.json")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # Use args values
    batch_size = args.batch_size
    max_workers = args.workers

    print("=" * 80)
    print("Ragas Parallel Evaluation - 10 Workers")
    print("=" * 80)
    print(f"Testset: {args.testset}")
    print(f"Top-K: {args.top_k}")
    print(f"Workers: {max_workers}")
    print(f"Batch Size: {batch_size}")
    print(f"Model: {MODEL_NAME}")
    print("")

    # Load questions
    log("Loading testset...")
    with open(args.testset, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_questions = data['questions']
    for i, q in enumerate(all_questions):
        q['id'] = i

    total = len(all_questions)
    log(f"Total questions: {total}")
    log("")

    # Initialize
    log("Initializing Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml"
    )
    log("Pipeline ready\n")

    # Create client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    log("DeepSeek client ready\n")

    # Split into batches
    batches = []
    for i in range(0, total, batch_size):
        batch = all_questions[i:i+batch_size]
        batches.append(batch)

    num_batches = len(batches)
    log(f"Split into {num_batches} batches of {batch_size} questions each")
    log("")

    # Process batches
    log("=" * 80)
    log("Starting parallel evaluation")
    log("=" * 80)
    log(f"Estimated time: {total * 22 / MAX_WORKERS / 60:.1f} minutes")
    log("=" * 80)
    log("")

    all_results = []
    start_time = time.time()

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        log(f"Batch {batch_num}/{num_batches}: {len(batch)} questions")

        batch_start = time.time()

        # Parallel processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(evaluate_one, q, pipeline, client): q
                for q in batch
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=TIMEOUT + 10)
                    all_results.append(result)
                    completed += 1

                    if completed % 5 == 0:
                        elapsed = time.time() - batch_start
                        speed = completed / elapsed
                        eta = (len(batch) - completed) / speed
                        log(f"  Progress: {completed}/{len(batch)} ({completed/len(batch)*100:.0f}%), "
                            f"{speed:.2f} q/s, ETA: {eta:.0f}s")

                except Exception as e:
                    log(f"  Task failed: {e}")
                    all_results.append({
                        'id': completed,
                        'context_precision': 0.0,
                        'context_recall': 0.0,
                        'error': str(e)
                    })
                    completed += 1

        batch_elapsed = time.time() - batch_start
        success = sum(1 for r in all_results if 'error' not in r)
        log(f"Batch {batch_num} complete: {success}/{len(batch)} successful, {batch_elapsed:.1f}s")

        # Save checkpoint
        checkpoint = {
            'batch_completed': batch_num,
            'total_completed': len(all_results),
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_path = Path(args.output).parent / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        log(f"Checkpoint saved\n")

    total_elapsed = time.time() - start_time

    # Calculate metrics
    valid_p = [r['context_precision'] for r in all_results if 'error' not in r]
    valid_r = [r['context_recall'] for r in all_results if 'error' not in r]

    avg_p = sum(valid_p) / len(valid_p) if valid_p else 0
    avg_r = sum(valid_r) / len(valid_r) if valid_r else 0
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

    # Results
    log("=" * 80)
    log("Evaluation Complete!")
    log("=" * 80)
    log(f"Total time: {total_elapsed/60:.1f} minutes")
    log(f"Questions: {len(all_results)}")
    log(f"Success rate: {len(valid_p)/len(all_results)*100:.1f}%")
    log("")
    log(f"Context Precision: {avg_p:.4f}")
    log(f"Context Recall:    {avg_r:.4f}")
    log(f"F1 Score:          {f1:.4f}")
    log("")
    log(f"Speed: {len(all_results)/total_elapsed:.2f} questions/second")
    log(f"Speedup: {464*22/total_elapsed:.1f}x (vs serial {464*22/60:.1f} min)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        'metrics': {
            'context_precision': avg_p,
            'context_recall': avg_r,
            'f1_score': f1
        },
        'num_samples': len(all_results),
        'successful_samples': len(valid_p),
        'parallel_workers': MAX_WORKERS,
        'total_time_seconds': total_elapsed,
        'speed_per_question': total_elapsed / len(all_results),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    log(f"\nResults saved: {output_path}")

    # Save detailed results
    detail_path = output_path.parent / "detailed_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    log(f"Detailed results: {detail_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
