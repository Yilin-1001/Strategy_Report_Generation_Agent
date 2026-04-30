# -*- coding: utf-8 -*-
"""
Ragas Parallel Evaluation - Thread-safe version
主线程预创建 ragas LLM，ragas.evaluate() 加锁序列化
pipeline.search() 基于 HTTP API，天然线程安全，可并行
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
    sys.path.insert(0, str(PROJECT_ROOT / "rag_project"))
    from pipeline import RAGPipeline


# Configuration
API_KEY = "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 10
BATCH_SIZE = 50
TIMEOUT = 120

# Thread-safe globals
print_lock = threading.Lock()
log_file = None
log_lock = threading.Lock()
ragas_lock = threading.Lock()     # Serialize ragas.evaluate()
ragas_llm = None
ragas_metrics = None


def setup_logging(output_path):
    """Setup log file"""
    global log_file
    log_dir = Path(output_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg, also_print=True):
    """Thread-safe log to file and optionally print"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {msg}\n"

    with log_lock:
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg)

    if also_print:
        with print_lock:
            print(f"[{timestamp}] {msg}", flush=True)


def setup_ragas():
    """Set env vars so ragas's internal OpenAI() calls work with DeepSeek API"""
    os.environ['OPENAI_API_KEY'] = API_KEY
    os.environ['OPENAI_BASE_URL'] = BASE_URL
    log(f"Ragas env vars set for OpenAI_API_KEY={API_KEY}, BASE_URL={BASE_URL}")


def evaluate_one(item, pipeline, client, use_reranker=False, top_k=3):
    """Evaluate one question — ragas.evaluate() is serialized via ragas_lock"""
    q_id = item.get('id', '')
    question = item['question']
    ground_truth = item['ground_truth']

    try:
        log(f"  Evaluating Q{q_id}: {question[:50]}...", also_print=False)

        # Search - API-based (SiliconFlow HTTP), thread-safe
        results = pipeline.search(question, top_k=top_k, use_reranker=use_reranker)
        contexts = [doc['text'] for doc in results]

        if not contexts:
            log(f"    Q{q_id}: No contexts found", also_print=False)
            return {
                'id': q_id,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'error': 'No contexts'
            }

        answer = contexts[0][:500]

        # Serialize ragas.evaluate() — asyncio is not thread-safe
        with ragas_lock:
            from ragas import evaluate
            from ragas.metrics import context_recall, context_precision
            from ragas.llms import llm_factory

            # Create fresh LLM per call (avoids shared state issues)
            llm = llm_factory(
                model=MODEL_NAME,
                provider="openai",
                client=client,
                max_tokens=8192,
                timeout=TIMEOUT
            )

            sample_ds = HFDataset.from_list([{
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            }])

            result = evaluate(
                dataset=sample_ds,
                metrics=[context_precision, context_recall],
                llm=llm,
                raise_exceptions=True
            )

        df = result.to_pandas()
        row = df.iloc[0]

        cp = float(row.get('context_precision', 0))
        cr = float(row.get('context_recall', 0))

        log(f"    Q{q_id}: CP={cp:.4f}, CR={cr:.4f}", also_print=False)

        return {
            'id': q_id,
            'context_precision': cp,
            'context_recall': cr
        }

    except Exception as e:
        log(f"    Q{q_id}: ERROR - {str(e)[:100]}", also_print=False)
        return {
            'id': q_id,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'error': str(e)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="rag_eval/evals/datasets/ragas_testset_doc_51q.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="rag_eval/evals/experiments/ragas_parallel_results.json")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-reranker", action="store_true", default=False,
                        help="启用重排序进行检索评估")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output)

    batch_size = args.batch_size
    max_workers = args.workers

    log("=" * 80, also_print=True)
    log("Ragas Parallel Evaluation (Thread-safe)", also_print=True)
    log("=" * 80, also_print=True)
    log(f"Testset: {args.testset}")
    log(f"Top-K: {args.top_k}")
    log(f"Workers: {max_workers}")
    log(f"Batch Size: {batch_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"Reranker: {'启用' if args.use_reranker else '禁用'}")
    log("")
    log(f"Log file: {log_file}")
    log("")

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

    # Initialize pipeline
    log("Initializing Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml"
    )
    log("Pipeline ready")
    log("")

    # Create client + pre-create ragas LLM
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    log("DeepSeek client ready")

    log("Initializing Ragas LLM...")
    setup_ragas()
    log("")

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
    log("Starting evaluation (ragas calls serialized, search parallel)")
    log("=" * 80)
    log("")

    all_results = []
    start_time = time.time()

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        log(f"Batch {batch_num}/{num_batches}: {len(batch)} questions")

        batch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_one, q, pipeline, args.use_reranker, args.top_k): q
                for q in batch
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=TIMEOUT + 30)
                    all_results.append(result)
                    completed += 1

                    if completed % 5 == 0 or completed == len(batch):
                        elapsed = time.time() - batch_start
                        speed = completed / elapsed if elapsed > 0 else 0
                        eta = (len(batch) - completed) / speed if speed > 0 else 0
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
        log(f"Checkpoint saved")
        log("")

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
        'parallel_workers': max_workers,
        'total_time_seconds': total_elapsed,
        'speed_per_question': total_elapsed / len(all_results),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    log(f"\nResults saved: {output_path}")

    # Save detailed results (separate file per run)
    run_tag = "reranker" if args.use_reranker else "baseline"
    detail_path = output_path.parent / f"detailed_results_{run_tag}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    log(f"Detailed results: {detail_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
