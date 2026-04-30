"""
Reranker comparison evaluation.

Runs three experiments on the same testset:
1. Baseline (vector search only)
2. BGE-Reranker-v2-m3
3. Qwen3-Reranker-0.6B

Produces comparison report with metrics.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_project.pipeline import RAGPipeline


# Reranker configurations to test
RERANKER_CONFIGS = {
    "baseline": None,
    "bge_reranker": {
        "reranker": {
            "enabled": True,
            "provider": "siliconflow",
            "model": "BAAI/bge-reranker-v2-m3",
            "api_base_url": "https://api.siliconflow.cn/v1",
            "api_key": "sk-galtgnlesfoydjzetapqerkapxzqbtdnlnyulgibiqymvqik",
            "expansion_factor": 3,
            "api_timeout": 30,
            "max_chunks_per_doc": 1,
            "overlap_tokens": 40,
        }
    },
    "qwen3_reranker": {
        "reranker": {
            "enabled": True,
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "api_base_url": "https://api.siliconflow.cn/v1",
            "api_key": "sk-galtgnlesfoydjzetapqerkapxzqbtdnlnyulgibiqymvqik",
            "expansion_factor": 3,
            "api_timeout": 30,
        }
    },
}


def run_single_experiment(
    experiment_name: str,
    testset: List[Dict],
    top_k: int = 5,
    reranker_config: Dict = None
) -> Dict[str, Any]:
    """
    Run a single retrieval experiment.

    Args:
        experiment_name: Name of the experiment
        testset: List of {'question', 'ground_truth'} dicts
        top_k: Number of results to retrieve
        reranker_config: If None, baseline. If dict, enable reranker.

    Returns:
        Experiment results dict
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")

    # Create pipeline
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
    )

    # If reranker config provided, create reranker and set it on the pipeline
    use_reranker = False
    if reranker_config is not None:
        from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
        pipeline.reranker = SiliconFlowReranker.from_config_dict(reranker_config)
        pipeline.reranker_config = reranker_config['reranker']
        use_reranker = True
        print(f"  Reranker: {reranker_config['reranker']['model']}")
    else:
        print("  Mode: Baseline (vector search only)")

    results = []
    total_latency = 0

    for i, item in enumerate(testset, 1):
        question = item['question']
        ground_truth = item['ground_truth']

        start_time = time.time()
        search_results = pipeline.search(
            question, top_k=top_k, use_reranker=use_reranker
        )
        latency = time.time() - start_time
        total_latency += latency

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "num_retrieved": len(search_results),
            "top1_score": search_results[0]['score'] if search_results else 0,
            "top1_source": (
                search_results[0]['metadata'].get('source', '')
                if search_results else ''
            ),
            "retrieved_docs": search_results,
            "latency": latency,
        })

        print(f"  [{i}/{len(testset)}] {question[:50]}... "
              f"(top1={results[-1]['top1_score']:.4f}, {latency:.2f}s)")

    # Compute statistics
    total = len(results)
    avg_score = sum(r['top1_score'] for r in results) / total
    avg_latency = total_latency / total
    high_score = sum(1 for r in results if r['top1_score'] >= 0.8)
    mid_score = sum(1 for r in results if 0.6 <= r['top1_score'] < 0.8)
    low_score = sum(1 for r in results if r['top1_score'] < 0.6)

    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_questions": total,
            "avg_top1_score": avg_score,
            "avg_latency": avg_latency,
            "total_latency": total_latency,
            "score_distribution": {
                "high_gte_0.8": high_score,
                "mid_0.6_0.8": mid_score,
                "low_lt_0.6": low_score,
            },
        },
        "results": results,
    }


def compare_experiments(all_results: List[Dict]) -> Dict:
    """Generate comparison report from multiple experiment results."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
        "summary_table": [],
    }

    for exp in all_results:
        name = exp["experiment_name"]
        stats = exp["statistics"]
        comparison["experiments"][name] = stats
        comparison["summary_table"].append({
            "experiment": name,
            "avg_top1_score": round(stats["avg_top1_score"], 4),
            "avg_latency": round(stats["avg_latency"], 2),
            "high_score_pct": round(
                stats["score_distribution"]["high_gte_0.8"] / stats["total_questions"] * 100, 1
            ),
            "mid_score_pct": round(
                stats["score_distribution"]["mid_0.6_0.8"] / stats["total_questions"] * 100, 1
            ),
            "low_score_pct": round(
                stats["score_distribution"]["low_lt_0.6"] / stats["total_questions"] * 100, 1
            ),
        })

    return comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reranker comparison evaluation")
    parser.add_argument(
        "--testset",
        type=str,
        default="rag_eval/evals/datasets/ragas_testset_doc_51q.json",
        help="Testset JSON file",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rag_eval/evals/experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["baseline", "bge_reranker", "qwen3_reranker"],
        choices=["baseline", "bge_reranker", "qwen3_reranker"],
        help="Which experiments to run",
    )

    args = parser.parse_args()

    # Load testset
    with open(args.testset, 'r', encoding='utf-8') as f:
        data = json.load(f)
        testset = data['questions']

    print(f"Loaded {len(testset)} questions from {args.testset}")
    print(f"Experiments: {args.experiments}")
    print(f"Top-K: {args.top_k}")

    # Run experiments
    all_results = []
    for exp_name in args.experiments:
        config = RERANKER_CONFIGS[exp_name]
        result = run_single_experiment(
            experiment_name=exp_name,
            testset=testset,
            top_k=args.top_k,
            reranker_config=config,
        )
        all_results.append(result)

        # Save individual results
        output_path = Path(args.output_dir) / f"reranker_{exp_name}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved: {output_path}")

    # Generate comparison report
    comparison = compare_experiments(all_results)
    comparison_path = Path(args.output_dir) / "reranker_comparison_report.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nComparison report: {comparison_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<25} {'Avg Score':>10} {'Avg Latency':>12} "
          f"{'High%':>8} {'Mid%':>8} {'Low%':>8}")
    print("-" * 80)
    for row in comparison["summary_table"]:
        print(f"{row['experiment']:<25} {row['avg_top1_score']:>10.4f} "
              f"{row['avg_latency']:>10.2f}s "
              f"{row['high_score_pct']:>7.1f}% {row['mid_score_pct']:>7.1f}% "
              f"{row['low_score_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
