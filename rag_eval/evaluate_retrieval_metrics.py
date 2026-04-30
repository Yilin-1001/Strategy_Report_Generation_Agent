# -*- coding: utf-8 -*-
"""
Reranker evaluation with MRR and NDCG metrics.

Compares baseline (vector search) vs reranker (BGE-Reranker-v2-m3 / Qwen3-Reranker-0.6B).

Metrics (computed without LLM, purely based on source document matching):
  - MRR@K    (Mean Reciprocal Rank)
  - NDCG@K   (Normalized Discounted Cumulative Gain)
  - Hit@K    (Hit Rate — fraction of queries with >=1 relevant doc in top-K)
  - P@K      (Precision@K — fraction of relevant docs in top-K)

Relevance criterion: a retrieved chunk is relevant if its source filename matches
the question's source_document field in the testset.

Note: Ragas does NOT provide MRR / NDCG — they are traditional IR metrics.
This script implements them using standard formulas widely used in information
retrieval research (Manning et al., "Introduction to Information Retrieval").

Usage:
    python rag_eval/evaluate_retrieval_metrics.py --top-k 10
    python rag_eval/evaluate_retrieval_metrics.py --experiments baseline bge_reranker --workers 5
    python rag_eval/evaluate_retrieval_metrics.py --experiments baseline qwen3_reranker --top-k 10
"""

import os
import sys
import json
import math
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.pipeline import RAGPipeline

# ─── Thread-safe printing ─────────────────────────────────────────────
_print_lock = threading.Lock()


def ts_print(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  IR Metrics
# ═══════════════════════════════════════════════════════════════════════

def relevance_labels(
    results: List[Dict],
    source_document: str,
) -> List[int]:
    """Return binary relevance labels for retrieved results.

    A result is relevant (1) if its ``metadata.source`` matches the
    question's ``source_document`` filename, otherwise 0.
    """
    labels = []
    for r in results:
        src = r.get("metadata", {}).get("source", "")
        labels.append(1 if src == source_document else 0)
    return labels


def mrr_at_k(labels: List[int], k: int) -> float:
    """MRR@K — reciprocal rank of the first relevant document.

    Returns 1/rank if a relevant doc is found within top-K, else 0.
    """
    for i in range(min(k, len(labels))):
        if labels[i] == 1:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(labels: List[int], k: int) -> float:
    """DCG@K = sum_{i=1}^{K} rel_i / log2(i + 1)"""
    val = 0.0
    for i in range(min(k, len(labels))):
        if labels[i] == 1:
            val += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed
    return val


def ndcg_at_k(labels: List[int], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K.

    IDCG is the ideal (best possible) DCG when all relevant docs are at
    the top positions.  With binary relevance the ideal ordering puts
    ``num_relevant`` ones at positions 1..num_relevant.
    """
    num_relevant = sum(labels)
    if num_relevant == 0:
        return 0.0

    dcg = dcg_at_k(labels, k)
    # Ideal: all 1s at the front
    ideal = [1] * num_relevant
    idcg = dcg_at_k(ideal, k)

    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(labels: List[int], k: int) -> float:
    """Hit@K — 1 if any relevant doc in top-K, else 0."""
    return 1.0 if any(labels[:k]) else 0.0


def precision_at_k(labels: List[int], k: int) -> float:
    """P@K — fraction of relevant docs in top-K."""
    if k == 0:
        return 0.0
    return sum(labels[:k]) / k


# ═══════════════════════════════════════════════════════════════════════
#  Single-query evaluation
# ═══════════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 10]


def evaluate_single_query(
    item: Dict,
    pipeline: RAGPipeline,
    top_k: int,
    use_reranker: bool = False,
) -> Dict[str, Any]:
    """Retrieve results for one question and compute all metrics."""

    question = item["question"]
    source_document = item.get("source_document", "")

    t0 = time.time()
    try:
        results = pipeline.search(question, top_k=top_k, use_reranker=use_reranker)
    except Exception as e:
        ts_print(f"  ERROR searching '{question[:40]}...': {e}")
        results = []
    latency = time.time() - t0

    labels = relevance_labels(results, source_document)

    metrics = {"latency": latency, "num_retrieved": len(results)}
    for k in K_VALUES:
        if k <= top_k:
            metrics[f"mrr@{k}"] = mrr_at_k(labels, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
            metrics[f"hit@{k}"] = hit_at_k(labels, k)
            metrics[f"p@{k}"] = precision_at_k(labels, k)

    # Also record top-1 score for reference
    metrics["top1_score"] = results[0]["score"] if results else 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Experiment runner
# ═══════════════════════════════════════════════════════════════════════

# Reranker configurations (same as evaluate_reranker.py)
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


def run_experiment(
    name: str,
    testset: List[Dict],
    top_k: int,
    reranker_config: Optional[Dict],
    workers: int = 5,
) -> Dict[str, Any]:
    """Run one experiment (baseline / reranker) on the full testset."""

    ts_print(f"\n{'='*70}")
    ts_print(f"Experiment: {name}")
    ts_print(f"{'='*70}")

    # Build pipeline
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
    )

    use_reranker = False
    if reranker_config is not None:
        from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
        pipeline.reranker = SiliconFlowReranker.from_config_dict(reranker_config)
        pipeline.reranker_config = reranker_config["reranker"]
        use_reranker = True
        ts_print(f"  Reranker: {reranker_config['reranker']['model']}")
    else:
        ts_print("  Mode: Baseline (vector search only)")

    ts_print(f"  Top-K: {top_k}  |  Workers: {workers}  |  Questions: {len(testset)}")

    # Parallel evaluation
    all_metrics: List[Dict] = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_query, q, pipeline, top_k, use_reranker
            ): idx
            for idx, q in enumerate(testset)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                m = future.result(timeout=180)
                all_metrics.append((idx, m))
            except Exception as e:
                ts_print(f"  Q{idx} failed: {e}")
                all_metrics.append((idx, {}))
            done += 1
            if done % 10 == 0 or done == len(testset):
                ts_print(f"  Progress: {done}/{len(testset)}")

    # Sort by original index
    all_metrics.sort(key=lambda x: x[0])
    metrics_list = [m for _, m in all_metrics if m]

    total_time = time.time() - t_start

    # ── Aggregate ──────────────────────────────────────────────────────
    n = len(metrics_list)
    agg: Dict[str, Any] = {
        "experiment": name,
        "top_k": top_k,
        "num_queries": n,
        "total_time_s": round(total_time, 2),
        "avg_latency_s": 0.0,
    }

    # Latency
    latencies = [m["latency"] for m in metrics_list if "latency" in m]
    if latencies:
        agg["avg_latency_s"] = round(sum(latencies) / len(latencies), 4)

    # Top-1 score
    scores = [m["top1_score"] for m in metrics_list if "top1_score" in m]
    if scores:
        agg["avg_top1_score"] = round(sum(scores) / len(scores), 4)

    # Per-K metrics
    for k in K_VALUES:
        if k > top_k:
            continue
        for metric_name in [f"mrr@{k}", f"ndcg@{k}", f"hit@{k}", f"p@{k}"]:
            vals = [m[metric_name] for m in metrics_list if metric_name in m]
            if vals:
                agg[f"avg_{metric_name}"] = round(sum(vals) / len(vals), 4)

    ts_print(f"  Done in {total_time:.1f}s")
    return {"aggregate": agg, "per_query": metrics_list}


# ═══════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_table(results: List[Dict], top_k: int):
    """Print a side-by-side comparison table."""

    header = f"{'Metric':<14}"
    for r in results:
        name = r["aggregate"]["experiment"]
        header += f" {name:>20}"
    ts_print(header)
    ts_print("-" * (14 + 21 * len(results)))

    for k in K_VALUES:
        if k > top_k:
            continue
        for metric in ["mrr", "ndcg", "hit", "p"]:
            key = f"avg_{metric}@{k}"
            row = f"{metric.upper()}@{k:<8}"
            for r in results:
                val = r["aggregate"].get(key, float("nan"))
                row += f" {val:>20.4f}"
            ts_print(row)
        ts_print("")

    # Latency
    row = f"{'Latency(s)':<14}"
    for r in results:
        val = r["aggregate"].get("avg_latency_s", 0)
        row += f" {val:>20.4f}"
    ts_print(row)

    # Top-1 score
    row = f"{'Top1 Score':<14}"
    for r in results:
        val = r["aggregate"].get("avg_top1_score", 0)
        row += f" {val:>20.4f}"
    ts_print(row)


def compute_improvements(results: List[Dict], top_k: int) -> Dict:
    """Compute % improvement of rerankers over baseline."""
    if len(results) < 2:
        return {}

    baseline_agg = results[0]["aggregate"]
    improvements = {}

    for r in results[1:]:
        name = r["aggregate"]["experiment"]
        exp_agg = r["aggregate"]
        diffs = {}
        for k in K_VALUES:
            if k > top_k:
                continue
            for metric in ["mrr", "ndcg", "hit", "p"]:
                key = f"avg_{metric}@{k}"
                base_val = baseline_agg.get(key, 0)
                exp_val = exp_agg.get(key, 0)
                if base_val > 0:
                    pct = (exp_val - base_val) / base_val * 100
                else:
                    pct = 0.0 if exp_val == 0 else float("inf")
                diffs[key] = round(pct, 2)
        improvements[name] = diffs

    return improvements


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reranker evaluation with MRR / NDCG / Hit / Precision"
    )
    parser.add_argument(
        "--testset",
        default="rag_eval/evals/datasets/ragas_testset_doc_51q.json",
        help="Testset JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve (default 10, up from 5 for 3422 chunks)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Concurrent workers (default 5)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline", "bge_reranker", "qwen3_reranker"],
        choices=["baseline", "bge_reranker", "qwen3_reranker"],
        help="Which experiments to run",
    )
    parser.add_argument(
        "--output-dir",
        default="rag_eval/evals/experiments",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load testset
    with open(args.testset, "r", encoding="utf-8") as f:
        data = json.load(f)
    testset = data["questions"]

    ts_print(f"Loaded {len(testset)} questions from {args.testset}")
    ts_print(f"Top-K: {args.top_k}  |  Workers: {args.workers}")
    ts_print(f"Experiments: {args.experiments}")
    ts_print(f"K values for metrics: {[k for k in K_VALUES if k <= args.top_k]}")

    # Run experiments
    all_results = []
    for exp_name in args.experiments:
        config = RERANKER_CONFIGS[exp_name]
        result = run_experiment(
            name=exp_name,
            testset=testset,
            top_k=args.top_k,
            reranker_config=config,
            workers=args.workers,
        )
        all_results.append(result)

        # Save individual results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-experiment detail
        detail_path = out_dir / f"retrieval_metrics_{exp_name}_top{args.top_k}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ts_print(f"Saved: {detail_path}")

    # ── Comparison report ──────────────────────────────────────────────
    ts_print(f"\n{'='*70}")
    ts_print("COMPARISON REPORT")
    ts_print(f"{'='*70}")

    print_comparison_table(all_results, args.top_k)

    # Improvements
    improvements = compute_improvements(all_results, args.top_k)
    if improvements:
        ts_print(f"\n{'='*70}")
        ts_print("IMPROVEMENT OVER BASELINE (%)")
        ts_print(f"{'='*70}")
        for reranker_name, diffs in improvements.items():
            ts_print(f"\n  {reranker_name}:")
            for key, pct in diffs.items():
                sign = "+" if pct >= 0 else ""
                ts_print(f"    {key}: {sign}{pct:.2f}%")

    # Save comparison report
    report = {
        "timestamp": datetime.now().isoformat(),
        "testset": args.testset,
        "top_k": args.top_k,
        "workers": args.workers,
        "num_questions": len(testset),
        "experiments": {
            r["aggregate"]["experiment"]: r["aggregate"] for r in all_results
        },
        "improvements_over_baseline": improvements,
    }

    report_path = out_dir / f"retrieval_comparison_top{args.top_k}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ts_print(f"\nComparison report saved: {report_path}")

    # CSV export for easy analysis
    try:
        import pandas as pd

        rows = []
        for r in all_results:
            agg = r["aggregate"]
            for pq in r["per_query"]:
                row = {"experiment": agg["experiment"]}
                row.update(pq)
                rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = out_dir / f"retrieval_detailed_top{args.top_k}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ts_print(f"Detailed CSV saved: {csv_path}")
    except ImportError:
        ts_print("pandas not available, skipping CSV export")

    return 0


if __name__ == "__main__":
    sys.exit(main())