# -*- coding: utf-8 -*-
"""
Layer 1 Retrieval Evaluation with Standard IR Metrics (Recall, Precision, F1)

Enhanced version that includes:
- Recall@K: proportion of relevant documents retrieved
- Precision@K: proportion of retrieved documents that are relevant
- F1@K: harmonic mean of precision and recall
- Uses source_document matching (document-level relevance)
- Loads total chunk counts per source for accurate recall calculation

Relevance criterion: source_document match (any chunk from same source file is relevant)

Usage:
    python rag_eval/evaluate_layer1_standard.py --testset rag_eval/evals/datasets/smart_testset_full.json --top-k 10
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
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.embeddings.embedding_client import EmbeddingClient
from rag_project.storage.milvus_hybrid_manager import MilvusHybridManager

_print_lock = threading.Lock()


def ts_print(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Load source document chunk counts for recall calculation
# ═══════════════════════════════════════════════════════════════════════

def load_chunk_counts(chunks_file: str = "data/all_chunks.json") -> Dict[str, int]:
    """Load chunk counts per source document from the chunks file."""
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        # Count chunks per source (using filename only)
        source_counts = Counter()
        for chunk in chunks:
            src = chunk.get('metadata', {}).get('source', '')
            # Extract filename
            src_name = Path(src).name if src else ''
            if src_name:
                source_counts[src_name] += 1
        return dict(source_counts)
    except Exception as e:
        ts_print(f"Warning: Could not load chunk counts: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════
#  IR Metrics (Standard - Recall, Precision, F1, MRR, NDCG, HIT)
# ═══════════════════════════════════════════════════════════════════════

def relevance_labels(results: List[Dict], source_document: str) -> List[int]:
    """Binary relevance: 1 if source matches, 0 otherwise."""
    labels = []
    target = source_document.strip()
    for r in results:
        src = r.get("metadata", {}).get("source", "")
        src_filename = Path(src).name if src else ""
        labels.append(1 if src_filename == target or src == target else 0)
    return labels


def _count_relevant(source_document: str, source_counts: Dict[str, int]) -> int:
    """Get total number of relevant chunks for this source document."""
    target = source_document.strip()
    # Try exact match
    if target in source_counts:
        return source_counts[target]
    # Try partial match (filename without extension)
    target_base = Path(target).stem
    for src, count in source_counts.items():
        if Path(src).stem == target_base:
            return count
    return 1  # Default fallback


def mrr_at_k(labels: List[int], k: int) -> float:
    for i in range(min(k, len(labels))):
        if labels[i] == 1:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(labels: List[int], k: int) -> float:
    val = 0.0
    for i in range(min(k, len(labels))):
        if labels[i] == 1:
            val += 1.0 / math.log2(i + 2)
    return val


def ndcg_at_k(labels: List[int], k: int) -> float:
    num_relevant = sum(labels)
    if num_relevant == 0:
        return 0.0
    dcg = dcg_at_k(labels, k)
    ideal = [1] * num_relevant
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(labels: List[int], k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0


def precision_at_k(labels: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(labels[:k]) / k


def recall_at_k(labels: List[int], total_relevant: int, k: int) -> float:
    """Recall@K = relevant items retrieved / total relevant items."""
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(labels[:k])
    return retrieved_relevant / total_relevant


def f1_at_k(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R)"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ap_at_k(labels: List[int], k: int) -> float:
    num_relevant = sum(labels)
    if num_relevant == 0:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i in range(min(k, len(labels))):
        if labels[i] == 1:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(num_relevant, k)


K_VALUES = [1, 3, 5, 10, 20]


# ═══════════════════════════════════════════════════════════════════════
#  Search Functions
# ═══════════════════════════════════════════════════════════════════════

class UnifiedSearcher:
    """All searches use the same hybrid collection for fair comparison."""

    def __init__(self, search_type: str = "dense"):
        self.search_type = search_type
        self.embedder = EmbeddingClient("config/milvus_config.yaml")
        self.milvus_mgr = MilvusHybridManager(config_path="config/milvus_config.yaml")

    def search(self, query: str, top_k: int) -> Tuple[List[Dict], float]:
        t0 = time.time()
        query_vector = self.embedder.embed_text(query).tolist()

        if self.search_type == "dense":
            results = self.milvus_mgr.search_dense_only(
                query_vector=query_vector,
                top_k=top_k,
            )
        else:
            # Hybrid search
            results = self.milvus_mgr.search(
                query_vector=query_vector,
                query_text=query,
                top_k=top_k,
            )

        latency = time.time() - t0
        return results, latency


# ═══════════════════════════════════════════════════════════════════════
#  Single-query evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_single_query(
    item: Dict,
    searcher: UnifiedSearcher,
    source_counts: Dict[str, int],
    top_k: int,
) -> Dict[str, Any]:
    question = item["question"]
    source_document = item.get("source_document", "")
    num_relevant = _count_relevant(source_document, source_counts)

    try:
        results, latency = searcher.search(question, top_k)
    except Exception as e:
        ts_print(f"  ERROR: {question[:40]}...: {e}")
        results = []
        latency = 0.0

    labels = relevance_labels(results, source_document)

    metrics = {
        "latency": latency,
        "num_retrieved": len(results),
        "top1_score": results[0]["score"] if results else 0.0,
        "total_relevant": num_relevant,
        "retrieved_relevant": sum(labels[:top_k]),
    }

    for k in K_VALUES:
        if k > top_k:
            continue
        p = precision_at_k(labels, k)
        r = recall_at_k(labels, num_relevant, k) if num_relevant > 0 else 0.0
        metrics[f"p@{k}"] = p
        metrics[f"r@{k}"] = r
        metrics[f"f1@{k}"] = f1_at_k(p, r)
        metrics[f"mrr@{k}"] = mrr_at_k(labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
        metrics[f"hit@{k}"] = hit_at_k(labels, k)
        metrics[f"map@{k}"] = ap_at_k(labels, k)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Experiment Runner
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    "dense_only": {
        "type": "dense",
        "label": "Dense-only (Vector Search)",
    },
    "hybrid_rrf": {
        "type": "hybrid",
        "label": "Hybrid RRF (Dense + BM25)",
    },
}


def run_experiment(
    name: str, config: Dict, testset: List[Dict],
    source_counts: Dict[str, int], top_k: int, workers: int = 5,
) -> Dict[str, Any]:
    label = config["label"]
    ts_print(f"\n{'='*70}")
    ts_print(f"[Layer 1] Experiment: {name} ({label})")
    ts_print(f"{'='*70}")

    searcher = UnifiedSearcher(search_type=config["type"])
    ts_print(f"  Top-K: {top_k}  |  Workers: {workers}  |  Questions: {len(testset)}")
    ts_print(f"  Relevance: source_document match (document-level)")

    all_metrics = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_single_query, q, searcher, source_counts, top_k): idx
            for idx, q in enumerate(testset)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                m = future.result(timeout=120)
                all_metrics.append((idx, m))
            except Exception as e:
                ts_print(f"  Q{idx} failed: {e}")
                all_metrics.append((idx, {}))
            done += 1
            if done % 50 == 0 or done == len(testset):
                ts_print(f"  Progress: {done}/{len(testset)}")

    all_metrics.sort(key=lambda x: x[0])
    metrics_list = [m for _, m in all_metrics if m]
    total_time = time.time() - t_start

    # Aggregate
    n = len(metrics_list)
    agg = {
        "experiment": name, "label": label,
        "top_k": top_k, "num_queries": n,
        "total_time_s": round(total_time, 2),
    }

    # Latency and scores
    latencies = [m["latency"] for m in metrics_list if "latency" in m]
    if latencies:
        agg["avg_latency_s"] = round(sum(latencies) / len(latencies), 4)

    scores = [m["top1_score"] for m in metrics_list if "top1_score" in m]
    if scores:
        agg["avg_top1_score"] = round(sum(scores) / len(scores), 4)

    # Recall/Precision summary
    total_rel = sum(m.get("total_relevant", 0) for m in metrics_list)
    retrieved_rel = sum(m.get("retrieved_relevant", 0) for m in metrics_list)
    agg["total_relevant_chunks"] = total_rel
    agg["retrieved_relevant_chunks"] = retrieved_rel
    agg["overall_recall"] = round(retrieved_rel / total_rel, 4) if total_rel > 0 else 0

    # All metrics
    for k in K_VALUES:
        if k > top_k:
            continue
        for metric_name in [f"p@{k}", f"r@{k}", f"f1@{k}", f"mrr@{k}", f"ndcg@{k}", f"hit@{k}", f"map@{k}"]:
            vals = [m[metric_name] for m in metrics_list if metric_name in m]
            if vals:
                agg[f"avg_{metric_name}"] = round(sum(vals) / len(vals), 4)

    ts_print(f"  Done in {total_time:.1f}s  |  Avg latency: {agg.get('avg_latency_s', 0):.4f}s")
    ts_print(f"  Overall Recall: {agg.get('overall_recall', 0):.4f}")
    return {"aggregate": agg, "per_query": metrics_list}


# ═══════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_table(results: List[Dict], top_k: int):
    names = [r["aggregate"]["experiment"] for r in results]
    col_w = 14

    header = f"{'Metric':<10}"
    for name in names:
        header += f" {name:>{col_w}}"
    ts_print(header)
    ts_print("-" * (10 + (col_w + 1) * len(names)))

    for k in K_VALUES:
        if k > top_k:
            continue
        for metric in ["p", "r", "f1", "mrr", "ndcg", "hit", "map"]:
            key = f"avg_{metric}@{k}"
            row = f"{metric.upper()}@{k:<6}"
            for r in results:
                val = r["aggregate"].get(key, float("nan"))
                row += f" {val:>{col_w}.4f}"
            ts_print(row)
        ts_print("")

    # Summary
    for label, key in [("Latency(s)", "avg_latency_s"), ("Top1 Score", "avg_top1_score"), ("Overall Recall", "overall_recall")]:
        row = f"{label:<10}"
        for r in results:
            val = r["aggregate"].get(key, 0)
            row += f" {val:>{col_w}.4f}"
        ts_print(row)


def compute_improvements(results: List[Dict], top_k: int) -> Dict:
    if len(results) < 2:
        return {}
    baseline = results[0]["aggregate"]
    improvements = {}
    for r in results[1:]:
        name = r["aggregate"]["experiment"]
        exp = r["aggregate"]
        diffs = {}
        keys = ["avg_latency_s", "avg_top1_score", "overall_recall"]
        for k in K_VALUES:
            if k > top_k:
                continue
            for metric in ["p", "r", "f1", "mrr", "ndcg", "hit", "map"]:
                keys.append(f"avg_{metric}@{k}")
        for key in keys:
            base_val = baseline.get(key, 0)
            exp_val = exp.get(key, 0)
            if base_val > 0:
                pct = (exp_val - base_val) / base_val * 100
            else:
                pct = 0.0 if exp_val == 0 else float("inf")
            diffs[key] = round(pct, 2)
        improvements[name] = diffs
    return improvements


def print_improvement_table(improvements: Dict, top_k: int):
    if not improvements:
        return
    ts_print(f"\n{'='*70}")
    ts_print("IMPROVEMENT OVER DENSE-ONLY BASELINE (%)")
    ts_print(f"{'='*70}")

    names = list(improvements.keys())
    col_w = 14
    header = f"{'Metric':<10}"
    for name in names:
        header += f" {name:>{col_w}}"
    ts_print(header)
    ts_print("-" * (10 + (col_w + 1) * len(names)))

    for k in K_VALUES:
        if k > top_k:
            continue
        for metric in ["p", "r", "f1", "mrr", "ndcg", "hit", "map"]:
            key = f"avg_{metric}@{k}"
            row = f"{metric.upper()}@{k:<6}"
            for name in names:
                pct = improvements[name].get(key, 0)
                sign = "+" if pct >= 0 else ""
                row += f" {sign}{pct:>{col_w - 2}.2f}%"
            ts_print(row)
        ts_print("")

    for label, key in [("Latency", "avg_latency_s"), ("Top1 Score", "avg_top1_score"), ("Overall Recall", "overall_recall")]:
        row = f"{label:<10}"
        for name in names:
            pct = improvements[name].get(key, 0)
            sign = "+" if pct >= 0 else ""
            row += f" {sign}{pct:>{col_w - 2}.2f}%"
        ts_print(row)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 1: Standard IR metrics (Recall, Precision, F1)")
    parser.add_argument("--testset", default="rag_eval/evals/datasets/smart_testset_full.json")
    parser.add_argument("--chunks-file", default="data/all_chunks.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--experiments", nargs="+",
                        default=["dense_only", "hybrid_rrf"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--output-dir", default="rag_eval/evals/experiments")

    args = parser.parse_args()

    # Load testset
    with open(args.testset, "r", encoding="utf-8") as f:
        data = json.load(f)
    testset = data["questions"]

    # Load source chunk counts for recall calculation
    source_counts = load_chunk_counts(args.chunks_file)
    ts_print(f"Loaded chunk counts for {len(source_counts)} source documents")

    ts_print(f"{'='*70}")
    ts_print(f"Layer 1 — Standard IR Metrics (Recall, Precision, F1)")
    ts_print(f"{'='*70}")
    ts_print(f"Testset: {args.testset} ({len(testset)} questions)")
    ts_print(f"Top-K: {args.top_k}  |  Workers: {args.workers}")
    ts_print(f"Experiments: {args.experiments}")
    ts_print(f"Relevance criterion: source_document match (document-level)")

    all_results = []
    for exp_name in args.experiments:
        config = EXPERIMENT_CONFIGS[exp_name]
        result = run_experiment(exp_name, config, testset, source_counts, args.top_k, args.workers)
        all_results.append(result)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"layer1_{exp_name}_standard_top{args.top_k}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ts_print(f"Saved: {path}")

    # Report
    ts_print(f"\n{'='*70}")
    ts_print("COMPARISON REPORT")
    ts_print(f"{'='*70}")
    print_comparison_table(all_results, args.top_k)

    improvements = compute_improvements(all_results, args.top_k)
    print_improvement_table(improvements, args.top_k)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "testset": args.testset,
        "chunks_file": args.chunks_file,
        "top_k": args.top_k,
        "workers": args.workers,
        "num_questions": len(testset),
        "k_values": [k for k in K_VALUES if k <= args.top_k],
        "relevance_criterion": "source_document_match",
        "source_documents_count": len(source_counts),
        "experiments": {r["aggregate"]["experiment"]: r["aggregate"] for r in all_results},
        "improvements_over_baseline": improvements,
    }
    report_path = out_dir / f"layer1_standard_comparison_top{args.top_k}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ts_print(f"\nReport saved: {report_path}")

    # CSV
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
        csv_path = out_dir / f"layer1_standard_detailed_top{args.top_k}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ts_print(f"CSV saved: {csv_path}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
