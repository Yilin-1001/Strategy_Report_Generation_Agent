# -*- coding: utf-8 -*-
"""
Hybrid Retrieval Comprehensive Evaluation.

Compares 4 retrieval configurations on the SAME hybrid collection:
  1. Dense-only       — Pure vector search on hybrid collection
  2. Hybrid RRF       — Dense + BM25 with Reciprocal Rank Fusion
  3. Hybrid + BGE     — Hybrid + BGE-Reranker-v2-m3
  4. Hybrid + Qwen3   — Hybrid + Qwen3-Reranker-0.6B

All experiments use enterprise_docs_hybrid collection (3686 chunks).

Metrics (traditional IR metrics, no LLM required):
  - MRR@K     (Mean Reciprocal Rank)
  - NDCG@K    (Normalized Discounted Cumulative Gain)
  - Hit@K     (Hit Rate)
  - P@K       (Precision@K)
  - MAP@K     (Mean Average Precision)
  - Latency   (average query latency)
  - Top-1 Score distribution

Relevance criterion: a retrieved chunk is relevant if its source filename
matches the question's source_document field in the testset.

Usage:
    python rag_eval/evaluate_hybrid_retrieval.py
    python rag_eval/evaluate_hybrid_retrieval.py --top-k 10 --workers 3
    python rag_eval/evaluate_hybrid_retrieval.py --experiments dense_only hybrid_rrf hybrid_bge
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

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.embeddings.embedding_client import EmbeddingClient
from rag_project.storage.milvus_hybrid_manager import MilvusHybridManager
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger

_print_lock = threading.Lock()


def ts_print(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  IR Metrics
# ═══════════════════════════════════════════════════════════════════════

def relevance_labels(results: List[Dict], source_document: str) -> List[int]:
    """Binary relevance: 1 if source matches, 0 otherwise.

    Uses flexible matching: checks if the result's source filename
    ends with the source_document filename.
    """
    labels = []
    target = source_document.strip()
    for r in results:
        src = r.get("metadata", {}).get("source", "")
        # Flexible match: filename only (handles both short and full paths)
        src_filename = Path(src).name if src else ""
        labels.append(1 if src_filename == target or src == target else 0)
    return labels


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


K_VALUES = [1, 3, 5, 10]


# ═══════════════════════════════════════════════════════════════════════
#  Reranker Configs
# ═══════════════════════════════════════════════════════════════════════

RERANKER_CONFIGS = {
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

EXPERIMENT_CONFIGS = {
    "dense_only": {
        "type": "dense",
        "reranker": None,
        "label": "Dense-only (Baseline)",
    },
    "hybrid_rrf": {
        "type": "hybrid",
        "reranker": None,
        "ranker": "rrf",
        "label": "Hybrid Dense+BM25 (RRF)",
    },
    "hybrid_bge": {
        "type": "hybrid",
        "reranker": RERANKER_CONFIGS["bge_reranker"],
        "ranker": "rrf",
        "label": "Hybrid + BGE-Reranker",
    },
    "hybrid_qwen3": {
        "type": "hybrid",
        "reranker": RERANKER_CONFIGS["qwen3_reranker"],
        "ranker": "rrf",
        "label": "Hybrid + Qwen3-Reranker",
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Search Functions (all use same hybrid collection)
# ═══════════════════════════════════════════════════════════════════════

class UnifiedSearcher:
    """All searches use the same hybrid collection for fair comparison."""

    def __init__(self, search_type: str = "dense",
                 reranker_config: Optional[Dict] = None,
                 ranker: str = "rrf"):
        self.search_type = search_type
        self.embedder = EmbeddingClient("config/milvus_config.yaml")
        self.milvus_mgr = MilvusHybridManager(
            config_path="config/milvus_config.yaml",
        )

        # Reranker setup
        self.reranker = None
        self.reranker_config = {}
        self.use_reranker = False

        if reranker_config is not None:
            from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
            self.reranker = SiliconFlowReranker.from_config_dict(reranker_config)
            self.reranker_config = reranker_config["reranker"]
            self.use_reranker = True

    def search(self, query: str, top_k: int) -> Tuple[List[Dict], float]:
        t0 = time.time()

        # Embed query
        query_vector = self.embedder.embed_text(query).tolist()

        if self.search_type == "dense":
            # Dense-only search
            results = self.milvus_mgr.search_dense_only(
                query_vector=query_vector,
                top_k=top_k,
            )
        else:
            # Hybrid search (dense + BM25)
            expansion = 1
            if self.use_reranker and self.reranker:
                expansion = self.reranker_config.get("expansion_factor", 3)
            retrieve_k = top_k * expansion

            results = self.milvus_mgr.search(
                query_vector=query_vector,
                query_text=query,
                top_k=retrieve_k,
            )

            # Apply reranker
            if self.use_reranker and self.reranker:
                results = self.reranker.rerank(query, results, top_k=top_k)

        latency = time.time() - t0
        return results, latency


def create_searcher(config: Dict) -> UnifiedSearcher:
    return UnifiedSearcher(
        search_type=config["type"],
        reranker_config=config.get("reranker"),
        ranker=config.get("ranker", "rrf"),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Single-query evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_single_query(
    item: Dict,
    searcher: UnifiedSearcher,
    top_k: int,
) -> Dict[str, Any]:
    question = item["question"]
    source_document = item.get("source_document", "")

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
        "num_relevant_in_topk": sum(labels),
    }

    for k in K_VALUES:
        if k > top_k:
            continue
        metrics[f"mrr@{k}"] = mrr_at_k(labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
        metrics[f"hit@{k}"] = hit_at_k(labels, k)
        metrics[f"p@{k}"] = precision_at_k(labels, k)
        metrics[f"map@{k}"] = ap_at_k(labels, k)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Experiment Runner
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(
    name: str, config: Dict, testset: List[Dict],
    top_k: int, workers: int = 3,
) -> Dict[str, Any]:
    label = config["label"]
    ts_print(f"\n{'='*70}")
    ts_print(f"Experiment: {name} ({label})")
    ts_print(f"{'='*70}")

    searcher = create_searcher(config)
    ts_print(f"  Top-K: {top_k}  |  Workers: {workers}  |  Questions: {len(testset)}")

    all_metrics = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_single_query, q, searcher, top_k): idx
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

    all_metrics.sort(key=lambda x: x[0])
    metrics_list = [m for _, m in all_metrics if m]
    total_time = time.time() - t_start

    # Aggregate
    n = len(metrics_list)
    agg = {
        "experiment": name, "label": label,
        "top_k": top_k, "num_queries": n,
        "total_time_s": round(total_time, 2),
        "avg_latency_s": 0.0,
    }

    latencies = [m["latency"] for m in metrics_list if "latency" in m]
    if latencies:
        agg["avg_latency_s"] = round(sum(latencies) / len(latencies), 4)

    scores = [m["top1_score"] for m in metrics_list if "top1_score" in m]
    if scores:
        agg["avg_top1_score"] = round(sum(scores) / len(scores), 4)
        agg["score_distribution"] = {
            "high_gte_0.8": sum(1 for s in scores if s >= 0.8),
            "mid_0.5_0.8": sum(1 for s in scores if 0.5 <= s < 0.8),
            "low_lt_0.5": sum(1 for s in scores if s < 0.5),
        }

    relevant_counts = [m.get("num_relevant_in_topk", 0) for m in metrics_list]
    agg["avg_relevant_in_topk"] = round(sum(relevant_counts) / len(relevant_counts), 4) if relevant_counts else 0

    for k in K_VALUES:
        if k > top_k:
            continue
        for metric_name in [f"mrr@{k}", f"ndcg@{k}", f"hit@{k}", f"p@{k}", f"map@{k}"]:
            vals = [m[metric_name] for m in metrics_list if metric_name in m]
            if vals:
                agg[f"avg_{metric_name}"] = round(sum(vals) / len(vals), 4)

    ts_print(f"  Done in {total_time:.1f}s  |  Avg latency: {agg['avg_latency_s']:.4f}s")
    return {"aggregate": agg, "per_query": metrics_list}


# ═══════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_table(results: List[Dict], top_k: int):
    names = [r["aggregate"]["experiment"] for r in results]
    col_w = 16

    header = f"{'Metric':<14}"
    for name in names:
        header += f" {name:>{col_w}}"
    ts_print(header)
    ts_print("-" * (14 + (col_w + 1) * len(names)))

    for k in K_VALUES:
        if k > top_k:
            continue
        for metric in ["mrr", "ndcg", "hit", "p", "map"]:
            key = f"avg_{metric}@{k}"
            row = f"{metric.upper()}@{k:<8}"
            for r in results:
                val = r["aggregate"].get(key, float("nan"))
                row += f" {val:>{col_w}.4f}"
            ts_print(row)
        ts_print("")

    for label, key in [("Latency(s)", "avg_latency_s"), ("Top1 Score", "avg_top1_score"), ("Avg Rel@K", "avg_relevant_in_topk")]:
        row = f"{label:<14}"
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
        keys = ["avg_latency_s", "avg_top1_score", "avg_relevant_in_topk"]
        for k in K_VALUES:
            if k > top_k:
                continue
            for metric in ["mrr", "ndcg", "hit", "p", "map"]:
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
    col_w = 16
    header = f"{'Metric':<14}"
    for name in names:
        header += f" {name:>{col_w}}"
    ts_print(header)
    ts_print("-" * (14 + (col_w + 1) * len(names)))

    for k in K_VALUES:
        if k > top_k:
            continue
        for metric in ["mrr", "ndcg", "hit", "p", "map"]:
            key = f"avg_{metric}@{k}"
            row = f"{metric.upper()}@{k:<8}"
            for name in names:
                pct = improvements[name].get(key, 0)
                sign = "+" if pct >= 0 else ""
                row += f" {sign}{pct:>{col_w - 2}.2f}%"
            ts_print(row)
        ts_print("")

    for label, key in [("Latency", "avg_latency_s"), ("Top1 Score", "avg_top1_score")]:
        row = f"{label:<14}"
        for name in names:
            pct = improvements[name].get(key, 0)
            sign = "+" if pct >= 0 else ""
            row += f" {sign}{pct:>{col_w - 2}.2f}%"
        ts_print(row)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval evaluation")
    parser.add_argument("--testset", default="rag_eval/evals/datasets/ragas_testset_doc_51q.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--experiments", nargs="+",
                        default=["dense_only", "hybrid_rrf", "hybrid_bge", "hybrid_qwen3"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--output-dir", default="rag_eval/evals/experiments")

    args = parser.parse_args()

    with open(args.testset, "r", encoding="utf-8") as f:
        data = json.load(f)
    testset = data["questions"]

    ts_print(f"{'='*70}")
    ts_print(f"Hybrid Retrieval Evaluation (All on enterprise_docs_hybrid)")
    ts_print(f"{'='*70}")
    ts_print(f"Testset: {args.testset} ({len(testset)} questions)")
    ts_print(f"Top-K: {args.top_k}  |  Workers: {args.workers}")
    ts_print(f"Experiments: {args.experiments}")

    all_results = []
    for exp_name in args.experiments:
        config = EXPERIMENT_CONFIGS[exp_name]
        result = run_experiment(exp_name, config, testset, args.top_k, args.workers)
        all_results.append(result)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"hybrid_eval_{exp_name}_top{args.top_k}.json"
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
        "top_k": args.top_k,
        "workers": args.workers,
        "num_questions": len(testset),
        "k_values": [k for k in K_VALUES if k <= args.top_k],
        "collection": "enterprise_docs_hybrid",
        "experiments": {r["aggregate"]["experiment"]: r["aggregate"] for r in all_results},
        "improvements_over_baseline": improvements,
    }
    report_path = out_dir / f"hybrid_comparison_top{args.top_k}.json"
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
        csv_path = out_dir / f"hybrid_detailed_top{args.top_k}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ts_print(f"CSV saved: {csv_path}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
