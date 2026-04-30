# -*- coding: utf-8 -*-
"""
Layer 1 — Retrieval Ranking Metrics (No LLM)

Compares baseline vs reranker on standard IR metrics:
  - MRR@K    (Mean Reciprocal Rank)
  - NDCG@K   (Normalized Discounted Cumulative Gain)
  - Hit@K    (Hit Rate)
  - P@K      (Precision@K)

Relevance criterion: **Embedding semantic similarity** between each retrieved
chunk and the question's ground_truth answer.

Uses the same BGE-M3 embedding model (via SiliconFlow API) already used for
indexing, so relevance scores are consistent with the retrieval model.

Usage:
    python rag_eval/evaluate_layer1_ranking.py --top-k 10
    python rag_eval/evaluate_layer1_ranking.py --experiments baseline bge_reranker
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
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.pipeline import RAGPipeline
from rag_project.embeddings.embedding_client import EmbeddingClient

# ─── Thread-safe helpers ──────────────────────────────────────────────
_print_lock = threading.Lock()


def ts_print(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Semantic relevance via embedding cosine similarity
# ═══════════════════════════════════════════════════════════════════════

class SemanticRelevanceScorer:
    """Compute graded relevance between retrieved chunks and ground_truth."""

    def __init__(self, config_path: str = "config/milvus_config.yaml"):
        self.client = EmbeddingClient(config_path)
        # Cache ground_truth embeddings to avoid re-computing
        self._cache: Dict[int, np.ndarray] = {}

    def embed_ground_truth(self, idx: int, text: str) -> np.ndarray:
        if idx not in self._cache:
            self._cache[idx] = self.client.embed_text(text)
        return self._cache[idx]

    def relevance_scores(
        self,
        results: List[Dict],
        ground_truth: str,
        gt_idx: int,
        threshold: float = 0.5,
    ) -> List[float]:
        """Return graded relevance scores for each retrieved result.

        Uses cosine similarity between chunk text and ground_truth.
        Values below *threshold* are set to 0 (not relevant).
        """
        if not results:
            return []

        gt_vec = self.embed_ground_truth(gt_idx, ground_truth)
        gt_norm = np.linalg.norm(gt_vec)

        chunk_texts = [r["text"] for r in results]
        try:
            chunk_vecs = self.client.embed_texts(chunk_texts)
        except Exception as e:
            ts_print(f"  Embedding error: {e}")
            return [0.0] * len(results)

        scores = []
        for vec in chunk_vecs:
            v_norm = np.linalg.norm(vec)
            if v_norm == 0 or gt_norm == 0:
                scores.append(0.0)
            else:
                sim = float(np.dot(gt_vec, vec) / (gt_norm * v_norm))
                scores.append(sim if sim >= threshold else 0.0)
        return scores


# ═══════════════════════════════════════════════════════════════════════
#  IR metrics (support graded relevance)
# ═══════════════════════════════════════════════════════════════════════

def mrr_at_k(labels: List[float], k: int) -> float:
    """MRR@K with graded relevance — uses first non-zero score."""
    for i in range(min(k, len(labels))):
        if labels[i] > 0:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(labels: List[float], k: int) -> float:
    val = 0.0
    for i in range(min(k, len(labels))):
        if labels[i] > 0:
            val += labels[i] / math.log2(i + 2)
    return val


def ndcg_at_k(labels: List[float], k: int) -> float:
    num_relevant = sum(1 for l in labels if l > 0)
    if num_relevant == 0:
        return 0.0
    dcg = dcg_at_k(labels, k)
    ideal = sorted([l for l in labels if l > 0], reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(labels: List[float], k: int) -> float:
    return 1.0 if any(l > 0 for l in labels[:k]) else 0.0


def precision_at_k(labels: List[float], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for l in labels[:k] if l > 0) / k


# ═══════════════════════════════════════════════════════════════════════
#  Single-query evaluation
# ═══════════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 10]


def evaluate_single_query(
    idx: int,
    item: Dict,
    pipeline: RAGPipeline,
    scorer: SemanticRelevanceScorer,
    top_k: int,
    use_reranker: bool = False,
) -> Dict[str, Any]:
    question = item["question"]
    ground_truth = item["ground_truth"]

    t0 = time.time()
    try:
        results = pipeline.search(question, top_k=top_k, use_reranker=use_reranker)
    except Exception as e:
        ts_print(f"  Q{idx} search error: {e}")
        results = []
    search_latency = time.time() - t0

    # Semantic relevance scoring
    t1 = time.time()
    labels = scorer.relevance_scores(results, ground_truth, idx)
    embed_latency = time.time() - t1

    metrics = {
        "latency": search_latency,
        "embed_latency": embed_latency,
        "total_latency": search_latency + embed_latency,
        "num_retrieved": len(results),
        "top1_score": results[0]["score"] if results else 0.0,
        "avg_relevance": sum(labels) / len(labels) if labels else 0.0,
    }

    for k in K_VALUES:
        if k <= top_k:
            metrics[f"mrr@{k}"] = mrr_at_k(labels, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k)
            metrics[f"hit@{k}"] = hit_at_k(labels, k)
            metrics[f"p@{k}"] = precision_at_k(labels, k)

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Experiment runner
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    # ── Dense-only experiments (original pipeline) ──
    "baseline": {"type": "dense", "reranker": None},
    "bge_reranker": {
        "type": "dense",
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
    # ── Hybrid experiments (BM25 + Dense via MilvusHybridManager) ──
    "hybrid_baseline": {"type": "hybrid", "reranker": None},
    "hybrid_bge_reranker": {
        "type": "hybrid",
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
}


def run_experiment(
    name: str,
    testset: List[Dict],
    top_k: int,
    workers: int = 5,
) -> Dict[str, Any]:
    ts_print(f"\n{'='*70}")
    ts_print(f"[Layer 1] Experiment: {name}")
    ts_print(f"{'='*70}")

    exp_config = EXPERIMENT_CONFIGS[name]
    exp_type = exp_config["type"]
    reranker_config = exp_config.get("reranker")

    # Build the right pipeline
    if exp_type == "hybrid":
        from rag_project.pipeline_hybrid import HybridRAGPipeline
        pipeline = HybridRAGPipeline(
            chunking_config_path="config/chunking_config.yaml",
            milvus_config_path="config/milvus_config.yaml",
        )
        ts_print(f"  Mode: Hybrid (BM25 + Dense)")
    else:
        pipeline = RAGPipeline(
            chunking_config_path="config/chunking_config.yaml",
            milvus_config_path="config/milvus_config.yaml",
        )
        ts_print("  Mode: Dense (vector search only)")

    use_reranker = False
    if reranker_config is not None:
        from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
        pipeline.reranker = SiliconFlowReranker.from_config_dict({"reranker": reranker_config})
        pipeline.reranker_config = reranker_config
        use_reranker = True
        ts_print(f"  Reranker: {reranker_config['model']}")

    scorer = SemanticRelevanceScorer()
    ts_print(f"  Top-K: {top_k}  |  Workers: {workers}  |  Questions: {len(testset)}")
    ts_print(f"  Relevance: embedding cosine similarity (threshold=0.5)")

    all_metrics: List[tuple] = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_query, idx, q, pipeline, scorer, top_k, use_reranker
            ): idx
            for idx, q in enumerate(testset)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                m = future.result(timeout=300)
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

    # ── Aggregate ──────────────────────────────────────────────────────
    n = len(metrics_list)
    agg: Dict[str, Any] = {
        "experiment": name,
        "relevance_method": "embedding_cosine_similarity",
        "top_k": top_k,
        "num_queries": n,
        "total_time_s": round(total_time, 2),
    }

    for key in ["latency", "embed_latency", "total_latency", "top1_score", "avg_relevance"]:
        vals = [m[key] for m in metrics_list if key in m]
        if vals:
            agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4)

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
    header = f"{'Metric':<14}"
    for r in results:
        header += f" {r['aggregate']['experiment']:>20}"
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

    for label, key in [("Latency(s)", "avg_latency"), ("Top1 Score", "avg_top1_score"),
                       ("Avg Relevance", "avg_avg_relevance")]:
        row = f"{label:<14}"
        for r in results:
            val = r["aggregate"].get(key, 0)
            row += f" {val:>20.4f}"
        ts_print(row)


def compute_improvements(results: List[Dict], top_k: int) -> Dict:
    if len(results) < 2:
        return {}
    baseline_agg = results[0]["aggregate"]
    improvements = {}
    for r in results[1:]:
        name = r["aggregate"]["experiment"]
        exp_agg = r["aggregate"]
        diffs = {}
        all_keys = (
            [f"avg_{metric}@{k}" for k in K_VALUES if k <= top_k for metric in ["mrr", "ndcg", "hit", "p"]]
            + ["avg_avg_relevance"]
        )
        for key in all_keys:
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
    parser = argparse.ArgumentParser(description="Layer 1: Retrieval ranking metrics (embedding relevance)")
    parser.add_argument("--testset", default="rag_eval/evals/datasets/ragas_testset_doc_51q.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--experiments", nargs="+",
                        default=["baseline", "bge_reranker"],
                        choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--output-dir", default="rag_eval/evals/experiments")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Cosine similarity threshold for relevance (default 0.5)")

    args = parser.parse_args()

    with open(args.testset, "r", encoding="utf-8") as f:
        testset = json.load(f)["questions"]

    ts_print(f"Layer 1 — Retrieval Ranking Metrics")
    ts_print(f"Loaded {len(testset)} questions | Top-K: {args.top_k} | Workers: {args.workers}")
    ts_print(f"Relevance: embedding cosine sim (threshold={args.threshold})")

    all_results = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp_name in args.experiments:
        result = run_experiment(
            name=exp_name,
            testset=testset,
            top_k=args.top_k,
            workers=args.workers,
        )
        all_results.append(result)

        path = out_dir / f"layer1_{exp_name}_top{args.top_k}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ts_print(f"Saved: {path}")

    # ── Report ─────────────────────────────────────────────────────────
    ts_print(f"\n{'='*70}")
    ts_print("LAYER 1 COMPARISON — Ranking Quality (Embedding Relevance)")
    ts_print(f"{'='*70}")
    print_comparison_table(all_results, args.top_k)

    improvements = compute_improvements(all_results, args.top_k)
    if improvements:
        ts_print(f"\n{'='*70}")
        ts_print("IMPROVEMENT OVER BASELINE (%)")
        ts_print(f"{'='*70}")
        for name, diffs in improvements.items():
            ts_print(f"\n  {name}:")
            for key, pct in diffs.items():
                sign = "+" if pct >= 0 else ""
                ts_print(f"    {key}: {sign}{pct:.2f}%")

    report = {
        "layer": 1,
        "description": "Retrieval ranking quality (MRR, NDCG, Hit, Precision)",
        "relevance_method": "embedding_cosine_similarity",
        "threshold": args.threshold,
        "timestamp": datetime.now().isoformat(),
        "top_k": args.top_k,
        "num_questions": len(testset),
        "experiments": {r["aggregate"]["experiment"]: r["aggregate"] for r in all_results},
        "improvements_over_baseline": improvements,
    }
    report_path = out_dir / f"layer1_comparison_top{args.top_k}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    ts_print(f"\nReport: {report_path}")

    # CSV
    try:
        import pandas as pd
        rows = []
        for r in all_results:
            for pq in r["per_query"]:
                row = {"experiment": r["aggregate"]["experiment"]}
                row.update(pq)
                rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = out_dir / f"layer1_detailed_top{args.top_k}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ts_print(f"CSV: {csv_path}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
