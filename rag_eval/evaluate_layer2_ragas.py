# -*- coding: utf-8 -*-
"""
Layer 2 — Context Quality Metrics (LLM-as-Judge via Ragas)

Uses Ragas framework to evaluate the *quality* of retrieved context:
  - Context Precision: are relevant chunks ranked higher than irrelevant ones?
  - Context Recall:    does the context cover all information in ground_truth?
  - F1 Score:          harmonic mean of precision and recall

Supports baseline vs reranker comparison with 5 concurrent workers.

Usage:
    python rag_eval/evaluate_layer2_ragas.py
    python rag_eval/evaluate_layer2_ragas.py --use-reranker bge_reranker
    python rag_eval/evaluate_layer2_ragas.py --use-reranker qwen3_reranker
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import Dataset as HFDataset
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.pipeline import RAGPipeline

# ─── Config ───────────────────────────────────────────────────────────
API_KEY = "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

_print_lock = threading.Lock()
_ragas_lock = threading.Lock()


def ts_print(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] {msg}", flush=True)


# ─── Reranker configs ─────────────────────────────────────────────────
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


def build_pipeline(reranker_name: str) -> tuple:
    """Build a pipeline with the specified reranker. Returns (pipeline, use_reranker)."""
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
    )
    use_reranker = False
    config = RERANKER_CONFIGS.get(reranker_name)

    if config is not None:
        from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
        pipeline.reranker = SiliconFlowReranker.from_config_dict(config)
        pipeline.reranker_config = config["reranker"]
        use_reranker = True
        ts_print(f"  Reranker: {config['reranker']['model']}")
    else:
        ts_print("  Mode: Baseline (vector search only)")

    return pipeline, use_reranker


def evaluate_single(
    idx: int,
    item: Dict,
    pipeline: RAGPipeline,
    client: OpenAI,
    top_k: int,
    use_reranker: bool,
) -> Dict[str, Any]:
    """Evaluate one question with Ragas context_precision + context_recall."""
    question = item["question"]
    ground_truth = item["ground_truth"]

    try:
        # Search
        results = pipeline.search(question, top_k=top_k, use_reranker=use_reranker)
        contexts = [doc["text"] for doc in results]

        if not contexts:
            return {"idx": idx, "context_precision": 0.0, "context_recall": 0.0,
                    "error": "no contexts"}

        # Use first context as proxy answer
        answer = contexts[0][:500]

        # Ragas evaluation (serialized for thread safety)
        with _ragas_lock:
            from ragas import evaluate
            from ragas.metrics import context_recall, context_precision
            from ragas.llms import llm_factory

            llm = llm_factory(
                model=MODEL_NAME,
                provider="openai",
                client=client,
                max_tokens=8192,
                timeout=120,
            )

            sample_ds = HFDataset.from_list([{
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }])

            result = evaluate(
                dataset=sample_ds,
                metrics=[context_precision, context_recall],
                llm=llm,
                raise_exceptions=True,
            )

        df = result.to_pandas()
        row = df.iloc[0]
        cp = float(row.get("context_precision", 0))
        cr = float(row.get("context_recall", 0))

        return {"idx": idx, "context_precision": cp, "context_recall": cr}

    except Exception as e:
        return {"idx": idx, "context_precision": 0.0, "context_recall": 0.0,
                "error": str(e)[:200]}


def run_layer2(
    testset: List[Dict],
    top_k: int,
    reranker_name: str,
    workers: int = 5,
) -> Dict[str, Any]:
    """Run Layer 2 evaluation for one experiment."""

    ts_print(f"\n{'='*70}")
    ts_print(f"[Layer 2] Experiment: {reranker_name}")
    ts_print(f"{'='*70}")

    pipeline, use_reranker = build_pipeline(reranker_name)
    ts_print(f"  Top-K: {top_k}  |  Workers: {workers}  |  Questions: {len(testset)}")

    # Set env for Ragas
    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_BASE_URL"] = BASE_URL
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    all_results: List[Dict] = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_single, idx, q, pipeline, client, top_k, use_reranker): idx
            for idx, q in enumerate(testset)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result(timeout=300)
                all_results.append(result)
            except Exception as e:
                ts_print(f"  Q{idx} failed: {e}")
                all_results.append({"idx": idx, "context_precision": 0.0,
                                    "context_recall": 0.0, "error": str(e)})
            done += 1
            if done % 5 == 0 or done == len(testset):
                ts_print(f"  Progress: {done}/{len(testset)}")

    all_results.sort(key=lambda x: x["idx"])
    total_time = time.time() - t_start

    # ── Aggregate ──────────────────────────────────────────────────────
    valid = [r for r in all_results if "error" not in r]
    n_valid = len(valid)

    avg_cp = sum(r["context_precision"] for r in valid) / n_valid if n_valid else 0
    avg_cr = sum(r["context_recall"] for r in valid) / n_valid if n_valid else 0
    f1 = 2 * avg_cp * avg_cr / (avg_cp + avg_cr) if (avg_cp + avg_cr) > 0 else 0

    agg = {
        "experiment": reranker_name,
        "layer": 2,
        "num_queries": len(all_results),
        "successful": n_valid,
        "total_time_s": round(total_time, 2),
        "context_precision": round(avg_cp, 4),
        "context_recall": round(avg_cr, 4),
        "f1_score": round(f1, 4),
    }

    ts_print(f"  Done in {total_time:.1f}s | Success: {n_valid}/{len(all_results)}")
    ts_print(f"  Ctx Precision: {avg_cp:.4f}  |  Ctx Recall: {avg_cr:.4f}  |  F1: {f1:.4f}")

    return {"aggregate": agg, "per_query": all_results}


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Layer 2: Context quality (Ragas)")
    parser.add_argument("--testset", default="rag_eval/evals/datasets/ragas_testset_doc_51q.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--experiments", nargs="+", default=["baseline", "bge_reranker"],
                        choices=["baseline", "bge_reranker", "qwen3_reranker"])
    parser.add_argument("--output-dir", default="rag_eval/evals/experiments")

    args = parser.parse_args()

    with open(args.testset, "r", encoding="utf-8") as f:
        testset = json.load(f)["questions"]

    ts_print(f"Layer 2 — Context Quality (Ragas)")
    ts_print(f"Loaded {len(testset)} questions | Top-K: {args.top_k} | Workers: {args.workers}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for exp_name in args.experiments:
        result = run_layer2(testset, args.top_k, exp_name, args.workers)
        all_results.append(result)

        path = out_dir / f"layer2_{exp_name}_top{args.top_k}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ts_print(f"Saved: {path}")

    # ── Comparison table ───────────────────────────────────────────────
    ts_print(f"\n{'='*70}")
    ts_print("LAYER 2 COMPARISON — Context Quality (LLM-as-Judge)")
    ts_print(f"{'='*70}")

    header = f"{'Metric':<22}"
    for r in all_results:
        header += f" {r['aggregate']['experiment']:>20}"
    ts_print(header)
    ts_print("-" * (22 + 21 * len(all_results)))

    for metric, label in [("context_precision", "Context Precision"),
                          ("context_recall", "Context Recall"),
                          ("f1_score", "F1 Score")]:
        row = f"{label:<22}"
        for r in all_results:
            val = r["aggregate"].get(metric, 0)
            row += f" {val:>20.4f}"
        ts_print(row)

    # Improvements
    if len(all_results) >= 2:
        ts_print(f"\n{'='*70}")
        ts_print("IMPROVEMENT OVER BASELINE (%)")
        ts_print(f"{'='*70}")
        base = all_results[0]["aggregate"]
        for r in all_results[1:]:
            name = r["aggregate"]["experiment"]
            ts_print(f"\n  {name}:")
            for m in ["context_precision", "context_recall", "f1_score"]:
                base_v = base.get(m, 0)
                exp_v = r["aggregate"].get(m, 0)
                if base_v > 0:
                    pct = (exp_v - base_v) / base_v * 100
                    ts_print(f"    {m}: {'+' if pct >= 0 else ''}{pct:.2f}%")

    report = {
        "layer": 2,
        "description": "Context quality via LLM-as-Judge (Ragas context_precision + context_recall)",
        "timestamp": datetime.now().isoformat(),
        "top_k": args.top_k,
        "num_questions": len(testset),
        "experiments": {r["aggregate"]["experiment"]: r["aggregate"] for r in all_results},
    }
    report_path = out_dir / f"layer2_comparison_top{args.top_k}.json"
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
        csv_path = out_dir / f"layer2_detailed_top{args.top_k}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ts_print(f"CSV: {csv_path}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
