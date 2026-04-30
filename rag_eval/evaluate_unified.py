# -*- coding: utf-8 -*-
"""
Unified RAG Evaluation — Two-Layer Hybrid Assessment

Runs both evaluation layers and produces a single merged comparison report:

  Layer 1 — Retrieval Ranking Quality (no LLM, fast)
    MRR@K, NDCG@K, Hit@K, P@K  (embedding semantic relevance)

  Layer 2 — Context Quality (LLM-as-Judge, slow)
    Context Precision, Context Recall, F1  (Ragas)

Usage:
    python rag_eval/evaluate_unified.py
    python rag_eval/evaluate_unified.py --experiments baseline bge_reranker
    python rag_eval/evaluate_unified.py --top-k 10 --workers 5 --layer both
    python rag_eval/evaluate_unified.py --layer 1       # only Layer 1
    python rag_eval/evaluate_unified.py --layer 2       # only Layer 2
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


RERANKER_CONFIGS = ["baseline", "bge_reranker", "qwen3_reranker"]


def print_unified_report(layer1_results: dict, layer2_results: dict, experiments: list):
    """Print a unified comparison table combining both layers."""
    print(f"\n{'='*80}")
    print("  UNIFIED RAG EVALUATION REPORT")
    print(f"{'='*80}")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Experiments: {experiments}")
    print()

    # ── Layer 1 ────────────────────────────────────────────────────────
    if layer1_results:
        print(f"  {'─'*76}")
        print("  Layer 1 — Retrieval Ranking Quality (Embedding Semantic Relevance)")
        print(f"  {'─'*76}")

        exps = layer1_results.get("experiments", {})
        metrics_order = []
        for k in [1, 3, 5, 10]:
            for m in ["mrr", "ndcg", "hit", "p"]:
                metrics_order.append(f"avg_{m}@{k}")
        metrics_order += ["avg_avg_relevance", "avg_latency"]

        header = f"  {'Metric':<18}"
        for name in experiments:
            header += f" {name:>20}"
        print(header)
        print(f"  {'─'*76}")

        for key in metrics_order:
            label = key.replace("avg_", "").upper()
            row = f"  {label:<18}"
            for name in experiments:
                val = exps.get(name, {}).get(key, float("nan"))
                if isinstance(val, (int, float)):
                    row += f" {val:>20.4f}"
                else:
                    row += f" {'N/A':>20}"
            print(row)

    # ── Layer 2 ────────────────────────────────────────────────────────
    if layer2_results:
        print(f"\n  {'─'*76}")
        print("  Layer 2 — Context Quality (LLM-as-Judge via Ragas)")
        print(f"  {'─'*76}")

        exps = layer2_results.get("experiments", {})
        metrics2 = [("context_precision", "Ctx Precision"),
                    ("context_recall", "Ctx Recall"),
                    ("f1_score", "F1 Score")]

        header = f"  {'Metric':<18}"
        for name in experiments:
            header += f" {name:>20}"
        print(header)
        print(f"  {'─'*76}")

        for key, label in metrics2:
            row = f"  {label:<18}"
            for name in experiments:
                val = exps.get(name, {}).get(key, float("nan"))
                if isinstance(val, (int, float)):
                    row += f" {val:>20.4f}"
                else:
                    row += f" {'N/A':>20}"
            print(row)

    # ── Improvement summary ────────────────────────────────────────────
    if layer1_results and "improvements_over_baseline" in layer1_results:
        improvements = layer1_results["improvements_over_baseline"]
        if improvements:
            print(f"\n  {'─'*76}")
            print("  Layer 1 Improvement over Baseline (%)")
            print(f"  {'─'*76}")
            for reranker, diffs in improvements.items():
                print(f"\n  {reranker}:")
                for key, pct in diffs.items():
                    sign = "+" if pct >= 0 else ""
                    label = key.replace("avg_", "").upper()
                    print(f"    {label:<20} {sign}{pct:.2f}%")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Unified two-layer RAG evaluation")
    parser.add_argument("--testset", default="rag_eval/evals/datasets/ragas_testset_doc_51q.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--experiments", nargs="+", default=["baseline", "bge_reranker"],
                        choices=RERANKER_CONFIGS)
    parser.add_argument("--output-dir", default="rag_eval/evals/experiments")
    parser.add_argument("--layer", default="both", choices=["1", "2", "both"],
                        help="Which layer(s) to run")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer1_results = None
    layer2_results = None

    # ── Layer 1 ────────────────────────────────────────────────────────
    if args.layer in ("1", "both"):
        print("\n" + "="*80)
        print("  RUNNING LAYER 1: Retrieval Ranking Metrics")
        print("="*80)

        from rag_eval.evaluate_layer1_ranking import (
            run_experiment as run_l1,
            RERANKER_CONFIGS as L1_CONFIGS,
            compute_improvements,
            print_comparison_table,
        )
        import json

        with open(args.testset, "r", encoding="utf-8") as f:
            testset = json.load(f)["questions"]

        l1_all = []
        for exp_name in args.experiments:
            result = run_l1(
                name=exp_name,
                testset=testset,
                top_k=args.top_k,
                reranker_config=L1_CONFIGS[exp_name],
                workers=args.workers,
            )
            l1_all.append(result)

            path = out_dir / f"layer1_{exp_name}_top{args.top_k}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        improvements = compute_improvements(l1_all, args.top_k)
        layer1_results = {
            "layer": 1,
            "description": "Retrieval ranking quality (embedding cosine similarity)",
            "top_k": args.top_k,
            "num_questions": len(testset),
            "experiments": {r["aggregate"]["experiment"]: r["aggregate"] for r in l1_all},
            "improvements_over_baseline": improvements,
        }

        l1_report_path = out_dir / f"layer1_comparison_top{args.top_k}.json"
        with open(l1_report_path, "w", encoding="utf-8") as f:
            json.dump(layer1_results, f, ensure_ascii=False, indent=2)

    # ── Layer 2 ────────────────────────────────────────────────────────
    if args.layer in ("2", "both"):
        print("\n" + "="*80)
        print("  RUNNING LAYER 2: Context Quality (Ragas)")
        print("="*80)

        from rag_eval.evaluate_layer2_ragas import run_layer2

        l2_all = []
        for exp_name in args.experiments:
            with open(args.testset, "r", encoding="utf-8") as f:
                testset = json.load(f)["questions"]
            result = run_layer2(testset, args.top_k, exp_name, args.workers)
            l2_all.append(result)

            path = out_dir / f"layer2_{exp_name}_top{args.top_k}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        layer2_results = {
            "layer": 2,
            "description": "Context quality via LLM-as-Judge (Ragas)",
            "top_k": args.top_k,
            "experiments": {r["aggregate"]["experiment"]: r["aggregate"] for r in l2_all},
        }

        l2_report_path = out_dir / f"layer2_comparison_top{args.top_k}.json"
        with open(l2_report_path, "w", encoding="utf-8") as f:
            json.dump(layer2_results, f, ensure_ascii=False, indent=2)

    # ── Unified report ─────────────────────────────────────────────────
    print_unified_report(layer1_results, layer2_results, args.experiments)

    # Save unified report
    unified = {
        "timestamp": datetime.now().isoformat(),
        "testset": args.testset,
        "top_k": args.top_k,
        "workers": args.workers,
        "experiments_run": args.experiments,
        "layer1": layer1_results,
        "layer2": layer2_results,
    }
    unified_path = out_dir / f"unified_report_top{args.top_k}.json"
    with open(unified_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)
    print(f"\nUnified report saved: {unified_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
