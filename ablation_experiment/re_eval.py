"""
Re-evaluate all 12 ablation experiment reports with updated scoring rubric.

Uses the new dimension 4 (创新性与前瞻洞察力) instead of KPI实操.
Results are saved to a timestamped folder under results/ to avoid overwriting.

Usage:
    python -m ablation_experiment.re_eval
    python -m ablation_experiment.re_eval --num-eval-rounds 10
    python -m ablation_experiment.re_eval --group 0 3
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_experiment.config import (
    REPORTS_DIR, RESULTS_DIR, NUM_RUNS, NUM_EVAL_ROUNDS,
)
from ablation_experiment.evaluation.evaluator import (
    ReportEvaluator, save_results_csv, print_summary,
)

GROUP_NAMES = {
    0: "组0: 基线RAG (无工作流/无记忆)",
    1: "组1: 单Agent+记忆 (无工作流/有记忆)",
    2: "组2: 多Agent无记忆 (有工作流/无记忆)",
    3: "组3: 完整系统 (有工作流/有记忆)",
}


def re_eval(groups=None, num_eval_rounds=None):
    groups = groups or [0, 1, 2, 3]
    num_eval_rounds = num_eval_rounds or NUM_EVAL_ROUNDS

    # Create timestamped output directory
    ts = datetime.now().strftime("%Y%m%d%H%M")
    out_dir = RESULTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  重新评估 (Updated Rubric: 维度四→创新性与前瞻洞察力)")
    print(f"  打分轮次: {num_eval_rounds}/report")
    print(f"  输出目录: {out_dir}")
    print(f"{'=' * 70}")

    evaluator = ReportEvaluator()
    results = []

    for group_id in groups:
        print(f"\n{'─' * 60}")
        print(f"  {GROUP_NAMES[group_id]}")
        print(f"{'─' * 60}")

        for run_id in range(1, NUM_RUNS + 1):
            report_path = REPORTS_DIR / f"group{group_id}" / f"run_{run_id}.md"

            if not report_path.exists():
                print(f"  [SKIP] {report_path.name} not found")
                continue

            print(f"\n  --- Group {group_id} Run {run_id} ({num_eval_rounds} rounds) ---")

            try:
                report_text = report_path.read_text(encoding="utf-8")
                evaluation = evaluator.evaluate_with_repetition(
                    report_text, num_rounds=num_eval_rounds
                )

                total = evaluation.get("total_score", 0)
                scores = evaluation.get("scores", {})
                dim_summary = " | ".join(
                    f"{v['score']}\u00b1{v['std']}" for v in scores.values()
                )

                results.append({
                    "group": f"group{group_id}",
                    "run": run_id,
                    **evaluation,
                })

                print(f"  [OK] Total: {total}/100 | {dim_summary}")

            except Exception as e:
                print(f"  [ERROR] {e}")

            if run_id < NUM_RUNS or group_id < max(groups):
                time.sleep(5)

    if not results:
        print("\n  [WARN] No results generated.")
        return

    # Save CSV to timestamped folder
    csv_path = str(out_dir / "evaluation_report.csv")
    save_results_csv(results, filepath=csv_path)
    print(f"\n  [CSV] {csv_path}")

    # Save summary to timestamped folder
    summary = print_summary(results)
    summary_path = out_dir / "evaluation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"  [Summary] {summary_path}")

    # Save raw JSON for each result
    import json
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    for r in results:
        fname = f"{r['group']}_run{r['run']}.json"
        # Remove non-serializable fields
        clean = {k: v for k, v in r.items() if k != "scores"}
        clean["scores"] = {}
        for dim_key, dim_val in r.get("scores", {}).items():
            clean["scores"][dim_key] = {
                k: v for k, v in dim_val.items() if k != "raw_values"
            }
        (raw_dir / fname).write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"  [Raw] {raw_dir}/")

    print(f"\n{'=' * 70}")
    print(f"  Done! All results saved to: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-evaluate reports with updated rubric")
    parser.add_argument("--group", type=int, nargs="*", default=None,
                        help="Group IDs (0-3). Default: all")
    parser.add_argument("--num-eval-rounds", type=int, default=None,
                        help=f"Scoring rounds per report (default: {NUM_EVAL_ROUNDS})")
    args = parser.parse_args()
    re_eval(groups=args.group, num_eval_rounds=args.num_eval_rounds)
