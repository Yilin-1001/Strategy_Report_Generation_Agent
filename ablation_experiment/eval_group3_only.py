"""
Evaluate only Group 3 and merge with existing results.

This script:
1. Reads the existing evaluation_report.csv (preserving group0/1/2 results)
2. Re-evaluates all Group 3 reports (run 1, 2, 3) with 10 rounds
3. Replaces Group 3 rows in the CSV with new results
4. Regenerates the summary
"""

import sys
import csv
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_experiment.config import (
    REPORTS_DIR, RESULTS_DIR, NUM_EVAL_ROUNDS
)
from ablation_experiment.evaluation.evaluator import (
    ReportEvaluator, print_summary
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    # Step 1: Load existing results (keep group0, group1, group2)
    csv_path = RESULTS_DIR / "evaluation_report.csv"
    existing_rows = []

    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                group_val = row.get('group', row.get('\ufeffgroup', ''))
                if not group_val.startswith('group3'):
                    existing_rows.append(row)
        print(f"  Loaded {len(existing_rows)} existing rows (group0/1/2)")
    else:
        print("  [WARN] No existing CSV found")
        fieldnames = None

    # Step 2: Evaluate Group 3 reports
    print(f"\n{'=' * 70}")
    print(f"  Evaluating Group 3 Reports ({NUM_EVAL_ROUNDS} rounds each)")
    print(f"{'=' * 70}")

    evaluator = ReportEvaluator()
    new_group3_results = []

    for run_id in range(1, 4):
        report_path = REPORTS_DIR / "group3" / f"run_{run_id}.md"

        if not report_path.exists():
            print(f"  [SKIP] Report not found: {report_path}")
            continue

        # Quick check: verify the report is valid
        report_text = report_path.read_text(encoding='utf-8')
        if "由于技术原因" in report_text:
            print(f"  [WARN] Run {run_id} still contains fallback text!")
            print(f"  [SKIP] Skipping evaluation for invalid report")
            continue

        print(f"\n  --- Group 3 Run {run_id} ({NUM_EVAL_ROUNDS} rounds) ---")

        try:
            evaluation = evaluator.evaluate_with_repetition(
                report_text, num_rounds=NUM_EVAL_ROUNDS
            )

            total = evaluation.get("total_score", 0)
            scores = evaluation.get("scores", {})
            dim_summary = " | ".join(
                f"{v['score']}±{v['std']}" for v in scores.values()
            )

            new_group3_results.append({
                "group": "group3",
                "run": run_id,
                **evaluation
            })

            print(f"  [OK] Total: {total}/100 | {dim_summary}")

        except Exception as e:
            print(f"  [ERROR] Evaluation failed: {e}")
            logger.error(f"Evaluation failed for Group 3 Run {run_id}", exc_info=True)

        # Rate limit between reports
        if run_id < 3:
            time.sleep(5)

    # Step 3: Merge and save
    all_results = existing_rows + new_group3_results

    # Re-evaluate with the evaluator to get proper result format
    # Convert existing CSV rows back to result dicts for print_summary
    from ablation_experiment.evaluation.evaluator import save_results_csv

    # Save combined results
    # First save the new group3 results in the expected format
    if new_group3_results:
        csv_out = save_results_csv(all_results)
        print(f"\n  [CSV] Saved to: {csv_out}")

        # Print summary for all results
        print_summary(all_results)
    else:
        print("\n  [WARN] No new Group 3 results to merge.")

    print(f"\n{'=' * 70}")
    print("  Done! Group 3 evaluation merged with existing results.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
