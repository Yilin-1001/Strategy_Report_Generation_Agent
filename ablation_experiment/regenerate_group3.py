"""
Regenerate only Group 3 Run 2 and Run 3 reports.

This script directly calls run_group3 for specific run IDs,
bypassing the full experiment runner.
"""

import sys
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    from ablation_experiment.groups.group3_full_system import run_group3

    run_ids = [2, 3]

    for run_id in run_ids:
        print(f"\n{'=' * 70}")
        print(f"  Regenerating Group 3 Run {run_id}")
        print(f"{'=' * 70}")

        start_time = time.time()
        try:
            result = run_group3(run_id)
            elapsed = time.time() - start_time

            if isinstance(result, tuple):
                report_text, token_stats = result
            else:
                report_text = result
                token_stats = {}

            print(f"\n  [OK] Run {run_id} complete ({len(report_text)} chars, {elapsed:.1f}s)")
            if token_stats and token_stats.get("total_tokens"):
                print(f"  [Tokens] {token_stats['total_tokens']:,} total")

            # Verify the report is not a fallback
            if "由于技术原因" in report_text:
                print(f"  [WARNING] Report still contains fallback text!")
            else:
                print(f"  [OK] Report looks valid (no fallback text detected)")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n  [ERROR] Run {run_id} failed after {elapsed:.1f}s: {e}")
            logger.error(f"Run {run_id} failed", exc_info=True)

    print(f"\n{'=' * 70}")
    print("  All runs complete. Ready for evaluation.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
