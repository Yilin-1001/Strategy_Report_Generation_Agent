"""
Ablation Experiment Runner

Orchestrates all 4 experiment groups x 3 runs, saves reports,
evaluates with Qwen, and generates CSV results with summary statistics.

Usage:
    python -m ablation_experiment.runner
    python -m ablation_experiment.runner --group 0       # Run only Group 0
    python -m ablation_experiment.runner --skip-eval     # Skip evaluation
    python -m ablation_experiment.runner --eval-only     # Only evaluate existing reports
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_experiment.config import (
    TEST_QUERY, NUM_RUNS, REPORTS_DIR, RESULTS_DIR,
    CHAPTER_PLAN, NUM_EVAL_ROUNDS
)
from ablation_experiment.evaluation.evaluator import (
    ReportEvaluator, save_results_csv, print_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GROUP IMPORTS
# ============================================================================

def get_group_runner(group_id: int):
    """Import and return the runner function for a given group."""
    if group_id == 0:
        from ablation_experiment.groups.group0_baseline_rag import run_group0
        return run_group0
    elif group_id == 1:
        from ablation_experiment.groups.group1_single_agent_memory import run_group1
        return run_group1
    elif group_id == 2:
        from ablation_experiment.groups.group2_multi_agent_no_memory import run_group2
        return run_group2
    elif group_id == 3:
        from ablation_experiment.groups.group3_full_system import run_group3
        return run_group3
    else:
        raise ValueError(f"Unknown group: {group_id}")


GROUP_NAMES = {
    0: "组0: 基线RAG (无工作流/无记忆)",
    1: "组1: 单Agent+记忆 (无工作流/有记忆)",
    2: "组2: 多Agent无记忆 (有工作流/无记忆)",
    3: "组3: 完整系统 (有工作流/有记忆)",
}


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_reports(groups: List[int] = None) -> Dict:
    """
    Run report generation for specified groups.

    Args:
        groups: List of group IDs to run (default: all 4 groups)

    Returns:
        Dict mapping (group_id, run_id) -> report_path
    """
    if groups is None:
        groups = [0, 1, 2, 3]

    report_paths = {}

    for group_id in groups:
        group_name = GROUP_NAMES[group_id]
        print(f"\n{'=' * 70}")
        print(f"  生成报告: {group_name}")
        print(f"{'=' * 70}")

        try:
            run_fn = get_group_runner(group_id)
        except Exception as e:
            print(f"  [ERROR] Failed to import Group {group_id}: {e}")
            continue

        token_stats_all = {}

        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n  --- Run {run_id}/{NUM_RUNS} ---")
            start_time = time.time()

            try:
                result = run_fn(run_id)
                elapsed = time.time() - start_time

                # Handle both old (str) and new (tuple) return types
                if isinstance(result, tuple):
                    report_text, token_stats = result
                else:
                    report_text = result
                    token_stats = {}

                # Save report path
                report_path = REPORTS_DIR / f"group{group_id}" / f"run_{run_id}.md"
                report_paths[(group_id, run_id)] = report_path

                print(f"  [OK] Report generated ({len(report_text)} chars, {elapsed:.1f}s)")
                if token_stats and token_stats.get("total_tokens"):
                    print(f"  [Tokens] {token_stats['total_tokens']:,} total "
                          f"(prompt: {token_stats['prompt_tokens']:,}, "
                          f"completion: {token_stats['completion_tokens']:,})")
                print(f"  [File] {report_path}")

                token_stats_all[run_id] = token_stats

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  [ERROR] Run {run_id} failed after {elapsed:.1f}s: {e}")
                logger.error(f"Group {group_id} Run {run_id} failed", exc_info=True)

        # Print group-level token summary
        if token_stats_all:
            group_total = sum(s.get("total_tokens", 0) for s in token_stats_all.values())
            group_prompt = sum(s.get("prompt_tokens", 0) for s in token_stats_all.values())
            group_completion = sum(s.get("completion_tokens", 0) for s in token_stats_all.values())
            print(f"\n  [Group {group_id} Total Tokens] {group_total:,} "
                  f"(prompt: {group_prompt:,}, completion: {group_completion:,})")

    return report_paths


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_reports(groups: List[int] = None, num_eval_rounds: int = None) -> List[Dict]:
    """
    Evaluate all generated reports using Qwen model with repeated scoring.

    Each report is scored independently num_eval_rounds times, then aggregated
    via trimmed mean to reduce LLM hallucination noise.

    Args:
        groups: List of group IDs to evaluate (default: all)
        num_eval_rounds: Number of independent scoring rounds per report

    Returns:
        List of evaluation result dicts
    """
    if groups is None:
        groups = [0, 1, 2, 3]

    if num_eval_rounds is None:
        num_eval_rounds = NUM_EVAL_ROUNDS

    evaluator = ReportEvaluator()
    results = []

    for group_id in groups:
        group_name = GROUP_NAMES[group_id]
        print(f"\n{'=' * 70}")
        print(f"  评估报告: {group_name}")
        print(f"  每份报告打分 {num_eval_rounds} 次")
        print(f"{'=' * 70}")

        for run_id in range(1, NUM_RUNS + 1):
            report_path = REPORTS_DIR / f"group{group_id}" / f"run_{run_id}.md"

            if not report_path.exists():
                print(f"  [SKIP] Report not found: {report_path}")
                continue

            print(f"\n  --- Group {group_id} Run {run_id} ({num_eval_rounds} rounds) ---")

            try:
                report_text = report_path.read_text(encoding='utf-8')
                evaluation = evaluator.evaluate_with_repetition(
                    report_text, num_rounds=num_eval_rounds
                )

                total = evaluation.get("total_score", 0)
                scores = evaluation.get("scores", {})
                dim_summary = " | ".join(
                    f"{v['score']}±{v['std']}" for v in scores.values()
                )

                results.append({
                    "group": f"group{group_id}",
                    "run": run_id,
                    **evaluation
                })

                print(f"  [OK] Total: {total}/100 | {dim_summary}")

            except Exception as e:
                print(f"  [ERROR] Evaluation failed: {e}")
                logger.error(f"Evaluation failed for Group {group_id} Run {run_id}", exc_info=True)

            # Rate limit between reports
            if run_id < NUM_RUNS or group_id < max(groups):
                time.sleep(5)

    return results


# ============================================================================
# EVAL-ONLY MODE (evaluate existing reports)
# ============================================================================

def eval_only(groups: List[int] = None, num_eval_rounds: int = None):
    """Only evaluate existing reports without regenerating."""
    print("\n" + "=" * 70)
    print("  [Eval-Only Mode] Evaluating existing reports...")
    print("=" * 70)

    results = evaluate_reports(groups, num_eval_rounds=num_eval_rounds)

    if results:
        csv_path = save_results_csv(results)
        print(f"\n  [CSV] Saved to: {csv_path}")
        print_summary(results)
    else:
        print("\n  [WARN] No reports found to evaluate.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--group", type=int, nargs="*", default=None,
        help="Group IDs to run (0-3). Default: all groups"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip Qwen evaluation (only generate reports)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate existing reports (skip generation)"
    )
    parser.add_argument(
        "--runs", type=int, default=None,
        help="Override number of runs per group"
    )
    parser.add_argument(
        "--num-eval-rounds", type=int, default=None,
        help="Number of independent scoring rounds per report (default: 10)"
    )

    args = parser.parse_args()

    # Override NUM_RUNS if specified
    if args.runs:
        import ablation_experiment.config as config
        config.NUM_RUNS = args.runs

    groups = args.group if args.group is not None else [0, 1, 2, 3]
    num_eval_rounds = args.num_eval_rounds if args.num_eval_rounds else NUM_EVAL_ROUNDS

    # Header
    print("\n" + "=" * 70)
    print("  消融实验 (Ablation Experiment)")
    print("=" * 70)
    print(f"  测试主题: {TEST_QUERY}")
    print(f"  实验组数: {groups}")
    print(f"  每组运行次数: {args.runs or NUM_RUNS}")
    print(f"  每份报告打分次数: {num_eval_rounds}")
    print(f"  离群值处理: 截尾均值 (去最高最低各{min(2, num_eval_rounds // 5)}个)")
    print(f"  章节数: {len(CHAPTER_PLAN)}")
    print(f"  报告目录: {REPORTS_DIR}")
    print(f"  结果目录: {RESULTS_DIR}")
    print("=" * 70)

    if args.eval_only:
        eval_only(groups, num_eval_rounds=num_eval_rounds)
        return

    # Step 1: Generate reports
    print("\n\n" + "#" * 70)
    print("# Phase 1: Report Generation")
    print("#" * 70)

    total_start = time.time()
    report_paths = generate_reports(groups)
    gen_elapsed = time.time() - total_start

    print(f"\n  Report generation complete: {len(report_paths)} reports in {gen_elapsed:.1f}s")

    if args.skip_eval:
        print("\n  [SKIP] Evaluation skipped (--skip-eval flag)")
        return

    # Step 2: Evaluate reports
    print("\n\n" + "#" * 70)
    print("# Phase 2: Report Evaluation (Qwen Model)")
    print("#" * 70)

    eval_start = time.time()
    results = evaluate_reports(groups, num_eval_rounds=num_eval_rounds)
    eval_elapsed = time.time() - eval_start

    if results:
        # Save CSV
        csv_path = save_results_csv(results)
        print(f"\n  [CSV] Saved to: {csv_path}")

        # Print summary
        print_summary(results)

        total_elapsed = time.time() - total_start
        print(f"  Total time: {total_elapsed:.1f}s (Gen: {gen_elapsed:.1f}s, Eval: {eval_elapsed:.1f}s)")
    else:
        print("\n  [WARN] No evaluation results generated.")


if __name__ == "__main__":
    main()
