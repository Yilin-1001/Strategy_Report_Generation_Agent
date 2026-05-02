"""
Hallucination Evaluation Main Entry Point

Processes 12 reports (Group0-3 x 3 runs) through the full pipeline:
  1. Split reports into chapters
  2. Decompose into atomic facts
  3. Verify each fact (A-type/B-type)
  4. Compute metrics (FActScore, Hallucination Rate, etc.)
  5. Run statistical analysis (ANOVA, Tukey HSD, Cohen's d)
  6. Generate report and charts

Usage:
    # Full evaluation (all 8 chapters, all runs)
    python -m ablation_experiment.hallucination_eval.run_eval

    # Step 1: Evaluate run2/run3 diagnosis phase only (chapters 1-3)
    python -m ablation_experiment.hallucination_eval.run_eval --phase diagnosis --runs 2 3

    # Step 2: Extract run1 diagnosis data from old dir -> merge into new dir -> compute stats
    python -m ablation_experiment.hallucination_eval.run_eval --merge-run1 \
        --source results/202604182203 \
        --target results/20260419HHMM
"""

import argparse
import json
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

from .config import (
    REPORTS_DIR, GROUPS, NUM_RUNS, create_experiment_dir, CHAPTER_PLAN
)
from .chapter_splitter import split_report, validate_chapters
from .citation_parser import CitationParser
from .atomic_decomposer import AtomicDecomposer
from .fact_verifier import FactVerifier
from .metrics import (
    compute_report_metrics, compute_chapter_metrics,
    aggregate_group_metrics, ReportMetrics
)
from .statistical_analysis import (
    run_anova, run_tukey_hsd, run_spearman_correlation,
    compute_cohens_d, generate_charts, generate_markdown_report,
    run_chi_square_test, run_two_way_anova, generate_enhanced_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Diagnosis phase chapter indices
DIAGNOSIS_CHAPTERS = [0, 1, 2]


def load_report(group: str, run: int) -> str:
    """Load a report markdown file."""
    path = REPORTS_DIR / group / f"run_{run}.md"
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return path.read_text(encoding='utf-8')


def save_raw_results(verdicts_data: dict, exp_dir: Path, group: str, run: int):
    """Save raw verification results for a single report."""
    raw_path = exp_dir / "raw" / f"{group}_run{run}.json"
    raw_path.write_text(
        json.dumps(verdicts_data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def save_metrics_csv(all_metrics: list, output_path: Path):
    """Save metrics summary as CSV."""
    if not all_metrics:
        return

    fieldnames = [
        "group", "run", "total_facts", "total_analytical",
        "supported_by_source", "supported_by_kb", "supported_by_parametric",
        "contradicted", "citation_hallucination", "unsupported", "unverifiable",
        "total_citations", "invalid_citations",
        "factscore", "strict_factscore", "hallucination_rate",
        "uncertainty_rate", "citation_error_rate", "kb_support_rate", "parametric_rate"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rm in all_metrics:
            writer.writerow({
                "group": rm.group,
                "run": rm.run,
                "total_facts": rm.total_facts,
                "total_analytical": rm.total_analytical,
                "supported_by_source": rm.supported_by_source,
                "supported_by_kb": rm.supported_by_kb,
                "supported_by_parametric": rm.supported_by_parametric,
                "contradicted": rm.contradicted,
                "citation_hallucination": rm.citation_hallucination,
                "unsupported": rm.unsupported,
                "unverifiable": rm.unverifiable,
                "total_citations": rm.total_citations,
                "invalid_citations": rm.invalid_citations,
                "factscore": round(rm.factscore, 4),
                "strict_factscore": round(rm.strict_factscore, 4),
                "hallucination_rate": round(rm.hallucination_rate, 4),
                "uncertainty_rate": round(rm.uncertainty_rate, 4),
                "citation_error_rate": round(rm.citation_error_rate, 4),
                "kb_support_rate": round(rm.kb_support_rate, 4),
                "parametric_rate": round(rm.parametric_rate, 4),
            })


def run_single_report(
    group: str,
    run: int,
    decomposer: AtomicDecomposer,
    verifier: FactVerifier,
    exp_dir: Path,
    chapter_filter: list = None
) -> ReportMetrics:
    """Process a single report through the full pipeline.

    Args:
        group: Group identifier (group0-3).
        run: Run number (1-3).
        decomposer: AtomicDecomposer instance.
        verifier: FactVerifier instance.
        exp_dir: Experiment output directory.
        chapter_filter: If provided, only process chapters whose index is in this list.
                        E.g. [0, 1, 2] for diagnosis phase only.
    """
    phase_label = f"chapters {chapter_filter}" if chapter_filter else "all chapters"
    logger.info(f"  Processing {group}/run_{run} ({phase_label})...")

    # Load and split report
    report_text = load_report(group, run)
    chapters = split_report(report_text)

    if chapter_filter is not None:
        chapters = [ch for ch in chapters if ch.index in chapter_filter]
        logger.info(f"    Filtered to {len(chapters)} diagnosis chapters")

    if not validate_chapters(chapters) and chapter_filter is None:
        logger.warning(f"  Chapter validation failed for {group}/run_{run}")

    logger.info(f"    Processing {len(chapters)} chapters")

    # Process each chapter
    all_verdicts = []
    all_verdicts_data = []
    chapter_metrics_list = []

    for chapter in chapters:
        logger.info(f"    Chapter {chapter.index + 1}: decomposing...")

        # Decompose into atomic facts
        facts = decomposer.decompose_chapter(chapter.content)
        fact_count = sum(1 for f in facts if f.fact_type == "FACT")
        analysis_count = sum(1 for f in facts if f.fact_type == "ANALYSIS")
        logger.info(f"      {len(facts)} facts ({fact_count} FACT, {analysis_count} ANALYSIS)")

        # Verify each fact
        verdicts = []
        for fact in facts:
            verdict = verifier.verify_fact(fact)
            verdicts.append(verdict)

        all_verdicts.extend(verdicts)

        # Chapter metrics
        ch_metrics = compute_chapter_metrics(verdicts, chapter.index)
        chapter_metrics_list.append(ch_metrics)

        # Save verdict details
        for v in verdicts:
            all_verdicts_data.append({
                "chapter": chapter.index,
                "chapter_title": chapter.title,
                "fact_text": v.fact_text,
                "fact_type": v.fact_type,
                "citation": v.citation,
                "category": v.category,
                "result": v.result,
                "detail": v.detail
            })

    # Compute report-level metrics
    metrics = compute_report_metrics(all_verdicts, group, run)
    metrics.chapter_metrics = chapter_metrics_list

    # Save raw results
    save_raw_results({
        "group": group,
        "run": run,
        "total_chapters": len(chapters),
        "total_facts": metrics.total_facts,
        "total_analytical": metrics.total_analytical,
        "factscore": metrics.factscore,
        "hallucination_rate": metrics.hallucination_rate,
        "verdicts": all_verdicts_data,
        "chapter_metrics": chapter_metrics_list,
        "eval_scope": "diagnosis" if chapter_filter else "full",
    }, exp_dir, group, run)

    logger.info(
        f"    Result: FActScore={metrics.factscore:.3f}, "
        f"HallucRate={metrics.hallucination_rate:.3f}, "
        f"CiteErr={metrics.citation_error_rate:.3f}"
    )

    return metrics


def extract_diagnosis_from_raw(raw_path: Path) -> dict:
    """Extract diagnosis-phase verdicts and recompute metrics from an existing raw JSON.

    Args:
        raw_path: Path to the raw JSON file (e.g., raw/group0_run1.json).

    Returns:
        Dict with filtered diagnosis-phase data, ready to merge.
    """
    with open(raw_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    verdicts = data.get("verdicts", [])
    diag_verdicts = [v for v in verdicts if v["chapter"] in DIAGNOSIS_CHAPTERS]

    # Recompute chapter_metrics for diagnosis chapters
    diag_chapter_metrics = []
    for ch_idx in DIAGNOSIS_CHAPTERS:
        ch_v = [v for v in diag_verdicts if v["chapter"] == ch_idx]
        ch_total = sum(1 for v in ch_v if v["fact_type"] != "ANALYSIS")
        ch_supported = sum(
            1 for v in ch_v
            if v["fact_type"] != "ANALYSIS"
            and v["result"] in ("supported_by_source", "supported_by_kb", "supported_by_parametric")
        )
        ch_hallucinated = ch_total - ch_supported
        diag_chapter_metrics.append({
            "chapter": ch_idx,
            "total_facts": ch_total,
            "supported": ch_supported,
            "hallucinated": ch_hallucinated,
            "factscore": ch_supported / ch_total if ch_total > 0 else 0.0,
        })

    # Recompute aggregate metrics
    total_facts = sum(cm["total_facts"] for cm in diag_chapter_metrics)
    total_supported = sum(cm["supported"] for cm in diag_chapter_metrics)
    total_analytical = sum(1 for v in diag_verdicts if v["fact_type"] == "ANALYSIS")
    factscore = total_supported / total_facts if total_facts > 0 else 0.0

    return {
        "group": data["group"],
        "run": data["run"],
        "total_chapters": len(DIAGNOSIS_CHAPTERS),
        "total_facts": total_facts,
        "total_analytical": total_analytical,
        "factscore": factscore,
        "hallucination_rate": 1.0 - factscore,
        "verdicts": diag_verdicts,
        "chapter_metrics": diag_chapter_metrics,
        "eval_scope": "diagnosis",
        "source": f"extracted from full-report raw: {raw_path.name}",
    }


def run_diagnosis_evaluation(runs: list = None):
    """Run hallucination evaluation for diagnosis phase only.

    Args:
        runs: List of run numbers to evaluate. Default: [2, 3].
    """
    if runs is None:
        runs = [2, 3]

    start_time = time.time()
    exp_dir = create_experiment_dir()
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Mode: DIAGNOSIS PHASE ONLY (chapters 1-3)")
    logger.info(f"Runs to evaluate: {runs}")

    # Initialize components
    logger.info("Initializing components...")
    citation_parser = CitationParser()
    decomposer = AtomicDecomposer()
    verifier = FactVerifier(citation_parser)

    kb_stats = citation_parser.get_stats()
    logger.info(f"KB loaded: {kb_stats['total_sources']} sources, {kb_stats['total_chunks']} chunks")

    # Process reports for specified runs only
    all_metrics = []
    report_metrics_by_group = {g: [] for g in GROUPS}

    for group in GROUPS:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {group}")
        logger.info(f"{'='*40}")

        for run in runs:
            try:
                metrics = run_single_report(
                    group, run, decomposer, verifier, exp_dir,
                    chapter_filter=DIAGNOSIS_CHAPTERS
                )
                all_metrics.append(metrics)
                report_metrics_by_group[group].append(metrics)
            except FileNotFoundError as e:
                logger.error(f"  Report not found: {e}")
            except Exception as e:
                logger.error(f"  Error processing {group}/run_{run}: {e}")
                import traceback
                traceback.print_exc()

    # Save metrics CSV
    save_metrics_csv(all_metrics, exp_dir / "metrics.csv")
    logger.info(f"\nMetrics saved to {exp_dir / 'metrics.csv'}")

    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("DIAGNOSIS PHASE EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"Evaluated runs: {runs}")
    logger.info(f"Note: Run 1 diagnosis data should be merged separately using --merge-run1")
    logger.info("")

    for group in GROUPS:
        rms = report_metrics_by_group[group]
        if rms:
            mean_fs = sum(rm.factscore for rm in rms) / len(rms)
            mean_hall = sum(rm.hallucination_rate for rm in rms) / len(rms)
            logger.info(
                f"  {group}: FActScore={mean_fs:.3f}, "
                f"HallucRate={mean_hall:.3f} ({len(rms)} runs)"
            )


def run_merge_run1(source_exp_dir: Path, target_exp_dir: Path):
    """Extract run1 diagnosis-phase data from existing full-report raw files
    and write into the target experiment directory (where run2/run3 results are).

    Source files are NEVER modified — only read from.

    Args:
        source_exp_dir: Directory containing run1 full-report raw files
                        (e.g., results/202604182203). Read-only.
        target_exp_dir: Directory where run2/run3 diagnosis results live.
                        Extracted run1 data will be written here.
    """
    source_raw = source_exp_dir / "raw"
    target_raw = target_exp_dir / "raw"

    if not source_raw.exists():
        logger.error(f"Source raw directory not found: {source_raw}")
        return
    if not target_raw.exists():
        logger.error(f"Target raw directory not found: {target_raw}")
        logger.info(f"  Make sure to run --phase diagnosis --runs 2 3 first")
        return

    logger.info(f"Source (read-only): {source_exp_dir}")
    logger.info(f"Target (write to):  {target_exp_dir}")

    merged_count = 0
    for group in GROUPS:
        source_file = source_raw / f"{group}_run1.json"
        if not source_file.exists():
            logger.warning(f"  {source_file.name} not found in source, skipping")
            continue

        # Extract diagnosis-phase data (source file NOT modified)
        diag_data = extract_diagnosis_from_raw(source_file)

        # Write to target directory
        target_file = target_raw / f"{group}_run1.json"
        target_file.write_text(
            json.dumps(diag_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.info(
            f"  {group}_run1.json: extracted {diag_data['total_facts']} diagnosis facts "
            f"-> {target_file}"
        )
        merged_count += 1

    logger.info(f"Merged {merged_count}/4 groups")

    # After merging, generate combined statistical report for all 3 runs
    _generate_combined_report(target_exp_dir)

    logger.info(f"\nMerge complete. Combined report in {target_exp_dir}")


def _generate_combined_report(exp_dir: Path):
    """Generate combined metrics CSV and summary from all raw files in exp_dir."""
    raw_dir = exp_dir / "raw"
    if not raw_dir.exists():
        return

    # Load all raw files and build ReportMetrics
    all_metrics = []
    report_metrics_by_group = {g: [] for g in GROUPS}

    for group in GROUPS:
        for run_id in [1, 2, 3]:
            raw_file = raw_dir / f"{group}_run{run_id}.json"
            if not raw_file.exists():
                continue

            with open(raw_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct ReportMetrics from saved data
            rm = ReportMetrics(group=group, run=run_id)
            rm.total_facts = data.get("total_facts", 0)
            rm.total_analytical = data.get("total_analytical", 0)
            rm.factscore = data.get("factscore", 0)
            rm.hallucination_rate = data.get("hallucination_rate", 0)
            rm.strict_factscore = data.get("strict_factscore", 0)

            # Count from verdicts for accurate breakdown
            verdicts = data.get("verdicts", [])
            for v in verdicts:
                if v.get("fact_type") == "ANALYSIS":
                    continue
                result = v.get("result", "")
                if result == "supported_by_source":
                    rm.supported_by_source += 1
                elif result == "supported_by_kb":
                    rm.supported_by_kb += 1
                elif result == "supported_by_parametric":
                    rm.supported_by_parametric += 1
                elif result == "contradicted":
                    rm.contradicted += 1
                elif result == "citation_hallucination":
                    rm.citation_hallucination += 1
                elif result == "unsupported":
                    rm.unsupported += 1
                elif result == "unverifiable":
                    rm.unverifiable += 1

            # Recompute derived metrics
            N = rm.total_facts
            if N > 0:
                S = rm.supported_by_source + rm.supported_by_kb + rm.supported_by_parametric
                rm.factscore = S / N
                rm.hallucination_rate = 1.0 - rm.factscore
                rm.strict_factscore = (rm.supported_by_source + rm.supported_by_kb) / N
                rm.kb_support_rate = (rm.supported_by_source + rm.supported_by_kb) / N
                rm.parametric_rate = rm.supported_by_parametric / N
                rm.uncertainty_rate = rm.unverifiable / N

            if rm.total_citations > 0:
                rm.citation_error_rate = rm.invalid_citations / rm.total_citations

            all_metrics.append(rm)
            report_metrics_by_group[group].append(rm)

    # Save combined metrics CSV
    save_metrics_csv(all_metrics, exp_dir / "diagnosis_metrics.csv")

    # Statistical analysis
    logger.info("Running statistical analysis on combined data...")
    group_metrics_list = []
    for group in GROUPS:
        gm = aggregate_group_metrics(report_metrics_by_group[group], group)
        group_metrics_list.append(gm)

    # ANOVA (requires at least 2 runs per group)
    anova_results = run_anova(report_metrics_by_group)

    # Tukey HSD
    tukey_results = run_tukey_hsd(report_metrics_by_group)

    # Cohen's d
    cohens_d_results = {}
    key_comparisons = [
        ("group0", "group3"),
        ("group0", "group1"),
        ("group0", "group2"),
        ("group1", "group3"),
        ("group2", "group3"),
    ]
    for g1, g2 in key_comparisons:
        vals1 = [rm.factscore for rm in report_metrics_by_group.get(g1, [])]
        vals2 = [rm.factscore for rm in report_metrics_by_group.get(g2, [])]
        d = compute_cohens_d(vals1, vals2)
        cohens_d_results[f"{g1}_vs_{g2}"] = d

    # Correlation
    correlation_results = None
    eval_csv = Path("ablation_experiment/results/evaluation_report.csv")
    if eval_csv.exists():
        try:
            eval_scores = _load_eval_scores(eval_csv)
            correlation_results = run_spearman_correlation(
                report_metrics_by_group, eval_scores
            )
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")

    # Generate charts
    try:
        generate_charts(group_metrics_list, anova_results, exp_dir)
    except Exception as e:
        logger.warning(f"Could not generate charts: {e}")

    # Load raw JSON data for tier 1 & tier 2 analysis
    raw_data_by_group = {g: [] for g in GROUPS}
    for group in GROUPS:
        for run_id in [1, 2, 3]:
            raw_file = raw_dir / f"{group}_run{run_id}.json"
            if raw_file.exists():
                with open(raw_file, 'r', encoding='utf-8') as f:
                    raw_data_by_group[group].append(json.load(f))

    # Tier 1: Verdict-level chi-square
    logger.info("Running Tier 1: Verdict-level chi-square test...")
    chi_square_results = run_chi_square_test(raw_data_by_group)
    logger.info(
        f"  Chi2={chi_square_results.get('chi2', 'N/A')}, "
        f"p={chi_square_results.get('p_value', 'N/A')}, "
        f"Cramer's V={chi_square_results.get('cramers_v', 'N/A')}"
    )

    # Tier 2: Chapter-level two-way ANOVA
    logger.info("Running Tier 2: Chapter-level two-way ANOVA...")
    two_way_results = run_two_way_anova(raw_data_by_group)
    if "error" not in two_way_results:
        ge = two_way_results.get("group_effect", {})
        ie = two_way_results.get("interaction", {})
        logger.info(
            f"  Group effect: F={ge.get('f_stat', 'N/A')}, p={ge.get('p_value', 'N/A')}"
        )
        logger.info(
            f"  Interaction: F={ie.get('f_stat', 'N/A')}, p={ie.get('p_value', 'N/A')}"
        )

    # Generate enhanced report (all three tiers)
    generate_enhanced_report(
        group_metrics_list, anova_results, tukey_results,
        correlation_results, report_metrics_by_group,
        chi_square_results, two_way_results, cohens_d_results,
        exp_dir / "diagnosis_statistical_report.md"
    )

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("COMBINED DIAGNOSIS PHASE REPORT (3-Tier Analysis)")
    logger.info(f"{'='*60}")
    for gm in group_metrics_list:
        logger.info(
            f"  {gm.group} ({gm.architecture}): "
            f"FActScore={gm.factscore_mean:.3f}+/-{gm.factscore_std:.3f}, "
            f"HallucRate={gm.hallucination_rate_mean:.3f}+/-{gm.hallucination_rate_std:.3f}"
        )


def run_full_evaluation():
    """Run the full hallucination evaluation pipeline (all chapters, all runs)."""
    start_time = time.time()

    # Create experiment directory
    exp_dir = create_experiment_dir()
    logger.info(f"Experiment directory: {exp_dir}")

    # Initialize components
    logger.info("Initializing components...")
    citation_parser = CitationParser()
    decomposer = AtomicDecomposer()
    verifier = FactVerifier(citation_parser)

    kb_stats = citation_parser.get_stats()
    logger.info(f"KB loaded: {kb_stats['total_sources']} sources, {kb_stats['total_chunks']} chunks")

    # Process all reports
    all_metrics = []
    report_metrics_by_group = {g: [] for g in GROUPS}

    for group in GROUPS:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {group}")
        logger.info(f"{'='*40}")

        for run in range(1, NUM_RUNS + 1):
            try:
                metrics = run_single_report(
                    group, run, decomposer, verifier, exp_dir
                )
                all_metrics.append(metrics)
                report_metrics_by_group[group].append(metrics)
            except FileNotFoundError as e:
                logger.error(f"  Report not found: {e}")
            except Exception as e:
                logger.error(f"  Error processing {group}/run_{run}: {e}")
                import traceback
                traceback.print_exc()

    # Save metrics CSV
    save_metrics_csv(all_metrics, exp_dir / "metrics.csv")
    logger.info(f"\nMetrics saved to {exp_dir / 'metrics.csv'}")

    # Statistical analysis
    logger.info("\nRunning statistical analysis...")

    # Aggregate group metrics
    group_metrics_list = []
    for group in GROUPS:
        gm = aggregate_group_metrics(report_metrics_by_group[group], group)
        group_metrics_list.append(gm)

    # ANOVA
    anova_results = run_anova(report_metrics_by_group)

    # Tukey HSD
    tukey_results = run_tukey_hsd(report_metrics_by_group)

    # Cohen's d for key comparisons
    cohens_d_results = {}
    key_comparisons = [
        ("group0", "group3"),  # Baseline vs Full
        ("group0", "group1"),  # Baseline vs Memory
        ("group0", "group2"),  # Baseline vs Multi-Agent
        ("group1", "group3"),  # Memory vs Full
        ("group2", "group3"),  # Multi-Agent vs Full
    ]
    for g1, g2 in key_comparisons:
        vals1 = [rm.factscore for rm in report_metrics_by_group.get(g1, [])]
        vals2 = [rm.factscore for rm in report_metrics_by_group.get(g2, [])]
        d = compute_cohens_d(vals1, vals2)
        cohens_d_results[f"{g1}_vs_{g2}"] = d

    # Correlation (load existing eval scores if available)
    correlation_results = None
    eval_csv = Path("ablation_experiment/results/evaluation_report.csv")
    if eval_csv.exists():
        try:
            eval_scores = _load_eval_scores(eval_csv)
            correlation_results = run_spearman_correlation(
                report_metrics_by_group, eval_scores
            )
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")

    # Generate charts
    try:
        generate_charts(group_metrics_list, anova_results, exp_dir)
        logger.info(f"Charts saved to {exp_dir}")
    except Exception as e:
        logger.warning(f"Could not generate charts: {e}")

    # Generate markdown report
    generate_markdown_report(
        group_metrics_list, anova_results, tukey_results,
        correlation_results, report_metrics_by_group,
        exp_dir / "statistical_report.md"
    )

    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("")

    for gm in group_metrics_list:
        logger.info(
            f"  {gm.group} ({gm.architecture}): "
            f"FActScore={gm.factscore_mean:.3f}+/-{gm.factscore_std:.3f}, "
            f"HallucRate={gm.hallucination_rate_mean:.3f}+/-{gm.hallucination_rate_std:.3f}"
        )


def _load_eval_scores(csv_path: Path) -> dict:
    """Load existing evaluation scores from CSV."""
    scores = {g: [] for g in GROUPS}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row.get('group', '')
            if group in scores:
                # Sum dimension scores as total
                total = 0
                for dim in ['score1', 'score2', 'score3', 'score4', 'score5']:
                    try:
                        total += float(row.get(dim, 0))
                    except (ValueError, TypeError):
                        pass
                scores[group].append(total)

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination Evaluation (FActScore-aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (all 8 chapters, all runs)
  python -m ablation_experiment.hallucination_eval.run_eval

  # Step 1: Evaluate run2/run3 diagnosis phase only
  python -m ablation_experiment.hallucination_eval.run_eval --phase diagnosis --runs 2 3

  # Step 2: Merge run1 diagnosis data into run2/run3 output directory
  python -m ablation_experiment.hallucination_eval.run_eval --merge-run1 ^
      --source results/202604182203 ^
      --target results/20260419HHMM

  # Re-run statistics on existing raw data (no LLM calls)
  python -m ablation_experiment.hallucination_eval.run_eval --stats-only results/202604182203
        """
    )
    parser.add_argument(
        "--phase", choices=["full", "diagnosis"], default="full",
        help="Evaluation scope: 'full' (all 8 chapters) or 'diagnosis' (chapters 1-3 only)"
    )
    parser.add_argument(
        "--runs", type=int, nargs="+", default=None,
        help="Run numbers to evaluate (e.g., --runs 2 3). Default: all runs"
    )
    parser.add_argument(
        "--merge-run1", action="store_true",
        help="Extract run1 diagnosis data from source dir, write to target dir, "
             "then compute combined statistics for all 3 runs"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="[--merge-run1] Directory containing run1 full-report raw files "
             "(read-only, never modified). e.g., results/202604182203"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="[--merge-run1] Directory where run2/run3 results live "
             "(extracted run1 data will be written here). e.g., results/20260419HHMM"
    )
    parser.add_argument(
        "--stats-only", type=str, default=None, metavar="EXP_DIR",
        help="Re-run three-tier statistical analysis on existing raw data. "
             "No LLM calls, pure computation. e.g., --stats-only results/202604182203"
    )

    args = parser.parse_args()
    results_base = Path(__file__).parent / "results"

    if args.stats_only:
        exp_dir = Path(args.stats_only)
        if not exp_dir.is_absolute():
            exp_dir = results_base / args.stats_only.replace("results/", "")
        if not (exp_dir / "raw").exists():
            logger.error(f"Raw directory not found: {exp_dir / 'raw'}")
            return
        logger.info(f"Stats-only mode: reading raw data from {exp_dir}")
        _generate_combined_report(exp_dir)

    elif args.merge_run1:
        # Resolve source directory (where run1 full-report raw files are)
        if args.source:
            source_dir = Path(args.source)
            if not source_dir.is_absolute():
                source_dir = results_base / args.source.replace("results/", "")
        else:
            # Auto-detect: use the oldest directory as source
            dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
            if not dirs:
                logger.error("No experiment directories found for --source")
                return
            source_dir = dirs[0]
            logger.info(f"Auto-detected source: {source_dir}")

        # Resolve target directory (where run2/run3 results are)
        if args.target:
            target_dir = Path(args.target)
            if not target_dir.is_absolute():
                target_dir = results_base / args.target.replace("results/", "")
        else:
            # Auto-detect: use the latest directory as target
            dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
            if not dirs:
                logger.error("No experiment directories found for --target")
                return
            target_dir = dirs[-1]
            logger.info(f"Auto-detected target: {target_dir}")

        if source_dir == target_dir:
            logger.error("Source and target are the same directory. "
                         "Please specify different directories.")
            return

        run_merge_run1(source_dir, target_dir)

    elif args.phase == "diagnosis":
        runs = args.runs if args.runs else [2, 3]
        run_diagnosis_evaluation(runs=runs)

    else:
        run_full_evaluation()


if __name__ == "__main__":
    main()
