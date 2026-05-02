"""
Statistical Analysis Module

Three-tier analysis for ablation experiment hallucination evaluation:

Tier 1 (Verdict-level): Chi-square test on 2x4 contingency table
  - N ~700 verdicts across 4 groups, highest statistical power
  - Tests whether hallucination proportion differs across groups

Tier 2 (Chapter-level): Two-way ANOVA (Group x Chapter)
  - N = 4 groups x 3 runs x 3 chapters = 36 observations
  - Tests Group main effect, Chapter main effect, and interaction
  - Repeated-measures structure accounts for within-report correlation

Tier 3 (Run-level): One-way ANOVA + Tukey HSD + Cohen's d
  - N = 4 groups x 3 runs = 12 observations (top-level summary)
  - Classic between-groups comparison

Also includes: Spearman correlation, visualization, report generation.
"""

from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter

from .metrics import ReportMetrics, GroupMetrics, aggregate_group_metrics
from .config import ALPHA, GROUPS


# ============================================================================
# TIER 1: Verdict-level Chi-square Test
# ============================================================================

def run_chi_square_test(raw_data_by_group: Dict[str, List[dict]]) -> Dict:
    """Pearson chi-square test on verdict-level 2x4 contingency table.

    Each verdict is a FACT-type atomic claim classified as either
    "supported" or "hallucinated".  We test whether the proportion
    of hallucinated facts differs across the 4 groups.

    Args:
        raw_data_by_group: {group_name: [list of raw JSON dicts, one per run]}

    Returns:
        Dict with chi2 statistic, p-value, Cramer's V, contingency table.
    """
    from scipy import stats as sp_stats
    import numpy as np

    # Build 2x4 contingency table: rows=[supported, hallucinated], cols=[group0..3]
    table = np.zeros((2, len(GROUPS)), dtype=int)

    for col_idx, group in enumerate(GROUPS):
        for raw_data in raw_data_by_group.get(group, []):
            verdicts = raw_data.get("verdicts", [])
            for v in verdicts:
                if v.get("fact_type") != "FACT":
                    continue
                result = v.get("result", "")
                if result in ("supported_by_source", "supported_by_kb", "supported_by_parametric"):
                    table[0, col_idx] += 1   # supported
                elif result in ("unsupported", "contradicted", "citation_hallucination", "unverifiable"):
                    table[1, col_idx] += 1   # hallucinated (includes unverifiable)

    # Pearson chi-square
    chi2, p_value, dof, expected = sp_stats.chi2_contingency(table)

    # Cramer's V (effect size for chi-square)
    n = table.sum()
    k = min(table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1))) if n > 0 and k > 1 else 0

    # Pairwise chi-square (2x2 tables) with Bonferroni correction
    pairwise = []
    group_pairs = [(GROUPS[i], GROUPS[j]) for i in range(len(GROUPS)) for j in range(i+1, len(GROUPS))]
    n_pairs = len(group_pairs)
    for g1, g2 in group_pairs:
        i1 = GROUPS.index(g1)
        i2 = GROUPS.index(g2)
        sub_table = table[:, [i1, i2]]
        try:
            chi2_pair, p_pair, _, _ = sp_stats.chi2_contingency(sub_table, correction=True)
            p_adj = min(p_pair * n_pairs, 1.0)  # Bonferroni
            pairwise.append({
                "group1": g1, "group2": g2,
                "chi2": round(chi2_pair, 4),
                "p_raw": round(p_pair, 6),
                "p_bonferroni": round(p_adj, 6),
                "significant": p_adj < ALPHA,
            })
        except Exception:
            pairwise.append({
                "group1": g1, "group2": g2,
                "chi2": float("nan"), "p_raw": float("nan"),
                "p_bonferroni": float("nan"), "significant": False,
            })

    # Hallucination proportions per group
    proportions = {}
    for col_idx, group in enumerate(GROUPS):
        total = table[0, col_idx] + table[1, col_idx]
        proportions[group] = {
            "total": int(total),
            "supported": int(table[0, col_idx]),
            "hallucinated": int(table[1, col_idx]),
            "hallucination_rate": round(table[1, col_idx] / total, 4) if total > 0 else 0,
        }

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": dof,
        "significant": p_value < ALPHA,
        "cramers_v": round(cramers_v, 4),
        "contingency_table": table.tolist(),
        "proportions": proportions,
        "pairwise": pairwise,
        "total_verdicts": int(n),
    }


# ============================================================================
# TIER 2: Chapter-level Two-way ANOVA (Group x Chapter)
# ============================================================================

def run_two_way_anova(raw_data_by_group: Dict[str, List[dict]]) -> Dict:
    """Two-way ANOVA on chapter-level factscore: Group (4) x Chapter (3).

    Data structure: each row = (group, run, chapter_index, factscore)
    Total rows: 4 groups x 3 runs x 3 chapters = 36 observations.

    Tests:
      - Group main effect: do groups differ overall?
      - Chapter main effect: do chapters differ overall?
      - Group x Chapter interaction: do group differences vary by chapter?

    Args:
        raw_data_by_group: {group_name: [list of raw JSON dicts]}

    Returns:
        Dict with ANOVA table, post-hoc tests, effect sizes.
    """
    import numpy as np
    import pandas as pd

    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False

    # Build chapter-level DataFrame from raw data
    rows = []
    for group in GROUPS:
        for raw_data in raw_data_by_group.get(group, []):
            run_id = raw_data.get("run", 0)
            cm_list = raw_data.get("chapter_metrics", [])
            for cm in cm_list:
                ch_idx = cm.get("chapter", -1)
                if ch_idx < 0 or ch_idx > 2:
                    continue
                rows.append({
                    "group": group,
                    "run": run_id,
                    "chapter": ch_idx,
                    "factscore": cm.get("factscore", 0),
                    "hall_rate": 1 - cm.get("factscore", 0),
                    "total_facts": cm.get("total_facts", 0),
                    "supported": cm.get("supported", 0),
                    "hallucinated": cm.get("hallucinated", 0),
                })

    if len(rows) < 12:
        return {"error": f"Not enough data ({len(rows)} rows), need at least 12"}

    df = pd.DataFrame(rows)
    n_obs = len(df)

    # Check minimum data requirements: need at least 2 runs per group
    runs_per_group = df.groupby("group")["run"].nunique()
    if runs_per_group.min() < 2:
        return {
            "error": "Need at least 2 runs per group for two-way ANOVA",
            "n_observations": n_obs,
            "runs_per_group": runs_per_group.to_dict(),
            "descriptive": {
                g: {"mean": round(df[df["group"] == g]["factscore"].mean(), 4),
                    "std": round(df[df["group"] == g]["factscore"].std(), 4) if len(df[df["group"] == g]) > 1 else 0,
                    "n": len(df[df["group"] == g])}
                for g in GROUPS
            },
        }

    result = {"n_observations": n_obs, "data_rows": rows}

    if has_statsmodels:
        # Two-way ANOVA with interaction
        model = ols('factscore ~ C(group) * C(chapter)', data=df).fit()
        anova_table = anova_lm(model, typ=2)

        # Convert to serializable dict
        anova_dict = {}
        for source in anova_table.index:
            row = anova_table.loc[source]
            anova_dict[source] = {
                "sum_sq": round(float(row["sum_sq"]), 4),
                "df": int(row["df"]),
                "f": round(float(row["F"]), 4) if not np.isnan(row.get("F", np.nan)) else None,
                "p": round(float(row["PR(>F)"]), 6) if not np.isnan(row.get("PR(>F)", np.nan)) else None,
            }
        result["anova_table"] = anova_dict

        # Group main effect significance
        group_key = "C(group)"
        result["group_effect"] = {
            "significant": anova_dict.get(group_key, {}).get("p", 1) < ALPHA,
            "p_value": anova_dict.get(group_key, {}).get("p"),
            "f_stat": anova_dict.get(group_key, {}).get("f"),
        }

        # Interaction significance
        interact_key = "C(group):C(chapter)"
        result["interaction"] = {
            "significant": anova_dict.get(interact_key, {}).get("p", 1) < ALPHA,
            "p_value": anova_dict.get(interact_key, {}).get("p"),
            "f_stat": anova_dict.get(interact_key, {}).get("f"),
        }

        # Partial eta-squared for group effect
        if group_key in anova_dict:
            ss_group = anova_dict[group_key]["sum_sq"]
            ss_total = sum(v["sum_sq"] for v in anova_dict.values())
            result["eta_squared_partial"] = round(ss_group / (ss_group + anova_dict.get("Residual", {}).get("sum_sq", 1)), 4)

        # Post-hoc: Tukey HSD on group (marginal means)
        try:
            tukey = pairwise_tukeyhsd(df["factscore"], df["group"], alpha=ALPHA)
            posthoc = []
            for i in range(len(tukey.groupsunique)):
                for j in range(i + 1, len(tukey.groupsunique)):
                    idx = i * len(tukey.groupsunique) + j - (i + 1) * len(tukey.groupsunique) // 2
                    if idx < len(tukey.reject):
                        posthoc.append({
                            "group1": str(tukey.groupsunique[i]),
                            "group2": str(tukey.groupsunique[j]),
                            "meandiff": round(float(tukey.meandiffs[idx]), 4),
                            "p_adj": round(float(tukey.pvalues[idx]), 6),
                            "reject": bool(tukey.reject[idx]),
                        })
            result["tukey_posthoc"] = posthoc
        except Exception as e:
            result["tukey_posthoc"] = {"error": str(e)}

        # Descriptive statistics per group (chapter-level)
        desc = df.groupby("group")["factscore"].agg(["mean", "std", "count"])
        result["descriptive"] = {
            g: {"mean": round(desc.loc[g, "mean"], 4),
                "std": round(desc.loc[g, "std"], 4),
                "n": int(desc.loc[g, "count"])}
            for g in GROUPS if g in desc.index
        }

        # Per-chapter breakdown
        ch_desc = df.groupby(["group", "chapter"])["factscore"].agg(["mean", "std"])
        result["chapter_descriptive"] = {}
        for (g, ch), row in ch_desc.iterrows():
            key = f"{g}_ch{ch}"
            result["chapter_descriptive"][key] = {
                "mean": round(row["mean"], 4),
                "std": round(row["std"], 4),
            }

    else:
        result["error"] = "statsmodels not installed. Run: pip install statsmodels"

    return result


# ============================================================================
# TIER 3: Run-level ANOVA (existing)
# ============================================================================

def run_anova(report_metrics: Dict[str, List[ReportMetrics]]) -> Dict:
    """
    One-way ANOVA across 4 groups for each metric.

    Args:
        report_metrics: {group_name: [ReportMetrics for each run]}

    Returns:
        Dict of ANOVA results per metric
    """
    import numpy as np
    from scipy import stats

    metrics_to_test = [
        "factscore", "strict_factscore", "hallucination_rate",
        "citation_error_rate", "kb_support_rate", "parametric_rate",
        "uncertainty_rate"
    ]

    results = {}
    for metric in metrics_to_test:
        groups_data = []
        for group in GROUPS:
            values = [getattr(rm, metric) for rm in report_metrics.get(group, [])]
            groups_data.append(values)

        # ANOVA
        if all(len(g) > 0 for g in groups_data):
            f_stat, p_value = stats.f_oneway(*groups_data)
        else:
            f_stat, p_value = float('nan'), float('nan')

        results[metric] = {
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < ALPHA if not np.isnan(p_value) else False,
            "group_means": {
                group: round(np.mean(vals), 4) if vals else None
                for group, vals in zip(GROUPS, groups_data)
            }
        }

    return results


def run_tukey_hsd(report_metrics: Dict[str, List[ReportMetrics]]) -> Dict:
    """
    Tukey HSD post-hoc test for pairwise group comparisons.

    Returns:
        Dict of Tukey results per metric
    """
    import numpy as np

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        return {"error": "statsmodels not installed. Run: pip install statsmodels"}

    metrics_to_test = ["factscore", "hallucination_rate", "strict_factscore"]
    results = {}

    for metric in metrics_to_test:
        all_values = []
        all_groups = []

        for group in GROUPS:
            for rm in report_metrics.get(group, []):
                all_values.append(getattr(rm, metric))
                all_groups.append(group)

        if len(set(all_groups)) < 2:
            results[metric] = {"error": "Not enough groups"}
            continue

        tukey = pairwise_tukeyhsd(all_values, all_groups, alpha=ALPHA)

        # Convert to readable format
        comparisons = []
        for i in range(len(tukey.groupsunique)):
            for j in range(i + 1, len(tukey.groupsunique)):
                idx = i * len(tukey.groupsunique) + j - (i + 1) * len(tukey.groupsunique) // 2
                if idx < len(tukey.reject):
                    g1 = tukey.groupsunique[i]
                    g2 = tukey.groupsunique[j]
                    comparisons.append({
                        "group1": g1,
                        "group2": g2,
                        "reject": bool(tukey.reject[idx]) if idx < len(tukey.reject) else None,
                        "p_adj": round(float(tukey.pvalues[idx]), 6) if idx < len(tukey.pvalues) else None
                    })

        results[metric] = {
            "comparisons": comparisons,
            "summary": str(tukey)
        }

    return results


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    import numpy as np

    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return float('nan')

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return float('nan')

    d = (mean1 - mean2) / pooled_std

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return round(d, 4)


def run_spearman_correlation(
    report_metrics: Dict[str, List[ReportMetrics]],
    eval_scores: Dict[str, List[float]]
) -> Dict:
    """
    Spearman correlation between hallucination metrics and existing evaluation scores.

    Args:
        report_metrics: {group: [ReportMetrics]}
        eval_scores: {group: [evaluation_dimension_scores]}

    Returns:
        Correlation results
    """
    from scipy import stats
    import numpy as np

    halluc_metrics = ["factscore", "hallucination_rate", "strict_factscore"]
    eval_dimensions = [
        "方法论严谨度", "战略一致性", "逻辑连贯性", "KPI实操性", "治理深度"
    ]

    results = {}

    # Collect paired data
    all_factscores = []
    all_halluc_rates = []
    all_eval_totals = []

    for group in GROUPS:
        rms = report_metrics.get(group, [])
        escores = eval_scores.get(group, [])

        for rm in rms:
            all_factscores.append(rm.factscore)
            all_halluc_rates.append(rm.hallucination_rate)

        for score in escores:
            all_eval_totals.append(score)

    # Compute correlations
    if len(all_factscores) == len(all_eval_totals) and len(all_factscores) > 2:
        r_fact, p_fact = stats.spearmanr(all_factscores, all_eval_totals)
        r_halluc, p_halluc = stats.spearmanr(all_halluc_rates, all_eval_totals)

        results["factscore_vs_eval"] = {
            "r": round(r_fact, 4),
            "p": round(p_fact, 6),
            "significant": p_fact < ALPHA
        }
        results["hallucination_rate_vs_eval"] = {
            "r": round(r_halluc, 4),
            "p": round(p_halluc, 6),
            "significant": p_halluc < ALPHA
        }
    else:
        results["error"] = f"Data length mismatch: factscore={len(all_factscores)}, eval={len(all_eval_totals)}"

    return results


def generate_charts(
    group_metrics_list: List[GroupMetrics],
    anova_results: Dict,
    output_dir: Path
):
    """Generate visualization charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = [gm.architecture for gm in group_metrics_list]
    x = np.arange(len(groups))

    # Chart 1: FActScore comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FActScore
    ax = axes[0, 0]
    means = [gm.factscore_mean for gm in group_metrics_list]
    stds = [gm.factscore_std for gm in group_metrics_list]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax.set_ylabel('FActScore')
    ax.set_title('FActScore by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    # Hallucination Rate
    ax = axes[0, 1]
    means = [gm.hallucination_rate_mean for gm in group_metrics_list]
    stds = [gm.hallucination_rate_std for gm in group_metrics_list]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax.set_ylabel('Hallucination Rate')
    ax.set_title('Hallucination Rate by Group (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    # Strict FActScore
    ax = axes[1, 0]
    means = [gm.strict_factscore_mean for gm in group_metrics_list]
    stds = [gm.strict_factscore_std for gm in group_metrics_list]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax.set_ylabel('Strict FActScore (KB-only)')
    ax.set_title('Strict FActScore by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    # Citation Error Rate
    ax = axes[1, 1]
    means = [gm.citation_error_rate_mean for gm in group_metrics_list]
    stds = [gm.citation_error_rate_std for gm in group_metrics_list]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
    ax.set_ylabel('Citation Error Rate')
    ax.set_title('Citation Error Rate by Group (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right', fontsize=8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "group_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_enhanced_report(
    group_metrics_list: List[GroupMetrics],
    anova_results: Dict,
    tukey_results: Optional[Dict],
    correlation_results: Optional[Dict],
    all_report_metrics: Dict[str, List[ReportMetrics]],
    chi_square_results: Optional[Dict],
    two_way_anova_results: Optional[Dict],
    cohens_d_results: Optional[Dict],
    output_path: Path
):
    """Generate enhanced markdown report with three-tier statistical analysis.

    Tier 1: Verdict-level chi-square (N~700, highest power)
    Tier 2: Chapter-level two-way ANOVA (N=36)
    Tier 3: Run-level one-way ANOVA (N=12)
    """
    lines = []

    lines.append("# Diagnosis Phase Hallucination Statistical Report")
    lines.append("")
    lines.append("## 1. Group Comparison Summary")
    lines.append("")

    # Table 1: Main metrics
    lines.append("| Group | Architecture | FActScore | Strict FS | Halluc Rate | Cite Error | KB Support | Parametric | Uncertain |")
    lines.append("|-------|-------------|-----------|-----------|-------------|------------|------------|------------|-----------|")

    for gm in group_metrics_list:
        lines.append(
            f"| {gm.group} | {gm.architecture} | "
            f"{gm.factscore_mean:.3f}+/-{gm.factscore_std:.3f} | "
            f"{gm.strict_factscore_mean:.3f}+/-{gm.strict_factscore_std:.3f} | "
            f"{gm.hallucination_rate_mean:.3f}+/-{gm.hallucination_rate_std:.3f} | "
            f"{gm.citation_error_rate_mean:.3f}+/-{gm.citation_error_rate_std:.3f} | "
            f"{gm.kb_support_rate_mean:.3f} | "
            f"{gm.parametric_rate_mean:.3f} | "
            f"{gm.uncertainty_rate_mean:.3f} |"
        )

    # Per-report details
    lines.append("")
    lines.append("## 2. Per-Report Results")
    lines.append("")

    for group in GROUPS:
        rms = all_report_metrics.get(group, [])
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Run | Total Facts | FActScore | Halluc Rate | Cite Error |")
        lines.append("|-----|-------------|-----------|-------------|------------|")
        for rm in rms:
            lines.append(
                f"| {rm.run} | {rm.total_facts} | {rm.factscore:.3f} | "
                f"{rm.hallucination_rate:.3f} | {rm.citation_error_rate:.3f} |"
            )
        lines.append("")

    # ================================================================
    # TIER 1: Verdict-level Chi-square
    # ================================================================
    if chi_square_results and "error" not in chi_square_results:
        lines.append("## 3. Tier 1: Verdict-Level Chi-Square Test")
        lines.append("")
        lines.append(f"Tests whether the hallucination proportion differs across 4 groups.")
        lines.append(f"Data: {chi_square_results['total_verdicts']} total atomic facts (diagnosis phase).")
        lines.append("")

        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Pearson Chi-square | {chi_square_results['chi2']:.4f} |")
        lines.append(f"| df | {chi_square_results['dof']} |")
        lines.append(f"| p-value | {chi_square_results['p_value']:.6f} |")
        sig = "**Yes**" if chi_square_results["significant"] else "No"
        lines.append(f"| Significant (alpha=0.05) | {sig} |")
        lines.append(f"| Cramer's V (effect size) | {chi_square_results['cramers_v']:.4f} |")
        lines.append("")

        # Proportions table
        lines.append("### Hallucination Proportions by Group")
        lines.append("")
        lines.append("| Group | Total Facts | Supported | Hallucinated | Halluc Rate |")
        lines.append("|-------|-------------|-----------|--------------|-------------|")
        props = chi_square_results["proportions"]
        for g in GROUPS:
            p = props.get(g, {})
            lines.append(
                f"| {g} | {p.get('total', 0)} | {p.get('supported', 0)} | "
                f"{p.get('hallucinated', 0)} | {p.get('hallucination_rate', 0):.4f} |"
            )
        lines.append("")

        # Pairwise chi-square
        if chi_square_results.get("pairwise"):
            lines.append("### Pairwise Chi-Square (with Bonferroni Correction)")
            lines.append("")
            lines.append("| Comparison | Chi2 | p-raw | p-adjusted | Significant |")
            lines.append("|------------|------|-------|------------|-------------|")
            for pw in chi_square_results["pairwise"]:
                sig = "**Yes**" if pw["significant"] else "No"
                lines.append(
                    f"| {pw['group1']} vs {pw['group2']} | {pw['chi2']:.4f} | "
                    f"{pw['p_raw']:.6f} | {pw['p_bonferroni']:.6f} | {sig} |"
                )
            lines.append("")

    # ================================================================
    # TIER 2: Chapter-level Two-way ANOVA
    # ================================================================
    if two_way_anova_results and "error" not in two_way_anova_results:
        lines.append("## 4. Tier 2: Chapter-Level Two-Way ANOVA (Group x Chapter)")
        lines.append("")
        lines.append(f"Data: {two_way_anova_results['n_observations']} observations "
                     f"(4 groups x 3 runs x 3 chapters).")
        lines.append("")

        if "anova_table" in two_way_anova_results:
            at = two_way_anova_results["anova_table"]
            lines.append("| Source | SS | df | F | p |")
            lines.append("|--------|-----|----|----|------|")
            for source, vals in at.items():
                f_val = f"{vals['f']:.4f}" if vals.get('f') is not None else "N/A"
                p_val = f"{vals['p']:.6f}" if vals.get('p') is not None else "N/A"
                lines.append(f"| {source} | {vals['sum_sq']:.4f} | {vals['df']} | {f_val} | {p_val} |")
            lines.append("")

        # Group main effect
        if "group_effect" in two_way_anova_results:
            ge = two_way_anova_results["group_effect"]
            lines.append(f"**Group main effect**: F={ge.get('f_stat', 'N/A')}, "
                         f"p={ge.get('p_value', 'N/A')}, "
                         f"significant={'**Yes**' if ge.get('significant') else 'No'}")
            lines.append("")

        # Interaction
        if "interaction" in two_way_anova_results:
            ie = two_way_anova_results["interaction"]
            lines.append(f"**Group x Chapter interaction**: F={ie.get('f_stat', 'N/A')}, "
                         f"p={ie.get('p_value', 'N/A')}, "
                         f"significant={'**Yes**' if ie.get('significant') else 'No'}")
            lines.append("")

        # Effect size
        if "eta_squared_partial" in two_way_anova_results:
            eta = two_way_anova_results["eta_squared_partial"]
            lines.append(f"**Partial eta-squared (group)**: {eta:.4f}")
            if eta < 0.01:
                lines.append("- Interpretation: negligible effect")
            elif eta < 0.06:
                lines.append("- Interpretation: small effect")
            elif eta < 0.14:
                lines.append("- Interpretation: medium effect")
            else:
                lines.append("- Interpretation: large effect")
            lines.append("")

        # Post-hoc
        if "tukey_posthoc" in two_way_anova_results and isinstance(two_way_anova_results["tukey_posthoc"], list):
            lines.append("### Tukey HSD Post-Hoc (Chapter-Level)")
            lines.append("")
            lines.append("| Comparison | Mean Diff | p-adjusted | Significant |")
            lines.append("|------------|-----------|------------|-------------|")
            for ph in two_way_anova_results["tukey_posthoc"]:
                sig = "**Yes**" if ph["reject"] else "No"
                lines.append(
                    f"| {ph['group1']} vs {ph['group2']} | {ph['meandiff']:.4f} | "
                    f"{ph['p_adj']:.6f} | {sig} |"
                )
            lines.append("")

        # Per-chapter descriptive
        if "chapter_descriptive" in two_way_anova_results:
            lines.append("### Per-Chapter FActScore (Mean +/- Std)")
            lines.append("")
            ch_desc = two_way_anova_results["chapter_descriptive"]
            lines.append("| Group | Ch1 (PEST) | Ch2 (Strategy) | Ch3 (SWOT) |")
            lines.append("|-------|------------|----------------|------------|")
            for g in GROUPS:
                vals = []
                for ch in range(3):
                    key = f"{g}_ch{ch}"
                    if key in ch_desc:
                        vals.append(f"{ch_desc[key]['mean']:.4f}+/-{ch_desc[key]['std']:.4f}")
                    else:
                        vals.append("N/A")
                lines.append(f"| {g} | {' | '.join(vals)} |")
            lines.append("")

    # ================================================================
    # TIER 3: Run-level ANOVA
    # ================================================================
    lines.append("## 5. Tier 3: Run-Level One-Way ANOVA")
    lines.append("")
    lines.append(f"Data: {sum(len(rms) for rms in all_report_metrics.values())} observations "
                 f"(4 groups x 3 runs).")
    lines.append("")

    lines.append("| Metric | F-statistic | p-value | Significant |")
    lines.append("|--------|------------|---------|-------------|")
    for metric, result in anova_results.items():
        sig = "**Yes**" if result["significant"] else "No"
        lines.append(
            f"| {metric} | {result['f_statistic']} | {result['p_value']} | {sig} |"
        )
    lines.append("")

    # Cohen's d
    if cohens_d_results:
        lines.append("### Cohen's d (Effect Size)")
        lines.append("")
        lines.append("| Comparison | Cohen's d | Direction | Magnitude |")
        lines.append("|------------|-----------|-----------|-----------|")
        for comp, d in cohens_d_results.items():
            import numpy as np
            if np.isnan(d):
                continue
            direction = "positive" if d > 0 else "negative"
            abs_d = abs(d)
            if abs_d < 0.2:
                mag = "negligible"
            elif abs_d < 0.5:
                mag = "small"
            elif abs_d < 0.8:
                mag = "medium"
            else:
                mag = "**large**"
            lines.append(f"| {comp} | {d:.4f} | {direction} | {mag} |")
        lines.append("")

    # Correlation results
    if correlation_results:
        lines.append("## 6. Correlation with Evaluation Scores")
        lines.append("")
        for key, result in correlation_results.items():
            if isinstance(result, dict) and "r" in result:
                lines.append(
                    f"- **{key}**: r={result['r']}, p={result['p']}, "
                    f"significant={'Yes' if result['significant'] else 'No'}"
                )
        lines.append("")

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')

    return '\n'.join(lines)


def generate_markdown_report(
    group_metrics_list: List[GroupMetrics],
    anova_results: Dict,
    tukey_results: Optional[Dict],
    correlation_results: Optional[Dict],
    all_report_metrics: Dict[str, List[ReportMetrics]],
    output_path: Path
):
    """Generate a comprehensive markdown statistical report."""
    lines = []

    lines.append("# Hallucination Evaluation Report (FActScore-aligned)")
    lines.append("")
    lines.append("## 1. Group Comparison Summary")
    lines.append("")

    # Table 1: Main metrics
    lines.append("| Group | Architecture | FActScore | Strict FActScore | Halluc Rate | Cite Error | KB Support | Parametric | Uncertain |")
    lines.append("|-------|-------------|-----------|-----------------|-------------|------------|------------|------------|-----------|")

    for gm in group_metrics_list:
        lines.append(
            f"| {gm.group} | {gm.architecture} | "
            f"{gm.factscore_mean:.3f}±{gm.factscore_std:.3f} | "
            f"{gm.strict_factscore_mean:.3f}±{gm.strict_factscore_std:.3f} | "
            f"{gm.hallucination_rate_mean:.3f}±{gm.hallucination_rate_std:.3f} | "
            f"{gm.citation_error_rate_mean:.3f}±{gm.citation_error_rate_std:.3f} | "
            f"{gm.kb_support_rate_mean:.3f} | "
            f"{gm.parametric_rate_mean:.3f} | "
            f"{gm.uncertainty_rate_mean:.3f} |"
        )

    lines.append("")
    lines.append("## 2. ANOVA Results")
    lines.append("")

    lines.append("| Metric | F-statistic | p-value | Significant |")
    lines.append("|--------|------------|---------|-------------|")

    for metric, result in anova_results.items():
        sig = "Yes **" if result["significant"] else "No"
        lines.append(
            f"| {metric} | {result['f_statistic']} | {result['p_value']} | {sig} |"
        )

    lines.append("")
    lines.append("## 3. Detailed Per-Report Results")
    lines.append("")

    for group in GROUPS:
        rms = all_report_metrics.get(group, [])
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Run | Total Facts | FActScore | Halluc Rate | Cite Error |")
        lines.append("|-----|-------------|-----------|-------------|------------|")

        for rm in rms:
            lines.append(
                f"| {rm.run} | {rm.total_facts} | {rm.factscore:.3f} | "
                f"{rm.hallucination_rate:.3f} | {rm.citation_error_rate:.3f} |"
            )
        lines.append("")

    # Correlation results
    if correlation_results:
        lines.append("## 4. Correlation with Evaluation Scores")
        lines.append("")
        for key, result in correlation_results.items():
            if isinstance(result, dict) and "r" in result:
                lines.append(
                    f"- **{key}**: r={result['r']}, p={result['p']}, "
                    f"significant={'Yes' if result['significant'] else 'No'}"
                )

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')

    return '\n'.join(lines)