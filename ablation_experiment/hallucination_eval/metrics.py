"""
Metrics Module

Compute FActScore-aligned metrics from verification results.
Metrics follow FActScore definition (Min et al., 2023):
  FActScore = S / N, where S = supported facts, N = total factual facts
"""

from typing import List, Dict
from dataclasses import dataclass, field
from .fact_verifier import Verdict


@dataclass
class ReportMetrics:
    """Metrics for a single report."""
    group: str
    run: int
    total_facts: int = 0           # N: total [FACT] items
    total_analytical: int = 0      # total [ANALYSIS] items
    supported_by_source: int = 0   # X: A-type, citation verified
    supported_by_kb: int = 0       # Y: B-type, KB search verified
    supported_by_parametric: int = 0  # Z: LLM commonsense verified
    contradicted: int = 0
    citation_hallucination: int = 0
    unsupported: int = 0
    unverifiable: int = 0
    total_citations: int = 0
    invalid_citations: int = 0

    # Computed metrics
    factscore: float = 0.0
    strict_factscore: float = 0.0
    hallucination_rate: float = 0.0
    uncertainty_rate: float = 0.0
    citation_error_rate: float = 0.0
    kb_support_rate: float = 0.0
    parametric_rate: float = 0.0

    # Per-chapter breakdown
    chapter_metrics: List[Dict] = field(default_factory=list)


@dataclass
class GroupMetrics:
    """Aggregated metrics for a group (mean ± std across runs)."""
    group: str
    architecture: str
    factscore_mean: float = 0.0
    factscore_std: float = 0.0
    hallucination_rate_mean: float = 0.0
    hallucination_rate_std: float = 0.0
    strict_factscore_mean: float = 0.0
    strict_factscore_std: float = 0.0
    citation_error_rate_mean: float = 0.0
    citation_error_rate_std: float = 0.0
    kb_support_rate_mean: float = 0.0
    parametric_rate_mean: float = 0.0
    uncertainty_rate_mean: float = 0.0


def compute_report_metrics(verdicts: List[Verdict], group: str, run: int) -> ReportMetrics:
    """
    Compute metrics from a list of verdicts for one report.

    FActScore = S / N
    where S = supported_by_source + supported_by_kb + supported_by_parametric
    and N = total_facts
    """
    metrics = ReportMetrics(group=group, run=run)

    # Count by result type
    for v in verdicts:
        if v.fact_type == "ANALYSIS":
            metrics.total_analytical += 1
            continue

        metrics.total_facts += 1

        if v.citation:
            metrics.total_citations += 1

        if v.result == "supported_by_source":
            metrics.supported_by_source += 1
        elif v.result == "supported_by_kb":
            metrics.supported_by_kb += 1
        elif v.result == "supported_by_parametric":
            metrics.supported_by_parametric += 1
        elif v.result == "contradicted":
            metrics.contradicted += 1
        elif v.result == "citation_hallucination":
            metrics.citation_hallucination += 1
            metrics.invalid_citations += 1
        elif v.result == "unsupported":
            metrics.unsupported += 1
        elif v.result == "unverifiable":
            metrics.unverifiable += 1

    # Compute derived metrics
    N = metrics.total_facts
    if N > 0:
        S = metrics.supported_by_source + metrics.supported_by_kb + metrics.supported_by_parametric

        # FActScore (Min et al., 2023): percentage of supported atomic facts
        metrics.factscore = S / N

        # Strict FActScore: only KB-verified
        metrics.strict_factscore = (metrics.supported_by_source + metrics.supported_by_kb) / N

        # Hallucination Rate: 1 - FActScore
        metrics.hallucination_rate = 1.0 - metrics.factscore

        # Uncertainty Rate
        metrics.uncertainty_rate = metrics.unverifiable / N

        # KB Support Rate
        metrics.kb_support_rate = (metrics.supported_by_source + metrics.supported_by_kb) / N

        # Parametric Knowledge Reliance Rate
        metrics.parametric_rate = metrics.supported_by_parametric / N

    # Citation Error Rate
    if metrics.total_citations > 0:
        metrics.citation_error_rate = metrics.invalid_citations / metrics.total_citations

    return metrics


def compute_chapter_metrics(verdicts: List[Verdict], chapter_index: int) -> Dict:
    """Compute metrics for a single chapter."""
    ch_total = 0
    ch_supported = 0
    ch_hallucinated = 0

    for v in verdicts:
        if v.fact_type == "ANALYSIS":
            continue
        ch_total += 1
        if v.result in ("supported_by_source", "supported_by_kb", "supported_by_parametric"):
            ch_supported += 1
        elif v.result in ("contradicted", "citation_hallucination", "unsupported"):
            ch_hallucinated += 1

    return {
        "chapter": chapter_index,
        "total_facts": ch_total,
        "supported": ch_supported,
        "hallucinated": ch_hallucinated,
        "factscore": ch_supported / ch_total if ch_total > 0 else 0.0
    }


def aggregate_group_metrics(report_metrics_list: List[ReportMetrics], group: str) -> GroupMetrics:
    """Aggregate metrics across runs for a group."""
    import numpy as np

    architecture_map = {
        "group0": "Baseline RAG",
        "group1": "Single Agent + Memory",
        "group2": "Multi-Agent (No Memory)",
        "group3": "Full System"
    }

    gm = GroupMetrics(
        group=group,
        architecture=architecture_map.get(group, group)
    )

    if not report_metrics_list:
        return gm

    scores = {key: [] for key in [
        "factscore", "hallucination_rate", "strict_factscore",
        "citation_error_rate", "kb_support_rate", "parametric_rate",
        "uncertainty_rate"
    ]}

    for rm in report_metrics_list:
        for key in scores:
            scores[key].append(getattr(rm, key))

    gm.factscore_mean = np.mean(scores["factscore"])
    gm.factscore_std = np.std(scores["factscore"], ddof=1) if len(scores["factscore"]) > 1 else 0
    gm.hallucination_rate_mean = np.mean(scores["hallucination_rate"])
    gm.hallucination_rate_std = np.std(scores["hallucination_rate"], ddof=1) if len(scores["hallucination_rate"]) > 1 else 0
    gm.strict_factscore_mean = np.mean(scores["strict_factscore"])
    gm.strict_factscore_std = np.std(scores["strict_factscore"], ddof=1) if len(scores["strict_factscore"]) > 1 else 0
    gm.citation_error_rate_mean = np.mean(scores["citation_error_rate"])
    gm.citation_error_rate_std = np.std(scores["citation_error_rate"], ddof=1) if len(scores["citation_error_rate"]) > 1 else 0
    gm.kb_support_rate_mean = np.mean(scores["kb_support_rate"])
    gm.parametric_rate_mean = np.mean(scores["parametric_rate"])
    gm.uncertainty_rate_mean = np.mean(scores["uncertainty_rate"])

    return gm