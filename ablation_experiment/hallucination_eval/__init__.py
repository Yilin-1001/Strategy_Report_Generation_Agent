"""Hallucination evaluation package for ablation experiment reports."""

from .config import GROUPS, NUM_RUNS
from .chapter_splitter import split_report, Chapter
from .citation_parser import CitationParser
from .atomic_decomposer import AtomicDecomposer, AtomicFact
from .fact_verifier import FactVerifier, Verdict
from .metrics import compute_report_metrics, ReportMetrics