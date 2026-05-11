"""Report viewing and export: final report display with TOC."""
from typing import List, Dict

from rag_project.agent.frontend.workspace_tabs import _markdown_to_html


CHAPTER_TITLES = [
    "摘要",
    "Ch1: 宏观政策环境",
    "Ch2: 区域战略分析",
    "Ch3: 行业趋势诊断",
    "战略蓝图",
    "Ch4: 总体战略思路",
    "Ch5: 主责主业举措",
    "Ch6: 创新驱动",
    "Ch7: 产业协同",
    "Ch8: 治理效能",
]


def render_report_viewer(report: str, chapters: List[str], report_evaluation: Dict = None) -> str:
    """Render final report with sidebar TOC and optional evaluation panel."""
    # Convert markdown to HTML with anchor IDs on h1 headers
    report_html = _markdown_to_html(report, add_anchor_ids=True)

    # Build TOC items as clickable links pointing to chapter anchors
    toc_items = ""
    for i, title in enumerate(CHAPTER_TITLES):
        toc_items += f'<a class="toc-item" href="#chapter-{i}">{title}</a>\n'

    # Build report evaluation panel if available
    eval_panel = ""
    if report_evaluation and report_evaluation.get("total_score", 0) > 0:
        eval_panel = _render_report_evaluation(report_evaluation)

    return f"""
    <div class="report-viewer">
        <div class="report-toc">
            <div class="toc-header">目录</div>
            {toc_items}
        </div>
        <div class="report-content">
            <div class="markdown-body markdown-report">{report_html}</div>
            {eval_panel}
        </div>
    </div>
    """


def _render_report_evaluation(evaluation: Dict) -> str:
    """Render the full-report evaluation panel with 5-dimension scores."""
    total = evaluation.get("total_score", 0)
    dims = evaluation.get("dimension_scores", {})
    suggestions = evaluation.get("suggestions", "")

    score_color = "#4ade80" if total >= 75 else "#d4a574" if total >= 60 else "#ef4444"

    dim_rows = ""
    dim_config = [
        ("methodology", "方法论运用与分析框架严谨度"),
        ("strategic_alignment", "战略一致性与外部环境契合度"),
        ("logical_coherence", "逻辑连贯性与战略闭环思维"),
        ("innovation_insight", "创新性与前瞻洞察力"),
        ("organizational_governance", "隐性约束洞察与组织治理深度"),
    ]

    for dim_key, default_label in dim_config:
        dim_data = dims.get(dim_key, {})
        if isinstance(dim_data, dict):
            score = dim_data.get("score", 0)
            label = dim_data.get("label", default_label)
            analysis = dim_data.get("analysis", "")
        else:
            score = dim_data
            label = default_label
            analysis = ""

        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0

        bar_pct = int((score / 20) * 100)
        bar_color = "#4ade80" if bar_pct >= 85 else "#d4a574" if bar_pct >= 70 else "#ef4444"
        analysis_row = f'<div class="dim-analysis">{analysis}</div>' if analysis else ""

        dim_rows += f"""
        <div class="score-row">
            <span class="dim-name">{label}</span>
            <div class="score-bar">
                <div class="score-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
            </div>
            <span class="dim-score">{score}/20</span>
        </div>
        {analysis_row}
        """

    suggestions_html = ""
    if suggestions:
        suggestions_html = f"""
        <div class="eval-suggestions">
            <strong>改进建议:</strong> {suggestions}
        </div>
        """

    return f"""
    <div class="report-eval-panel">
        <div class="eval-header">
            <div class="eval-title">全文质量评估</div>
            <div class="eval-total">
                <span style="font-family:var(--font-display); font-size:2rem; font-weight:700; color:{score_color};">{total}</span>
                <span style="color:var(--text-muted); font-size:1rem;">/100</span>
            </div>
        </div>
        <div class="dimension-scores">{dim_rows}</div>
        {suggestions_html}
    </div>
    """


def render_report_card(report_meta: Dict) -> str:
    """Render a single report card for history list."""
    status = report_meta.get("status", "unknown")
    status_icon = "&#10003;" if status == "completed" else "&#9208;"
    status_text = "已完成" if status == "completed" else "已终止"

    return f"""
    <div class="report-card">
        <div class="report-card-title">&#128196; {report_meta.get("title", "未命名报告")}</div>
        <div class="report-card-meta">
            章节: {report_meta.get("chapters", "0")}/8 &middot;
            字数: {report_meta.get("word_count", "0")} &middot;
            状态: {status_icon} {status_text}
        </div>
    </div>
    """