"""Report viewing and export: final report display with TOC."""
from typing import List, Dict


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


def render_report_viewer(report: str, chapters: List[str]) -> str:
    """Render final report with sidebar TOC."""
    toc_items = ""
    for i, title in enumerate(CHAPTER_TITLES):
        toc_items += f'<div class="toc-item" data-chapter="{i}">{title}</div>'

    return f"""
    <div class="report-viewer">
        <div class="report-toc">
            <div class="toc-header">目录</div>
            {toc_items}
        </div>
        <div class="report-content">
            <div class="markdown-body">{report}</div>
        </div>
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