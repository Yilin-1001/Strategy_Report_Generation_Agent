"""
Report Formatting Utilities

Shared functions for generating document summaries, formatting reports,
and extracting metadata. Used by all ablation groups.
"""

import re
from typing import Dict, List, Any
from datetime import datetime

from ablation_experiment.config import TEST_QUERY


def smart_extract(text: str, budget: int = 900) -> str:
    """
    Smart text extraction that preserves head and tail content.
    Source: analyst.py:187-210

    Args:
        text: Original document text
        budget: Total character budget

    Returns:
        Extracted text within budget
    """
    if len(text) <= budget:
        return text

    head_budget = int(budget * 0.6)
    tail_budget = budget - head_budget - 20

    head = text[:head_budget]
    tail = text[-tail_budget:] if tail_budget > 0 else ""

    return f"{head}\n\n...[中间内容省略]...\n\n{tail}"


def generate_document_summary(documents: List[Dict[str, Any]], total_budget: int = 18000) -> str:
    """
    Generate document summary with head-tail extraction and dynamic budget allocation.
    Source: analyst.py:213-285

    Args:
        documents: List of document dictionaries
        total_budget: Total character budget

    Returns:
        Formatted document summary string
    """
    if not documents:
        return "No documents retrieved for analysis."

    metadata_overhead = 80
    per_doc_budget = min(900, (total_budget // max(len(documents), 1)) - metadata_overhead)

    if per_doc_budget < 400:
        max_docs = total_budget // (400 + metadata_overhead)
        documents = documents[:max(1, max_docs)]
        per_doc_budget = (total_budget // len(documents)) - metadata_overhead

    summary_parts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        source = metadata.get("source", "")
        page = metadata.get("page_number", "N/A")
        title = metadata.get("title", "")
        doc_type = metadata.get("doc_type", "")

        file_name = None
        if source and str(source).strip():
            source_clean = str(source).strip()
            if '/' in source_clean:
                file_name = source_clean.split('/')[-1]
            elif '\\' in source_clean:
                file_name = source_clean.split('\\')[-1]
            else:
                file_name = source_clean
            for ext in ('.txt', '.pdf', '.docx', '.doc'):
                if file_name.endswith(ext):
                    file_name = file_name[:-len(ext)]
                    break
        if not file_name and title and str(title).strip():
            file_name = str(title).strip()[:80]
        if not file_name and doc_type and str(doc_type).strip():
            file_name = f"{str(doc_type).strip()}文档"
        if not file_name:
            file_name = f"来源文档_{i}"

        text_preview = smart_extract(text, budget=per_doc_budget)

        if page and str(page) != "N/A":
            doc_entry = f"Document {i} [来源: {file_name}, 第{page}页]:\n{text_preview}"
        else:
            doc_entry = f"Document {i} [来源: {file_name}]:\n{text_preview}"
        summary_parts.append(doc_entry)

    return "\n\n".join(summary_parts)


def extract_filename_mapping(document_summary: str) -> Dict[str, str]:
    """
    Extract Document number to filename mapping from document summary.
    Source: writer.py:110-139

    Args:
        document_summary: Document summary string

    Returns:
        Dict mapping "Document X" to filename
    """
    mapping = {}
    new_pattern = r'Document (\d+)\s*\[来源:\s*([^,\]]+)'
    new_matches = re.findall(new_pattern, document_summary)
    for doc_num, filename in new_matches:
        mapping[f"Document {doc_num}"] = filename.strip()
    return mapping


def extract_chapter_question(chapter_title: str) -> str:
    """
    Extract research question from chapter title.
    Source: prep_chapter.py:228-233

    Args:
        chapter_title: Full chapter title (e.g., "第一章：宏观政策环境与时代要求")

    Returns:
        Research question string
    """
    if "：" in chapter_title or ":" in chapter_title:
        separator = "：" if "：" in chapter_title else ":"
        parts = chapter_title.split(separator, 1)
        return parts[1].strip() if len(parts) > 1 else chapter_title
    return chapter_title


def build_chapter_knowledge(chapter_title: str, key_facts: Any, insights: List[str]) -> str:
    """
    Build knowledge text from chapter analysis for context compression.
    Source: prep_chapter.py:47-61

    Args:
        chapter_title: Chapter title
        key_facts: Key facts (list or dict)
        insights: Insights list

    Returns:
        Formatted knowledge text
    """
    new_knowledge = f"\n【{chapter_title}】\n"

    if isinstance(key_facts, dict):
        for category, facts in key_facts.items():
            if isinstance(facts, list):
                new_knowledge += f"  {category}: {'; '.join(str(f) for f in facts[:5])}\n"
            else:
                new_knowledge += f"  {category}: {facts}\n"
    elif isinstance(key_facts, list):
        new_knowledge += "  关键事实: " + "; ".join(str(f) for f in key_facts[:8]) + "\n"

    if insights:
        new_knowledge += "  核心洞察: " + "; ".join(str(i) for i in insights[:5]) + "\n"

    return new_knowledge


def assemble_report(
    chapters: List[str],
    user_input: str,
    strategic_blueprint: Dict = None,
    include_executive_summary: bool = False,
    executive_summary: str = ""
) -> str:
    """
    Assemble final report from chapters with cover, TOC, and optional blueprint appendix.

    Args:
        chapters: List of chapter content strings
        user_input: Original user request
        strategic_blueprint: Optional strategic blueprint for appendix
        include_executive_summary: Whether to include executive summary section
        executive_summary: Pre-generated executive summary text

    Returns:
        Complete report string
    """
    # Cover
    cover = f"""# 江西交通投资集团战略规划报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日')}

**主题**: {user_input}

---

"""

    # Table of contents
    toc = "## 目录\n\n"
    for i, chapter in enumerate(chapters, 1):
        lines = chapter.strip().split('\n')
        title = f"第{i}章"
        for line in lines:
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        toc += f"{i}. {title}\n"

    # Chapters
    chapters_content = "\n\n---\n\n".join(chapters)

    # Build report
    parts = [cover, toc, "\n\n---\n\n", chapters_content]

    # Optional executive summary
    if include_executive_summary and executive_summary:
        parts.insert(1, executive_summary + "\n\n---\n\n")

    # Optional blueprint appendix
    if strategic_blueprint:
        appendix = build_blueprint_appendix(strategic_blueprint)
        parts.append("\n\n---\n\n" + appendix)

    return "".join(parts)


def build_blueprint_appendix(strategic_blueprint: Dict) -> str:
    """Build strategic blueprint appendix section."""
    if not strategic_blueprint:
        return ""

    mission = strategic_blueprint.get("mission", "")
    pillars = strategic_blueprint.get("strategic_pillars", [])
    kpis = strategic_blueprint.get("kpis", {})

    appendix = f"""# 附录：战略蓝图详述

## 核心使命

{mission}

## 战略支柱

"""
    for i, pillar in enumerate(pillars, 1):
        appendix += f"{i}. {pillar}\n"

    if kpis:
        appendix += "\n## 关键绩效指标 (KPIs)\n\n"
        for dimension, metrics in kpis.items():
            if isinstance(metrics, dict):
                appendix += f"### {dimension}\n"
                for metric, value in metrics.items():
                    appendix += f"- **{metric}**: {value}\n"
                appendix += "\n"

    appendix += f"\n---\n*报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}*\n"

    return appendix
