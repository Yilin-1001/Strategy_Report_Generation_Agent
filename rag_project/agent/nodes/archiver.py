"""
Archiver node for final report generation with executive summary and strategic blueprint appendix.

Enhanced to support the two-stage strategic deduction architecture:
- Generates 1000-character executive summary in SOE official tone
- Creates comprehensive blueprint appendix
- Assembles final report in professional structure
"""

from datetime import datetime
from typing import Dict, Any, List
import re
import logging

from rag_project.utils.logger import setup_logger
from rag_project.agent.retriever import RAGRetriever

logger = setup_logger(__name__)


def archiver_node(state: Dict[str, Any], llm_manager) -> Dict[str, Any]:
    """
    合并所有章节生成最终报告，包括执行摘要和战略蓝图附录。

    新的工作流程:
    1. 读取 strategic_blueprint 和 context_pool (所有8章)
    2. 生成1000字执行摘要 (国企公文语态)
    3. 组装最终报告:
       - 封面 (标题、日期、主题)
       - 执行摘要 (新增)
       - 目录
       - 第1-8章 (完整内容)
       - 附录: 战略蓝图详述 (新增)

    Args:
        state: Current workflow state containing:
            - context_pool: List of completed chapter contents
            - strategic_blueprint: Strategic blueprint with mission, SWOT, TOWS, pillars, KPIs
            - user_input: User's query/topic for the report
        llm_manager: LLMManager instance for executive summary generation

    Returns:
        Updated state dictionary containing:
            - final_report: Complete report with cover, executive summary, chapters, and appendix

    Example:
        >>> state = {
        ...     "context_pool": ["第一章内容", "第二章内容", ...],
        ...     "strategic_blueprint": {"mission": "...", "swot_analysis": {...}, ...},
        ...     "user_input": "江西交投集团战略规划"
        ... }
        >>> result = archiver_node(state, llm_manager)
        >>> "执行摘要" in result["final_report"]
        True
        >>> "战略蓝图详述" in result["final_report"]
        True
    """
    context_pool = state.get("context_pool", [])
    strategic_blueprint = state.get("strategic_blueprint", {})
    user_input = state.get("user_input", "")

    logger.info(f"Archiver: Generating final report with {len(context_pool)} chapters and executive summary")

    # Step 0: Deduplicate chapters - keep only the LAST approved version of each chapter
    # This ensures that when chapters are revised during human review, only the final
    # approved version appears in the report, not the rejected earlier versions.
    context_pool = _deduplicate_chapters(context_pool)

    # Step 1: Create cover
    cover = _create_cover(user_input)

    # Step 2: Generate executive summary (NEW)
    executive_summary = _generate_executive_summary(
        context_pool=context_pool,
        strategic_blueprint=strategic_blueprint,
        user_input=user_input,
        llm_manager=llm_manager
    )

    # Step 3: Create table of contents
    toc = _create_table_of_contents(context_pool)

    # Step 4: Merge chapters with separator
    chapters_content = "\n\n---\n\n".join(context_pool)

    # Step 5: Create strategic blueprint appendix (NEW)
    blueprint_appendix = _create_blueprint_appendix(strategic_blueprint)

    # Step 6: Combine all parts in order
    full_report = (
        cover +
        executive_summary +
        "\n\n---\n\n" +
        toc +
        "\n\n---\n\n" +
        chapters_content +
        "\n\n---\n\n" +
        blueprint_appendix
    )

    # Step 7: Fix citations - replace generic 来源文档_X with actual filenames
    full_report = _fix_generic_citations(full_report)

    # Step 8: Cross-chapter consistency check (Agent化决策点)
    consistency_issues = _validate_report_consistency(context_pool, llm_manager)
    if consistency_issues:
        # 将一致性问题附加到报告末尾作为审查备注
        issues_text = "\n".join(f"- {issue}" for issue in consistency_issues)
        full_report += f"\n\n---\n**[一致性审查备注]**\n{issues_text}\n"

    return {
        "final_report": full_report
    }


def _validate_report_consistency(chapters: List[str], llm_manager) -> List[str]:
    """
    检查跨章一致性（数据矛盾、重复内容、引用缺失）。

    使用 LLM 快速扫描各章节摘要，识别潜在的跨章问题。

    Args:
        chapters: 各章节内容列表
        llm_manager: LLM 实例

    Returns:
        问题列表（空列表表示无问题）
    """
    if not chapters or len(chapters) < 2:
        return []

    # 提取各章摘要（标题 + 前 300 字）
    chapter_summaries = []
    for ch in chapters:
        lines = ch.strip().split('\n')
        title = next((l.lstrip('# ').strip() for l in lines if l.startswith('#')), 'Unknown')
        preview = ch[:300]
        chapter_summaries.append(f"{title}: {preview}")

    prompt = f"""检查以下报告各章节之间是否存在一致性问题。

{chr(10).join(chapter_summaries)}

请检查:
1. 跨章数据矛盾（同一指标不同数字）
2. 重大内容重复（整段相同）
3. 关键引用缺失（正文中引用但不存在的引用标记）

如果没有问题，返回 OK。如果有问题，每行列出一个问题。"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=300)
        if "OK" in response.upper():
            logger.info("Report consistency check passed")
            return []
        issues = [line.strip() for line in response.strip().split('\n') if line.strip() and line.strip() != 'OK']
        if issues:
            logger.warning(f"Report consistency issues found: {issues}")
        return issues
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")
        return []


def _fix_generic_citations(report: str) -> str:
    """
    Fix remaining [来源: 来源文档_X] citations by searching Milvus for actual source filenames.
    """
    # Find all unique generic citations
    source_refs = set(re.findall(r'\[来源:\s*来源文档_\d+[^\]]*\]', report))
    if not source_refs:
        logger.debug("No generic citations to fix")
        return report

    logger.info(f"Found {len(source_refs)} unique generic citations to fix")

    # Extract unique ref numbers
    ref_nums = set()
    for ref in source_refs:
        m = re.search(r'来源文档_(\d+)', ref)
        if m:
            ref_nums.add(m.group(1))

    # Build ref_num -> source_name mapping via Milvus search
    replacements = {}
    try:
        retriever = RAGRetriever()
    except Exception as e:
        logger.error(f"Cannot init retriever for citation fix: {e}")
        return report

    for ref_num in ref_nums:
        # Get context around first occurrence of this citation
        ref_pattern = rf'\[来源:\s*来源文档_{re.escape(ref_num)}[^\]]*\]'
        match = re.search(ref_pattern, report)
        if not match:
            continue

        # Extract a longer context window (300 chars before the citation)
        start = max(0, match.start() - 300)
        snippet = report[start:match.start()]
        # Remove any remaining citations and clean up
        clean_snippet = re.sub(r'\[来源:[^\]]+\]', '', snippet).strip()
        # Take the last 150 chars (closest to citation) as search query
        search_query = clean_snippet[-150:] if len(clean_snippet) > 150 else clean_snippet
        if len(search_query) < 10:
            continue

        try:
            results = retriever.search(search_query, top_k=1)
            if results:
                text = results[0].get('text', '')
                source_name = None
                # First try === filename === pattern (chunk header)
                for line in text.split('\n')[:10]:
                    line_s = line.strip()
                    if line_s.startswith('===') and line_s.endswith('==='):
                        source_name = line_s.strip('= ').strip()
                        # Remove file extensions and _merged suffix
                        for ext in ('.txt', '.pdf', '.docx', '.doc'):
                            if source_name.endswith(ext):
                                source_name = source_name[:-len(ext)]
                        if source_name.endswith('_merged'):
                            source_name = source_name[:-7]
                        break
                # Fallback: use first short non-empty line
                if not source_name:
                    for line in text.split('\n')[:5]:
                        line_s = line.strip()
                        if line_s and 3 < len(line_s) < 50:
                            source_name = line_s
                            break
                if source_name:
                    replacements[ref_num] = source_name
                    logger.debug(f"来源文档_{ref_num} -> {source_name}")
        except Exception as e:
            logger.debug(f"Citation search error for ref {ref_num}: {e}")

    # Apply replacements
    fixed_report = report
    for ref_num, source_name in replacements.items():
        pattern = rf'\[来源:\s*来源文档_{re.escape(ref_num)}[^\]]*\]'
        fixed_report = re.sub(pattern, f'[来源: {source_name}]', fixed_report)

    remaining = len(re.findall(r'\[来源:\s*来源文档_\d+', fixed_report))
    if remaining > 0:
        logger.warning(f"{remaining} generic citations still remain")
    else:
        logger.info(f"Fixed all {len(replacements)} generic citations")

    return fixed_report


def _create_cover(user_input: str) -> str:
    """
    创建报告封面。

    Args:
        user_input: User's original request/topic

    Returns:
        Formatted cover section
    """
    return f"""# 江西交通投资集团战略规划报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日')}

**主题**: {user_input}

---

"""


def _generate_chapter_brief(chapter: str, max_chars: int = 300) -> str:
    """
    提取章节简要概述（非 LLM 方式）。

    策略: 提取标题 + 各小节标题 + 首段关键句，生成约300字概述。
    避免 LLM 调用（Archiver 已有一次 LLM 调用用于执行摘要）。

    Args:
        chapter: 章节完整文本
        max_chars: 最大字符数

    Returns:
        章节简要概述
    """
    lines = chapter.split('\n')
    title = ""
    section_headers = []
    first_paragraphs = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith('# ') and not title:
            title = stripped.lstrip('# ').strip()
        elif stripped.startswith('## '):
            section_headers.append(stripped.lstrip('# ').strip())
        elif not stripped.startswith('#') and len(first_paragraphs) < 3 and len(stripped) > 20:
            # 取前3个非标题、非空、长度足够的行作为关键句
            first_paragraphs.append(stripped[:100])

    parts = []
    if title:
        parts.append(title)
    if section_headers:
        parts.append("涵盖: " + "、".join(section_headers[:5]))
    if first_paragraphs:
        parts.append("要点: " + "; ".join(first_paragraphs))

    result = " | ".join(parts)
    return result[:max_chars] if len(result) > max_chars else result


def _generate_executive_summary(
    context_pool: List[str],
    strategic_blueprint: Dict,
    user_input: str,
    llm_manager
) -> str:
    """
    生成1000字执行摘要，使用国企公文语态。

    要求:
    - 高层次综合整份报告
    - 强调战略使命和政策对标
    - 专业国企 (SOE) 语言风格
    - 1000字以内
    - 全中文输出

    Args:
        context_pool: List of all completed chapters
        strategic_blueprint: Strategic blueprint with mission, pillars, KPIs
        user_input: Original user request
        llm_manager: LLMManager instance

    Returns:
        Executive summary section (starts with "# 执行摘要")
    """
    mission = strategic_blueprint.get("mission", "")
    pillars = strategic_blueprint.get("strategic_pillars", [])
    kpis = strategic_blueprint.get("kpis", {})

    # Build chapter overview
    chapter_overview = ""
    for i, chapter in enumerate(context_pool):
        # Extract title (first line starting with #)
        lines = chapter.split('\n')
        title = f"第{i+1}章"
        for line in lines:
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        brief = _generate_chapter_brief(chapter, max_chars=300)
        chapter_overview += f"- {title}: {brief}\n"

    prompt = f"""你是一位资深的国企公文写作专家。请为以下省属国企战略规划报告撰写执行摘要（Executive Summary）。

## 报告背景

**主题**: {user_input}

**核心使命**: {mission}

**战略支柱**:
{chr(10).join(f"{i+1}. {p}" for i, p in enumerate(pillars))}

**关键KPI**:
{_format_kpis_for_prompt(kpis)}

## 报告章节概要

{chapter_overview}

## 写作要求

1. **语态**: 国企公文语态，高度凝练、严谨、权威
2. **篇幅**: 1000字以内
3. **结构**:
   - 开篇（100-150字）：政策背景与时代要求
   - 主体（700-850字）：分点阐述战略重点（3-5个要点）
   - 结尾（50-100字）：愿景与承诺
4. **语言特点**:
   - 使用"深入贯彻"、"全面落实"、"牢牢把握"、"扎实推进"等规范表述
   - 强调"服务国家战略"、"承担社会责任"、"推动高质量发展"
   - 避免口语化，使用正式书面语
5. **输出格式**: 使用Markdown格式，以"# 执行摘要"开头
6. **必须使用中文输出**

请撰写执行摘要:"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.5)

        # Ensure it starts with proper header
        if not response.strip().startswith("#"):
            response = "# 执行摘要\n\n" + response

        logger.info(f"Generated executive summary ({len(response)} characters)")
        return response

    except Exception as e:
        logger.error(f"Error generating executive summary: {e}. Using fallback.")
        return _get_fallback_executive_summary(strategic_blueprint)


def _format_kpis_for_prompt(kpis: Dict) -> str:
    """Format KPIs for prompt display."""
    if not kpis:
        return "未设定"

    formatted = []
    for dimension, metrics in kpis.items():
        if isinstance(metrics, dict) and metrics:
            formatted.append(f"- {dimension}:")
            for metric, value in metrics.items():
                formatted.append(f"  - {metric}: {value}")

    return "\n".join(formatted) if formatted else "未设定"


def _get_fallback_executive_summary(strategic_blueprint: Dict) -> str:
    """Generate fallback executive summary when LLM fails."""
    mission = strategic_blueprint.get("mission", "服务国家战略，推动高质量发展")
    pillars = strategic_blueprint.get("strategic_pillars", [])

    summary = f"""# 执行摘要

本报告深入贯彻落实国家交通强省战略，全面分析江西省交通投资集团发展面临的机遇与挑战，系统谋划未来发展蓝图。

## 战略定位与使命

{mission}

## 战略重点

"""
    for i, pillar in enumerate(pillars, 1):
        summary += f"{i}. {pillar}\n"

    summary += """
## 发展愿景

集团将牢牢把握高质量发展这条主线，扎实推进各项战略举措落地见效，为建设交通强省、服务江西经济社会发展作出新的更大贡献。

---
*注：执行摘要为系统自动生成，建议结合具体报告内容进行调整完善。*
"""

    return summary


def _create_table_of_contents(context_pool: List[str]) -> str:
    """
    创建目录。

    Args:
        context_pool: List of all completed chapters

    Returns:
        Table of contents section
    """
    toc = "## 目录\n\n"

    for i, chapter in enumerate(context_pool, 1):
        # Extract chapter title (first line starting with #)
        lines = chapter.split('\n')
        title = f"第{i}章"
        for line in lines:
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        toc += f"{i}. {title}\n"

    return toc


def _create_blueprint_appendix(strategic_blueprint: Dict) -> str:
    """
    创建战略蓝图附录。

    Args:
        strategic_blueprint: Strategic blueprint with all components

    Returns:
        Blueprint appendix section
    """
    if not strategic_blueprint:
        return "## 附录\n\n战略蓝图数据不可用。\n"

    mission = strategic_blueprint.get("mission", "")
    swot = strategic_blueprint.get("swot_analysis", {})
    tows = strategic_blueprint.get("tows_strategies", {})
    pillars = strategic_blueprint.get("strategic_pillars", [])
    kpis = strategic_blueprint.get("kpis", {})

    appendix = f"""# 附录：战略蓝图详述

## 核心使命

{mission}

## SWOT分析矩阵

### 优势 (Strengths)
{_format_list(swot.get('strengths', []))}

### 劣势 (Weaknesses)
{_format_list(swot.get('weaknesses', []))}

### 机会 (Opportunities)
{_format_list(swot.get('opportunities', []))}

### 威胁 (Threats)
{_format_list(swot.get('threats', []))}

## TOWS战略组合

### SO策略 (优势-机会)
{_format_list(tows.get('SO', []))}

### WO策略 (劣势-机会)
{_format_list(tows.get('WO', []))}

### ST策略 (优势-威胁)
{_format_list(tows.get('ST', []))}

### WT策略 (劣势-威胁)
{_format_list(tows.get('WT', []))}

## 战略支柱

{_format_enumerated_list(pillars)}

## 关键绩效指标 (KPIs)

{_format_kpis(kpis)}

---
*报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}*
"""

    return appendix


def _deduplicate_chapters(context_pool: List[str]) -> List[str]:
    """
    Deduplicate chapters in context_pool, keeping only the LAST version of each chapter.

    This is a safety net to prevent duplicate chapters from appearing in the final report.
    Chapters are identified by their title (first # heading line).

    Args:
        context_pool: List of chapter contents (may contain duplicates)

    Returns:
        Deduplicated list with only the last version of each chapter
    """
    if not context_pool:
        return context_pool

    seen_titles = {}  # title -> index in result
    deduplicated = []

    for chapter in context_pool:
        # Extract chapter title from first # heading
        title = None
        for line in chapter.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                title = line.lstrip('#').strip()
                break

        if title is None:
            # No title found, keep as-is (append)
            deduplicated.append(chapter)
            continue

        if title in seen_titles:
            # Duplicate found - replace with latest version
            logger.info(f"Deduplicating chapter: replacing earlier version of '{title}' with latest")
            deduplicated[seen_titles[title]] = chapter
        else:
            seen_titles[title] = len(deduplicated)
            deduplicated.append(chapter)

    if len(deduplicated) != len(context_pool):
        logger.info(f"Deduplicated context_pool: {len(context_pool)} -> {len(deduplicated)} chapters")

    return deduplicated


def _format_list(items: List[str], indent: str = "") -> str:
    """Format a list for display."""
    if not items:
        return f"{indent}*（无）*"

    return "\n".join(f"{indent}- {item}" for item in items if item)


def _format_enumerated_list(items: List[str]) -> str:
    """Format an enumerated list for display."""
    if not items:
        return "*（无）*"

    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _format_kpis(kpis: Dict) -> str:
    """Format KPIs for display."""
    if not kpis:
        return "*（无KPI数据）*"

    formatted = []
    for dimension, metrics in kpis.items():
        if isinstance(metrics, dict) and metrics:
            formatted.append(f"### {dimension}")
            for metric, value in metrics.items():
                formatted.append(f"- **{metric}**: {value}")
            formatted.append("")

    return "\n".join(formatted) if formatted else "*（无KPI数据）*"
