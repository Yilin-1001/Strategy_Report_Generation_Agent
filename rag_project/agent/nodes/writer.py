"""
Writer Node - Generates chapter drafts from facts and insights

This node synthesizes the key facts and insights extracted by the Analyst node
to generate a comprehensive chapter draft with board-level professional tone.
"""

import re
import logging
from typing import Dict, Any, List

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def _validate_draft(draft: str, min_chars: int = 1000, max_chars: int = 2500) -> Dict:
    """
    检查草稿质量，返回校验结果。

    Args:
        draft: 生成的草稿文本
        min_chars: 最小中文字符数（默认1000，适度丰富）
        max_chars: 最大中文字符数（默认2500，允许更详尽的内容）

    Returns:
        Dict with validation results including char_count, in_range, citation_count, needs_revision
    """
    # 统计中文字符数（排除 markdown 标记和空行）
    chinese_chars = sum(1 for c in draft if '\u4e00' <= c <= '\u9fff')
    citations = len(re.findall(r'\[来源:', draft))
    has_structure = '## ' in draft

    return {
        "char_count": chinese_chars,
        "in_range": min_chars <= chinese_chars <= max_chars,
        "citation_count": citations,
        "has_structure": has_structure,
        "needs_revision": chinese_chars > max_chars or citations < 1
    }


def _revise_draft(draft: str, validation: Dict, llm_manager) -> str:
    """
    当草稿超标时，用 LLM 精简。

    Args:
        draft: 原始草稿
        validation: 校验结果
        llm_manager: LLM管理器

    Returns:
        精简后的草稿文本
    """
    issues = []
    if validation["char_count"] > 2500:
        issues.append(f"当前{validation['char_count']}字，需要精简到1200-2500字")
    if validation["citation_count"] < 1:
        issues.append("缺少引用标记，需要添加[来源: ...]格式引用")

    prompt = f"""请精简以下章节草稿。
问题: {'; '.join(issues)}
要求: 保留所有关键论点和数据，删除冗余表述，控制在1200-2500字。
必须保留原有的引用标记[来源: ...]。

**重要**: 只输出精简后的章节正文，不要添加任何"改写说明"、"修订说明"等元信息或注释。

原文:
{draft}

请输出精简后的章节:"""

    revised = llm_manager.invoke(prompt, temperature=0.3, max_tokens=2048)

    # Post-process: remove any revision notes that LLM might have added
    revised = _strip_revision_notes(revised)

    return revised


def _strip_revision_notes(text: str) -> str:
    """
    移除 LLM 可能添加的改写说明/修订说明等元信息。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # Remove common revision note patterns
    patterns = [
        r'\n*---\n*\*\*改写说明\*\*[^\n]*\n.*?(?=\n---\n|\n#\s|$)',
        r'\n*---\n*\*\*修订说明\*\*[^\n]*\n.*?(?=\n---\n|\n#\s|$)',
        r'\n*---\n*\*\*写说明\*\*[^\n]*\n.*?(?=\n---\n|\n#\s|$)',
        r'\n*\*\*改写说明\*\*[：:].*?(?=\n---\n|\n#\s|$)',
        r'\n*\*\*修订说明\*\*[：:].*?(?=\n---\n|\n#\s|$)',
        r'如果您需要更简洁或更详细的版本.*?优化调整。',
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _extract_filename_mapping(document_summary: str) -> Dict[str, str]:
    """
    从文档摘要中提取Document编号到文件名的映射

    支持两种格式:
    - Document X [来源: filename, 第Y页]:  (新格式)
    - Document X (Source: filename, Page: Y):  (旧格式)

    Args:
        document_summary: Analyst节点生成的文档摘要字符串

    Returns:
        Dict mapping "Document X" to filename
    """
    mapping = {}

    # New format: "Document X [来源: filename, 第Y页]:" or "Document X [来源: filename]:"
    new_pattern = r'Document (\d+)\s*\[来源:\s*([^,\]]+)'
    new_matches = re.findall(new_pattern, document_summary)
    for doc_num, filename in new_matches:
        mapping[f"Document {doc_num}"] = filename.strip()

    # Old format: "Document X (Source: filename, Page: Y):" (backward compat)
    if not mapping:
        old_pattern = r'Document (\d+)\s*\(Source:\s*([^,)]+)'
        old_matches = re.findall(old_pattern, document_summary)
        for doc_num, filename in old_matches:
            mapping[f"Document {doc_num}"] = filename.strip()

    return mapping


def _replace_document_x_with_filenames(
    draft: str,
    filename_mapping: Dict[str, str],
    document_summary: str
) -> str:
    """
    后处理：将draft中的"Document X"和"来源文档_X"引用替换为实际文件名

    如果filename_mapping中的文件名本身也是通用引用(来源文档_X),
    说明analyst无法获取真实文件名，此时跳过替换。

    Args:
        draft: LLM生成的章节草稿
        filename_mapping: Document编号到文件名的映射
        document_summary: 文档摘要（用于提取页码信息）

    Returns:
        替换后的草稿文本
    """
    if not filename_mapping:
        logger.debug("No filename mapping available, skipping citation replacement")
        return draft

    # Filter out generic filenames from mapping - only replace with real filenames
    real_mapping = {}
    for doc_ref, filename in filename_mapping.items():
        if filename and not re.match(r'^来源文档_\d+$', filename.strip()):
            real_mapping[doc_ref] = filename
        else:
            logger.debug(f"Skipping generic filename for {doc_ref}: {filename}")

    if not real_mapping:
        logger.warning("All filenames are generic (来源文档_X), cannot replace citations")
        return draft

    # 从document_summary中提取每个文档的页码信息
    page_mapping = {}
    page_pattern = r'Document (\d+)\s*\[来源:\s*[^,\]]+,\s*第(\d+)页\]'
    for doc_num, page_num in re.findall(page_pattern, document_summary):
        page_mapping[f"Document {doc_num}"] = page_num

    total_replacements = 0

    for doc_ref, filename in real_mapping.items():
        doc_num = doc_ref.replace("Document ", "")
        page_num = page_mapping.get(doc_ref)

        if page_num:
            replacement = f'[来源: {filename}, 第{page_num}页]'
        else:
            replacement = f'[来源: {filename}]'

        patterns = [
            rf'\[(?:来源|Source):\s*{re.escape(doc_ref)}(?:[^\]]*)?\]',
            rf'\[(?:来源|Source):\s*来源文档_{re.escape(doc_num)}(?:[^\]]*)?\]',
        ]

        for pattern in patterns:
            before_count = len(re.findall(pattern, draft))
            if before_count > 0:
                draft = re.sub(pattern, replacement, draft)
                total_replacements += before_count

    if total_replacements > 0:
        logger.info(f"Replaced {total_replacements} citation(s) with filenames")
    else:
        logger.debug("No Document X / 来源文档_X citations found to replace")

    return draft


def writer_node(state: Dict[str, Any], llm_manager) -> Dict[str, Any]:
    """
    Generate chapter draft from key facts and insights with strategic model constraints.

    该节点会根据当前章节的元数据（analysis_model, phase）以及是否处于推演阶段，
    动态注入战略蓝图约束和模型特定的写作指令，确保生成的章节内容符合战略分析框架。

    推演阶段特殊处理:
    - 如果 phase == "initiatives" 且 strategic_blueprint 已批准
    - 将自动注入核心使命、战略支柱、KPI等战略蓝图内容
    - 要求章节明确说明如何支撑战略目标

    Args:
        state: Current workflow state containing:
            - chapter_title: Title of the current chapter
            - chapter_question: The research question
            - chapter_context: Optional context for the chapter
            - chapter_scratchpad: Dict with intermediate results including:
                - key_facts: List of extracted key facts (from Analyst)
                - insights: List of generated insights (from Analyst)
                - context_summary: Optional summary context
                - analysis_model_used: The strategic model applied
            - global_plan: List[Dict] with chapter metadata
            - current_chapter_index: Index of current chapter
            - strategic_blueprint: Dict with mission, pillars, KPIs (if approved)
            - current_phase: Current phase (diagnosis/initiatives)
        llm_manager: LLMManager instance for content generation

    Returns:
        Dict with updates:
            - current_draft: Generated chapter draft text

    Example:
        >>> state = {
        ...     "chapter_title": "第五章：主责主业建设举措",
        ...     "global_plan": [{"analysis_model": "BCG波士顿矩阵", "phase": "initiatives"}],
        ...     "strategic_blueprint": {"mission": "...", "approved": True},
        ...     "current_phase": "initiatives",
        ...     "chapter_scratchpad": {"key_facts": [...], "insights": [...]}
        ... }
        >>> result = writer_node(state, llm_manager)
        >>> "现金牛业务" in result["current_draft"]  # BCG model reflected
        True
        >>> "战略蓝图" in result["current_draft"] or "支撑" in result["current_draft"]
        True
    """
    chapter_title = state.get("chapter_title", "")
    chapter_question = state.get("chapter_question", "")
    chapter_context = state.get("chapter_context", "")
    chapter_scratchpad = state.get("chapter_scratchpad", {})

    # Get current chapter metadata for model injection
    global_plan = state.get("global_plan", [])
    current_chapter_index = state.get("current_chapter_index", 0)

    chapter_metadata = global_plan[current_chapter_index] if current_chapter_index < len(global_plan) else {}
    analysis_model = chapter_metadata.get("analysis_model", "")
    phase = chapter_metadata.get("phase", "")

    # Get strategic blueprint if in initiatives phase
    strategic_blueprint = state.get("strategic_blueprint", {})
    is_initiatives_phase = phase == "initiatives" and strategic_blueprint.get("approved", False)

    # Get key facts and insights from scratchpad (processed by Analyst)
    key_facts = chapter_scratchpad.get("key_facts", [])
    insights = chapter_scratchpad.get("insights", [])
    context_summary = state.get("context_summary", "")  # Read from top-level state (set by prep_chapter)
    document_summary = chapter_scratchpad.get("document_summary", "")
    analysis_model_used = chapter_scratchpad.get("analysis_model_used", analysis_model)
    revision_feedback = chapter_scratchpad.get("revision_feedback")

    logger.info(f"Writer node generating chapter '{chapter_title}' "
                f"with {len(key_facts)} facts and {len(insights)} insights "
                f"(model: {analysis_model_used}, phase: {phase}, blueprint: {'Yes' if is_initiatives_phase else 'No'}, "
                f"revision: {'Yes' if revision_feedback else 'No'})")

    # Step 1: Generate chapter draft using LLM with strategic constraints
    try:
        draft = _generate_chapter_draft(
            chapter_title=chapter_title,
            chapter_question=chapter_question,
            chapter_context=chapter_context,
            key_facts=key_facts,
            insights=insights,
            context_summary=context_summary,
            document_summary=document_summary,
            analysis_model=analysis_model_used,
            strategic_blueprint=strategic_blueprint if is_initiatives_phase else None,
            phase=phase,
            revision_feedback=revision_feedback,
            llm_manager=llm_manager
        )

        logger.info(f"Generated chapter draft ({len(draft)} characters)")

    except Exception as e:
        logger.error(f"Error generating chapter draft: {e}. Using fallback.")
        draft = _get_fallback_draft(chapter_title, key_facts, insights)

    # Step 1.5: Self-validation - check draft quality and revise if needed (max 1 retry)
    validation = _validate_draft(draft)
    logger.info(f"Draft validation: {validation}")
    if validation["needs_revision"]:
        logger.info(f"Draft needs revision: char_count={validation['char_count']}, citations={validation['citation_count']}. Attempting self-revision...")
        try:
            revised = _revise_draft(draft, validation, llm_manager)
            # Only adopt revision if it's actually shorter
            revised_chars = sum(1 for c in revised if '\u4e00' <= c <= '\u9fff')
            if revised_chars < validation["char_count"]:
                logger.info(f"Self-revision succeeded: {validation['char_count']} -> {revised_chars} chars")
                draft = revised
            else:
                logger.info(f"Revision not shorter ({revised_chars} chars), keeping original")
        except Exception as e:
            logger.warning(f"Self-revision failed: {e}. Using original draft.")

    # Step 1.6: Strip any LLM-added meta notes (改写说明 etc.)
    draft = _strip_revision_notes(draft)

    # Step 2: Post-process - Replace generic citations with actual filenames
    try:
        filename_mapping = _extract_filename_mapping(document_summary)
        if filename_mapping:
            draft = _replace_document_x_with_filenames(
                draft=draft,
                filename_mapping=filename_mapping,
                document_summary=document_summary
            )
            logger.info(f"Post-processed citations with filename mapping")
        else:
            logger.warning("No filename mapping found, skipping citation post-processing")
    except Exception as e:
        logger.error(f"Error during citation post-processing: {e}. Continuing with original draft.")

    # Step 3: Return only the current_draft (do not modify scratchpad)
    return {
        "current_draft": draft
    }


def _generate_chapter_draft(
    chapter_title: str,
    chapter_question: str,
    chapter_context: str,
    key_facts: List[str],
    insights: List[str],
    context_summary: str,
    document_summary: str,
    analysis_model: str = "",
    strategic_blueprint: Dict = None,
    phase: str = "",
    revision_feedback: Dict = None,
    llm_manager = None
) -> str:
    """
    使用LLM基于事实和洞察生成章节草稿，支持战略模型约束和蓝图注入。

    Args:
        chapter_title: Title of the chapter
        chapter_question: The research question
        chapter_context: Optional context
        key_facts: List of key facts extracted by Analyst
        insights: List of insights generated by Analyst
        context_summary: Optional context summary
        document_summary: Summary of retrieved documents
        analysis_model: Strategic analysis model used
        strategic_blueprint: Strategic blueprint (for initiatives phase)
        phase: Current phase (diagnosis/initiatives)
        revision_feedback: Revision feedback from reviewer
        llm_manager: LLMManager instance

    Returns:
        Generated chapter draft text

    Raises:
        Exception: If LLM call fails
    """
    # Generate prompt for chapter writing with strategic context
    prompt = _generate_writing_prompt(
        chapter_title=chapter_title,
        chapter_question=chapter_question,
        chapter_context=chapter_context,
        key_facts=key_facts,
        insights=insights,
        context_summary=context_summary,
        document_summary=document_summary,
        analysis_model=analysis_model,
        strategic_blueprint=strategic_blueprint,
        phase=phase,
        revision_feedback=revision_feedback
    )

    # Invoke LLM with higher temperature for creative writing
    response = llm_manager.invoke(prompt, temperature=0.7)

    return response


def _generate_writing_prompt(
    chapter_title: str,
    chapter_question: str,
    chapter_context: str,
    key_facts: List[str],
    insights: List[str],
    context_summary: str,
    document_summary: str,
    analysis_model: str = "",
    strategic_blueprint: Dict = None,
    phase: str = "",
    revision_feedback: Dict = None
) -> str:
    """
    生成LLM章节写作提示词，支持战略模型约束和蓝图注入。

    Args:
        chapter_title: Title of the chapter
        chapter_question: The research question
        chapter_context: Optional context
        key_facts: List of key facts
        insights: List of insights
        context_summary: Optional context summary
        document_summary: Summary of retrieved documents
        analysis_model: Strategic analysis model used
        strategic_blueprint: Strategic blueprint (for initiatives phase)
        phase: Current phase (diagnosis/initiatives)
        revision_feedback: Revision feedback from reviewer

    Returns:
        Formatted prompt string with strategic constraints
    """
    # Extract filename mapping from document_summary
    filename_mapping = _extract_filename_mapping(document_summary)

    # Build facts section (handle structured key_facts from models like PEST/SWOT)
    facts_section = _build_facts_section(key_facts, analysis_model)

    # Build insights section
    insights_section = ""
    if insights:
        insights_section = "\n重要洞察 (Key Insights):\n" + "\n".join(f"- {insight}" for insight in insights)
    else:
        insights_section = "\n重要洞察: 无"

    # Build context section
    context_parts = []
    if chapter_context:
        context_parts.append(f"背景: {chapter_context}")
    if context_summary:
        context_parts.append(
            f"前序章节已确立的关键信息（请在写作中保持一致，不要矛盾）:\n{context_summary}"
        )
    if document_summary:
        # Analyst 已对文档做了预算控制，直接传递完整摘要
        context_parts.append(f"参考文档摘要:\n{document_summary}")

    context_section = "\n".join(context_parts) if context_parts else "无额外背景信息"

    # Build filename reference section - show actual filenames for direct citation
    filename_ref = ""
    if filename_mapping:
        filename_ref = "\n可用来源文件名列表 (引用时直接使用文件名):\n"
        for doc_ref, filename in filename_mapping.items():
            filename_ref += f"- {filename}\n"
    else:
        filename_ref = "\n注意: 无可用文件名映射，请在引用时标注[来源: 相关文档名]\n"

    # Build strategic blueprint constraint (for initiatives phase)
    blueprint_constraint = _build_blueprint_constraint(strategic_blueprint)

    # Build model-specific writing instruction
    model_instruction = _get_model_writing_instruction(analysis_model, phase)

    # Build revision feedback section
    revision_section = ""
    if revision_feedback:
        issues = revision_feedback.get("issues", [])
        comments = revision_feedback.get("comments", "")
        previous_draft = revision_feedback.get("previous_draft_summary", "")
        dimension_scores = revision_feedback.get("dimension_scores", {})
        improvement_hints = revision_feedback.get("improvement_hints", {})
        writer_hint = improvement_hints.get("writer", "")

        issues_lines = "\n".join(f"   - {issue}" for issue in issues) if issues else f"   - {comments}"
        draft_preview = f"\n上一轮草稿摘录:\n{previous_draft}" if previous_draft else ""

        # 构建维度评分摘要
        dim_summary = ""
        if dimension_scores:
            dim_items = [f"   - {k}: {v}分" for k, v in dimension_scores.items() if v is not None]
            dim_summary = "\n维度评分:\n" + "\n".join(dim_items)

        # 构建针对性改进方向
        hint_section = ""
        if writer_hint:
            hint_section = f"\n**针对性改进方向**: {writer_hint}"

        revision_section = f"""
## 上一轮评审反馈（请针对这些问题改进写作）

评审分数: {revision_feedback.get('score', 'N/A')}/100
主要问题:
{issues_lines}{dim_summary}
{draft_preview}{hint_section}

**改进要求**: 请针对上述问题重写章节，避免重复同样的错误。
"""

    # Build cross-chapter consistency instruction (only when context_summary exists)
    consistency_instruction = ""
    if context_summary:
        consistency_instruction = """
7. **跨章节一致性**:
   - 如果前序章节已提及具体数字（如投资额、增长率），必须使用相同数字
   - 不要重复前序章节已有的内容，而是在其基础上深化
   - 如果发现前序结论与当前分析矛盾，以当前分析为准但需说明变化原因
"""

    # 硬性字数约束：放在 prompt 末尾，利用模型的"近因效应"增强约束力
    hard_limit = """
## ⚠️ 字数指导（写作参考）

- **目标范围**: 1200-2500个中文字符
- **最佳长度**: 1500-2200个中文字符
- **计数方式**: 统计正文中的中文字符数（不含标题、标记符号）
- **写作原则**: 宁可充分展开论点，也不要因字数限制而牺牲分析深度
- **控制技巧**: 开头概述2-3句话，每个分析维度300-500字，结尾总结1-2句话

注意：优先保证分析质量和论证完整性，字数在合理范围内即可。
"""

    return f"""你是一位资深的国企战略规划报告撰写专家。

## 章节信息

章节标题: {chapter_title}

研究问题: {chapter_question}

**指定分析模型**: {analysis_model}
**当前阶段**: {phase}

{context_section}

{facts_section}

{insights_section}

{filename_ref}

{blueprint_constraint}

{model_instruction}

{revision_section}
## 通用写作要求

1. **语态**: 国企公文语态，高度凝练、严谨、权威
   - 使用规范表述："深入贯彻"、"全面落实"、"扎实推进"、"牢牢把握"
   - 强调"服务国家战略"、"承担社会责任"、"推动高质量发展"
   - 避免口语化，使用正式书面语

2. **结构**:
   - 开头 (2-3句话): 概述本章主题和核心观点
   - 主体 (2-3个小节): 详细阐述，使用小标题组织内容
   - 结尾 (1-2句话): 总结要点或提出展望

3. **引用要求**:
   - 在适当位置添加引用标记，必须使用上面"可用来源文件名列表"中的实际文件名
   - 禁止使用"Document X"、"来源文档_X"或任何编号简称
   - 格式: [来源: 文件名, 第X页] 或 [来源: 文件名]
   - 引用应该与提供的关键事实相关联

4. **格式要求**:
   - 使用markdown格式
   - 必须以章节标题开头（使用 # {chapter_title}）
   - 主体部分使用 ## 标记小节标题
   - 段落之间空一行

5. **输出语言**: 必须使用中文输出
{consistency_instruction}
{hard_limit}
现在请撰写该章节内容:"""


def _build_facts_section(key_facts: Any, analysis_model: str) -> str:
    """
    构建关键事实部分，支持结构化key_facts（如SWOT/PEST模型的字典格式）。

    Args:
        key_facts: Key facts (can be list or dict)
        analysis_model: Analysis model name

    Returns:
        Formatted facts section string
    """
    if not key_facts:
        return "\n关键事实: 无"

    # If key_facts is a dict (structured by model like PEST/SWOT)
    if isinstance(key_facts, dict):
        facts_lines = ["\n关键事实 (按分析模型框架组织):"]
        for category, facts in key_facts.items():
            if isinstance(facts, list):
                facts_lines.append(f"\n### {category}")
                for fact in facts:
                    facts_lines.append(f"- {fact}")
            else:
                facts_lines.append(f"\n### {category}\n- {facts}")
        return "\n".join(facts_lines)

    # If key_facts is a simple list
    facts_section = "\n关键事实 (Key Facts):\n" + "\n".join(f"- {fact}" for fact in key_facts)
    return facts_section


def _build_blueprint_constraint(strategic_blueprint: Dict = None) -> str:
    """
    构建战略蓝图约束文本（用于推演阶段）。

    Args:
        strategic_blueprint: Strategic blueprint dict with mission, pillars, KPIs

    Returns:
        Formatted blueprint constraint section
    """
    if not strategic_blueprint:
        return ""

    mission = strategic_blueprint.get("mission", "")
    pillars = strategic_blueprint.get("strategic_pillars", [])
    kpis = strategic_blueprint.get("kpis", {})

    constraint_parts = ["## 战略蓝图约束（本章节必须遵循）\n"]

    if mission:
        constraint_parts.append(f"**核心使命**: {mission}\n")

    if pillars:
        constraint_parts.append("**战略支柱**:")
        for i, pillar in enumerate(pillars, 1):
            constraint_parts.append(f"{i}. {pillar}")
        constraint_parts.append("")

    if kpis:
        constraint_parts.append("**关键绩效指标 (KPIs)**:")
        for dimension, metrics in kpis.items():
            if isinstance(metrics, dict):
                constraint_parts.append(f"\n- {dimension}:")
                for metric, value in metrics.items():
                    constraint_parts.append(f"  - {metric}: {value}")

    constraint_parts.append("\n**强制要求**: 在撰写本章时，必须:")
    constraint_parts.append("1. 显式说明本章举措如何支撑上述核心使命")
    constraint_parts.append("2. 阐明本章内容与战略支柱的关系")
    constraint_parts.append("3. 确保提出的目标与KPI体系保持一致")
    constraint_parts.append("4. 使用'为支撑...使命'、'为实现...目标'、'落实...战略支柱'等表述")

    return "\n".join(constraint_parts)


def _get_model_writing_instruction(analysis_model: str, phase: str) -> str:
    """
    根据分析模型生成特定的写作指令。

    Args:
        analysis_model: Analysis model name
        phase: Current phase

    Returns:
        Model-specific writing instruction
    """
    if not analysis_model:
        return ""

    base_instruction = f"\n## 分析模型写作要求\n\n**指定模型**: {analysis_model}\n\n"

    if "PEST" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现PEST模型的四个维度
- 使用小标题明确区分: 政策环境 (Political)、经济影响 (Economic)、社会因素 (Social)、技术发展 (Technological)
- 在每个维度下展开分析，确保逻辑清晰
- 重点突出政策与经济维度（如要求所述）"""

    elif "SWOT" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现SWOT分析的逻辑框架
- 主体部分应按照"优势-劣势-机会-威胁"的结构组织
- 可使用小标题如"内部优势"、"存在的不足"、"外部机遇"、"面临挑战"
- 在章节结尾应给出综合性的SWOT总结或矩阵"""

    elif "BCG" in analysis_model or "波士顿" in analysis_model or "现金牛" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现BCG波士顿矩阵的业务分类逻辑
- 按业务类型分类阐述：现金牛业务、明星业务、问题业务、瘦狗业务
- 对于主业，强调其"现金牛"属性：稳定回报、精益化运营
- 使用业务组合的视角进行分析，体现资源配置思路"""

    elif "波特五力" in analysis_model or "五力模型" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现波特五力模型的竞争分析框架
- 从五个维度展开：现有竞争者、潜在进入者、替代品、供应商、买方
- 每个维度使用小标题单独阐述
- 最后综合评估行业竞争态势"""

    elif "平衡计分卡" in analysis_model or "BSC" in analysis_model or "计分卡" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现平衡计分卡的四个维度
- 使用小标题明确区分: 财务维度、客户/民生维度、内部运营维度、学习与成长维度
- 每个维度下提出具体的目标或指标
- 强调四个维度的平衡性和协同性"""

    elif "安索夫" in analysis_model or "Ansoff" in analysis_model or "增长矩阵" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现安索夫矩阵的增长战略组合
- 按四个象限展开：市场渗透、市场开发、产品开发、多元化
- 重点识别"第二增长曲线"机会
- 强调创新业务的新产品/新市场属性"""

    elif "7S" in analysis_model or "麦肯锡" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现麦肯锡7S模型的系统性思维
- 涵盖七个要素：战略、结构、制度、共同价值观、风格、员工、技能
- 强调各要素间的协调性和一致性
- 构建支撑战略目标的完整组织保障体系"""

    elif "ESG" in analysis_model or "社会责任" in analysis_model or "产业链协同" in analysis_model:
        return base_instruction + """**写作要求**: 必须体现ESG社会责任与产业链协同
- 从环境、社会、治理、产业链协同四个维度展开
- 强调国企的社会责任担当
- 突出产业链龙头带动作用"""

    else:
        # 通用模型指令
        return base_instruction + f"""**写作要求**: 体现{analysis_model}的分析逻辑
- 在章节结构和内容中自然融入该分析模型的视角
- 使用对应的专业术语和框架
- 确保逻辑符合该模型的分析范式"""


def _get_fallback_draft(chapter_title: str, key_facts: List[str], insights: List[str]) -> str:
    """
    Generate fallback chapter draft when LLM fails.

    Args:
        chapter_title: Title of the chapter
        key_facts: List of key facts
        insights: List of insights

    Returns:
        Fallback draft text
    """
    logger.warning("Using fallback draft generation")

    # Build simple draft from available facts and insights
    draft_parts = []

    # Opening
    draft_parts.append(f"# {chapter_title}\n")
    draft_parts.append(f"本章节主要分析{chapter_title}的相关情况。\n")

    # Body - facts
    if key_facts:
        draft_parts.append("## 关键事实\n")
        for i, fact in enumerate(key_facts, 1):
            draft_parts.append(f"{i}. {fact}")
        draft_parts.append("")

    # Body - insights
    if insights:
        draft_parts.append("## 主要发现\n")
        for i, insight in enumerate(insights, 1):
            draft_parts.append(f"{i}. {insight}")
        draft_parts.append("")

    # Closing
    draft_parts.append("## 总结\n")
    draft_parts.append("由于技术原因，本章节内容为自动生成的简化版本。")
    draft_parts.append("建议后续进行人工完善和补充。")

    fallback_draft = "\n".join(draft_parts)

    return fallback_draft
