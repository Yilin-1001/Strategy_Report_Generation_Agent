"""
Analyst Node - Extracts key facts and generates insights from retrieved documents

This node analyzes the documents retrieved by the Researcher node and:
- Generates a summary of the retrieved documents (limited to 10 docs)
- Extracts key facts from the documents
- Generates insights based on the analysis
- Merges results into the existing chapter scratchpad
"""

import json
import logging
from typing import Dict, List, Any

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def _validate_analysis(key_facts, insights: list, analysis_model: str) -> Dict:
    """
    校验分析结果质量。

    Args:
        key_facts: 提取的关键事实（可能是 list 或 dict）
        insights: 提取的洞察列表
        analysis_model: 使用的分析模型名称

    Returns:
        Dict with "valid" (bool) and "issues" (list of strings)
    """
    issues = []

    # 检查 key_facts 数量
    if isinstance(key_facts, dict):
        total_facts = sum(len(v) for v in key_facts.values() if isinstance(v, list))
        if total_facts < 3:
            issues.append(f"key_facts 总数不足（{total_facts} < 3）")
        # 检查结构化模型的维度完整性
        if analysis_model:
            if "PEST" in analysis_model:
                expected = {"Political", "Economic", "Social", "Technological"}
                actual = set(key_facts.keys())
                if len(expected & actual) < 3:
                    issues.append(f"PEST 维度不完整: {actual}")
            elif "SWOT" in analysis_model:
                expected = {"Strengths", "Weaknesses", "Opportunities", "Threats"}
                actual = set(key_facts.keys())
                if len(expected & actual) < 3:
                    issues.append(f"SWOT 维度不完整: {actual}")
    elif isinstance(key_facts, list):
        if len(key_facts) < 3:
            issues.append(f"key_facts 数量不足（{len(key_facts)} < 3）")

    # 检查 insights
    if len(insights) < 2:
        issues.append(f"insights 数量不足（{len(insights)} < 2）")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def analyst_node(state: Dict[str, Any], llm_manager) -> Dict[str, Any]:
    """
    使用指定战略分析模型分析检索到的文档，提取关键事实并生成洞察。

    该节点会根据当前章节元数据中的 analysis_model 字段，动态注入对应的战略分析框架
    （如PEST、SWOT、BCG、7S等），强制LLM按照指定模型的结构化方式进行分析。

    战略模型注入示例:
    - PEST模型: 按Political/Economic/Social/Technological分类返回key_facts
    - SWOT模型: 按Strengths/Weaknesses/Opportunities/Threats分类返回
    - BCG矩阵: 按现金牛/明星/问题/瘦狗业务分类
    - 7S模型: 按结构/制度/风格等维度分析

    Args:
        state: Current workflow state containing:
            - chapter_question: The research question
            - chapter_context: Optional context for the question
            - chapter_scratchpad: Dict with intermediate results including:
                - queries: List of search queries used
                - retrieved_docs: List of retrieved documents
            - global_plan: List[Dict] with chapter metadata including analysis_model
            - current_chapter_index: Index of current chapter
        llm_manager: LLMManager instance for analysis

    Returns:
        Dict with updates:
            - chapter_scratchpad: Updated scratchpad with:
                - queries: Preserved original queries
                - retrieved_docs: Preserved original documents
                - document_summary: Summary of documents
                - key_facts: List of extracted key facts (may be structured by model)
                - insights: List of generated insights
                - analysis_model_used: The analysis model applied (for reference)

    Example:
        >>> state = {
        ...     "chapter_question": "分析江西省交通运输行业的内外部环境",
        ...     "global_plan": [{"analysis_model": "SWOT分析", ...}],
        ...     "current_chapter_index": 2,
        ...     "chapter_scratchpad": {"retrieved_docs": [...]}
        ... }
        >>> result = analyst_node(state, llm_manager)
        >>> scratchpad = result["chapter_scratchpad"]
        >>> scratchpad["analysis_model_used"]
        'SWOT分析'
        >>> "Strengths" in scratchpad["key_facts"]  # SWOT-structured facts
        True
    """
    chapter_question = state.get("chapter_question", "")
    chapter_context = state.get("chapter_context", "")
    chapter_scratchpad = state.get("chapter_scratchpad", {})

    # Get current chapter metadata for model injection
    global_plan = state.get("global_plan", [])
    current_chapter_index = state.get("current_chapter_index", 0)

    chapter_metadata = global_plan[current_chapter_index] if current_chapter_index < len(global_plan) else {}
    analysis_model = chapter_metadata.get("analysis_model", "")
    phase = chapter_metadata.get("phase", "")

    logger.info(f"Analyst node analyzing with strategic model: {analysis_model} (phase: {phase})")

    # Get retrieved documents from scratchpad
    retrieved_docs = chapter_scratchpad.get("retrieved_docs", [])
    revision_feedback = chapter_scratchpad.get("revision_feedback")

    logger.info(f"Analyst node analyzing {len(retrieved_docs)} retrieved documents")

    # Step 1: Generate document summary with smart extraction
    document_summary = _generate_document_summary_v2(retrieved_docs, total_budget=30000)

    # Step 2: Extract key facts and insights using LLM with model injection
    try:
        key_facts, insights = _extract_facts_and_insights(
            chapter_question=chapter_question,
            chapter_context=chapter_context,
            document_summary=document_summary,
            analysis_model=analysis_model,
            phase=phase,
            revision_feedback=revision_feedback,
            llm_manager=llm_manager
        )
        logger.info(f"Extracted {len(key_facts)} key facts and {len(insights)} insights using {analysis_model}")

    except Exception as e:
        logger.error(f"Error extracting facts and insights: {e}. Using fallback.")
        key_facts, insights = _get_fallback_analysis(retrieved_docs)

    # Step 2.5: Quality self-evaluation (max 1 retry with lower temperature)
    validation = _validate_analysis(key_facts, insights, analysis_model)
    if not validation["valid"]:
        logger.warning(f"Analysis quality insufficient: {validation['issues']}. Retrying with lower temperature...")
        try:
            retry_facts, retry_insights = _extract_facts_and_insights(
                chapter_question=chapter_question,
                chapter_context=chapter_context,
                document_summary=document_summary,
                analysis_model=analysis_model,
                phase=phase,
                revision_feedback=revision_feedback,
                llm_manager=llm_manager
            )
            retry_validation = _validate_analysis(retry_facts, retry_insights, analysis_model)
            if retry_validation["valid"]:
                logger.info("Retry analysis passed validation")
                key_facts, insights = retry_facts, retry_insights
            else:
                logger.warning(f"Retry still insufficient: {retry_validation['issues']}. Using as-is.")
        except Exception as e:
            logger.warning(f"Analysis retry failed: {e}. Using initial results.")

    # Step 3: Merge results into scratchpad (preserve existing data)
    chapter_scratchpad["document_summary"] = document_summary
    chapter_scratchpad["key_facts"] = key_facts
    chapter_scratchpad["insights"] = insights
    chapter_scratchpad["analysis_model_used"] = analysis_model  # Track which model was used

    return {
        "chapter_scratchpad": chapter_scratchpad
    }


def _smart_extract(text: str, budget: int = 900) -> str:
    """
    智能提取文档文本，优先保留首尾内容。

    策略: 分配 60% 预算给开头，40% 给结尾，中间标记省略。
    如果文本长度在预算内，直接返回原文。

    Args:
        text: 原始文档文本
        budget: 总字符预算

    Returns:
        提取后的文本
    """
    if len(text) <= budget:
        return text

    head_budget = int(budget * 0.6)
    tail_budget = budget - head_budget - 20  # 20 chars for ellipsis marker

    head = text[:head_budget]
    tail = text[-tail_budget:] if tail_budget > 0 else ""

    return f"{head}\n\n...[中间内容省略]...\n\n{tail}"


def _generate_document_summary_v2(documents: List[Dict[str, Any]], total_budget: int = 18000) -> str:
    """
    生成文档摘要（v2: 首尾提取 + 动态预算分配）。

    改进:
    - 不再硬限制为10个文档，根据总预算动态决定处理数量
    - 每个文档使用首尾提取而非纯截断
    - 元数据（来源、页码）占用独立预算，不影响文本预算

    Args:
        documents: 文档列表
        total_budget: 总字符预算（默认18000）

    Returns:
        格式化的文档摘要字符串
    """
    if not documents:
        return "No documents retrieved for analysis."

    # 计算每个文档可分配的文本预算
    # 预留元数据和格式化开销（每文档约80字符）
    metadata_overhead = 80
    per_doc_budget = min(900, (total_budget // max(len(documents), 1)) - metadata_overhead)

    # 如果文档太多导致单文档预算太小，限制文档数量
    if per_doc_budget < 400:
        # 保证每个文档至少400字符
        max_docs = total_budget // (400 + metadata_overhead)
        documents = documents[:max(1, max_docs)]
        per_doc_budget = (total_budget // len(documents)) - metadata_overhead

    summary_parts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        # 提取元数据
        source = metadata.get("source", "")
        page = metadata.get("page_number", "N/A")
        title = metadata.get("title", "")
        doc_type = metadata.get("doc_type", "")

        # 确定显示名称（fallback chain）
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

        # 使用首尾提取替代硬截断
        text_preview = _smart_extract(text, budget=per_doc_budget)

        if page and str(page) != "N/A":
            doc_entry = f"Document {i} [来源: {file_name}, 第{page}页]:\n{text_preview}"
        else:
            doc_entry = f"Document {i} [来源: {file_name}]:\n{text_preview}"
        summary_parts.append(doc_entry)

    return "\n\n".join(summary_parts)


def _generate_document_summary(documents: List[Dict[str, Any]], limit: int = 10) -> str:
    """
    Generate a summary of retrieved documents for LLM analysis.

    Args:
        documents: List of document dictionaries with 'text' and 'metadata' fields
        limit: Maximum number of documents to include in summary

    Returns:
        Formatted string summary of documents
    """
    # Limit documents to avoid overwhelming the LLM
    docs_to_summarize = documents[:limit]

    if not docs_to_summarize:
        return "No documents retrieved for analysis."

    summary_parts = []
    for i, doc in enumerate(docs_to_summarize, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        # Extract metadata fields for citation
        source = metadata.get("source", "")
        page = metadata.get("page_number", "N/A")
        title = metadata.get("title", "")
        doc_type = metadata.get("doc_type", "")

        # Determine display name with fallback chain: source → title → doc_type → index
        file_name = None

        # Try 1: Use source path to extract filename
        if source and str(source).strip():
            source_clean = str(source).strip()
            if '/' in source_clean:
                file_name = source_clean.split('/')[-1]
            elif '\\' in source_clean:
                file_name = source_clean.split('\\')[-1]
            else:
                file_name = source_clean
            # Remove common extensions for cleaner display
            for ext in ('.txt', '.pdf', '.docx', '.doc'):
                if file_name.endswith(ext):
                    file_name = file_name[:-len(ext)]
                    break

        # Try 2: Use title directly if source is empty
        if not file_name and title and str(title).strip():
            file_name = str(title).strip()[:80]

        # Try 3: Use doc_type as last resort before index
        if not file_name and doc_type and str(doc_type).strip():
            file_name = f"{str(doc_type).strip()}文档"

        # Final fallback: Use document index
        if not file_name:
            file_name = f"来源文档_{i}"

        # Truncate text if too long (keep first 500 chars)
        text_preview = text[:500] if len(text) > 500 else text

        # Use filename directly in citation format, keep "Document i" as reference label
        if page and str(page) != "N/A":
            doc_entry = f"Document {i} [来源: {file_name}, 第{page}页]:\n{text_preview}"
        else:
            doc_entry = f"Document {i} [来源: {file_name}]:\n{text_preview}"
        summary_parts.append(doc_entry)

    return "\n\n".join(summary_parts)


def _extract_facts_and_insights(
    chapter_question: str,
    chapter_context: str,
    document_summary: str,
    analysis_model: str = "",
    phase: str = "",
    revision_feedback: Dict = None,
    llm_manager = None
) -> tuple[List[str], List[str]]:
    """
    使用LLM从文档中提取关键事实并生成洞察，支持战略分析模型注入。

    Args:
        chapter_question: The research question
        chapter_context: Optional context
        document_summary: Summary of retrieved documents
        analysis_model: Strategic analysis model to apply (e.g., "PEST", "SWOT", "BCG")
        phase: Current phase (diagnosis/initiatives)
        llm_manager: LLMManager instance

    Returns:
        Tuple of (key_facts, insights) as lists of strings

    Raises:
        Exception: If LLM call fails or response cannot be parsed
    """
    # Generate prompt for analysis with model-specific instructions
    prompt = _generate_analysis_prompt(
        chapter_question=chapter_question,
        chapter_context=chapter_context,
        document_summary=document_summary,
        analysis_model=analysis_model,
        phase=phase,
        revision_feedback=revision_feedback
    )

    # Invoke LLM with moderate temperature for balanced analysis
    response = llm_manager.invoke(prompt, temperature=0.5)

    # Parse JSON response
    key_facts, insights = _parse_analysis_response(response, analysis_model)

    return key_facts, insights


def _generate_analysis_prompt(
    chapter_question: str,
    chapter_context: str,
    document_summary: str,
    analysis_model: str = "",
    phase: str = "",
    revision_feedback: Dict = None
) -> str:
    """
    Generate prompt for LLM analysis with optional strategic model injection.

    Args:
        chapter_question: The research question
        chapter_context: Optional context
        document_summary: Summary of retrieved documents
        analysis_model: Strategic analysis model to apply
        phase: Current phase (diagnosis/initiatives)
        revision_feedback: Revision feedback from reviewer (for revise:logic)

    Returns:
        Formatted prompt string with model-specific instructions
    """
    context_section = f"\nContext: {chapter_context}" if chapter_context else ""

    # Generate model-specific instruction
    model_instruction = _get_model_instruction(analysis_model)

    # Phase-specific context
    phase_context = ""
    if phase == "diagnosis":
        phase_context = "\n**当前阶段**: 诊断阶段 - 侧重现状分析、问题识别和环境扫描"
    elif phase == "initiatives":
        phase_context = "\n**当前阶段**: 推演阶段 - 侧重战略举措、实施路径和资源配置"

    # Build revision feedback section
    revision_section = ""
    if revision_feedback:
        issues = revision_feedback.get("issues", [])
        comments = revision_feedback.get("comments", "")
        previous_draft = revision_feedback.get("previous_draft_summary", "")
        dimension_scores = revision_feedback.get("dimension_scores", {})
        improvement_hints = revision_feedback.get("improvement_hints", {})
        analyst_hint = improvement_hints.get("analyst", "")

        issues_lines = "\n".join(f"   - {issue}" for issue in issues)
        if comments:
            issues_lines += f"\n   - [用户补充指令] {comments}"
        draft_preview = f"\n上一轮草稿摘录:\n{previous_draft}" if previous_draft else ""

        # 构建维度评分摘要
        dim_summary = ""
        if dimension_scores:
            dim_items = [f"   - {k}: {v}分" for k, v in dimension_scores.items() if v is not None]
            dim_summary = "\n维度评分:\n" + "\n".join(dim_items)

        # 构建针对性改进方向
        hint_section = ""
        if analyst_hint:
            hint_section = f"\n**针对性改进方向**: {analyst_hint}"

        revision_section = f"""
## 上一轮评审反馈（请针对这些问题改进分析）

评审分数: {revision_feedback.get('score', 'N/A')}/100
主要问题:
{issues_lines}{dim_summary}
{draft_preview}{hint_section}

**改进要求**: 请针对上述问题重新分析，避免重复同样的错误。
"""

    return f"""使用指定的战略分析模型分析以下检索到的文档。

研究问题: {chapter_question}{context_section}
**指定分析模型**: {analysis_model}{phase_context}

{model_instruction}

检索到的文档:
{document_summary}
{revision_section}
你的任务:
1. 使用指定的分析模型框架，从文档中提取3-5个关键事实
2. 生成2-4个洞察，提供更深层次的分析或连接不同信息点

关键事实要求:
- 必须有文档直接支持
- 应具体且基于事实（非观点）
- 应简洁（每项1-2句话）
- 必须按照指定的分析模型框架分类组织

洞察要求:
- 应提供超越事实的深度分析
- 可识别模式、趋势或关系
- 可强调影响或意义
- 应体现战略思维

**输出要求**:
- 必须使用中文输出
- 返回有效的JSON对象，结构如下：
{{
    "key_facts": [根据分析模型分类组织的事实列表],
    "insights": ["洞察1", "洞察2"]
}}

现在请进行分析:"""


def _get_model_instruction(analysis_model: str) -> str:
    """
    根据指定的战略分析模型生成对应的指令文本。

    Args:
        analysis_model: 分析模型名称（如"PEST模型"、"SWOT分析"等）

    Returns:
        模型特定的分析指令
    """
    if not analysis_model:
        return "**通用分析**: 请按照常规逻辑提取关键事实和洞察。"

    instruction = ""

    if "PEST" in analysis_model:
        instruction = """
**强制使用PEST模型框架分析**：
你必须从以下维度组织分析结果：

- **Political (政策/政治)**: 提取相关政策、法规、政府举措、政治环境变化
- **Economic (经济)**: 提取经济数据、市场趋势、财务影响、投资环境
- **Social (社会)**: 提取社会因素、人口结构、公众需求、社会态度
- **Technological (技术)**: 提取技术发展、创新应用、数字化转型、技术壁垒

在返回的JSON中，key_facts必须按照P-E-S-T分类返回：
{{
    "key_facts": {{
        "Political": ["政策事实1", "政策事实2"],
        "Economic": ["经济事实1", "经济事实2"],
        "Social": ["社会事实1"],
        "Technological": ["技术事实1", "技术事实2"]
    }},
    "insights": ["洞察1", "洞察2"]
}}

**特别要求**: 侧重P（政策）与E（经济）维度，关注国家战略、省级政策、财政支持等信息。"""

    elif "SWOT" in analysis_model or "SWOT分析" in analysis_model:
        instruction = """
**强制使用SWOT模型框架分析**：
你必须从以下维度组织分析结果：

- **Strengths (优势)**: 内部优势资源、核心能力、竞争壁垒
- **Weaknesses (劣势)**: 内部不足、瓶颈问题、资源短板
- **Opportunities (机会)**: 外部机遇、有利条件、市场空间
- **Threats (威胁)**: 外部挑战、风险因素、竞争压力

在返回的JSON中，必须包含结构化的SWOT矩阵：
{{
    "key_facts": {{
        "Strengths": ["优势1", "优势2"],
        "Weaknesses": ["劣势1", "劣势2"],
        "Opportunities": ["机会1", "机会2"],
        "Threats": ["威胁1", "威胁2"]
    }},
    "insights": ["基于SWOT的洞察1", "基于SWOT的洞察2"]
}}

**特别要求**: 必须在每个维度下至少提取2-3项，确保SWOT矩阵的完整性和平衡性。"""

    elif "波士顿" in analysis_model or "BCG" in analysis_model or "现金牛" in analysis_model:
        instruction = """
**强制使用BCG波士顿矩阵分析**：
将业务/产品/项目按照以下维度分类：

- **现金牛业务 (Cash Cow)**: 高市场份额、低增长 → 侧重精益化运营、稳健回报
- **明星业务 (Star)**: 高市场份额、高增长 → 侧重持续投资、扩大优势
- **问题业务 (Question Mark)**: 低市场份额、高增长 → 侧重战略选择、资源配置
- **瘦狗业务 (Dog)**: 低市场份额、低增长 → 侧重退出或转型

在返回的JSON中：
{{
    "key_facts": {{
        "现金牛业务": ["相关事实1", "相关事实2"],
        "明星业务": ["相关事实1"],
        "问题业务": ["相关事实1"],
        "瘦狗业务": ["相关事实1"]
    }},
    "insights": ["业务组合洞察1", "业务组合洞察2"]
}}

**特别要求**: 识别主业作为现金牛业务，强调其稳定性和对整体业务的支撑作用。"""

    elif "波特五力" in analysis_model or "五力模型" in analysis_model:
        instruction = """
**强制使用波特五力模型分析**：
从以下五个竞争力量维度分析：

- **现有竞争者竞争强度**: 市场竞争格局、主要竞争对手、竞争策略
- **潜在进入者威胁**: 行业壁垒、准入门槛、新进入者可能性
- **替代品威胁**: 替代方案、替代技术、替代服务
- **供应商议价能力**: 供应商集中度、依赖程度、成本压力
- **买方议价能力**: 客户集中度、价格敏感度、需求变化

在返回的JSON中：
{{
    "key_facts": {{
        "现有竞争者": ["事实1"],
        "潜在进入者": ["事实1"],
        "替代品": ["事实1"],
        "供应商": ["事实1"],
        "买方": ["事实1"]
    }},
    "insights": ["行业竞争态势洞察1", "洞察2"]
}}"""

    elif "平衡计分卡" in analysis_model or "BSC" in analysis_model or "计分卡" in analysis_model:
        instruction = """
**强制使用平衡计分卡(BSC)模型分析**：
从以下四个维度设定目标和分析：

- **财务维度**: 营收增长、成本控制、资产效率、盈利能力
- **客户/民生维度**: 客户满意度、公共服务质量、社会评价
- **内部运营维度**: 运营效率、项目管理、服务质量、安全保障
- **学习与成长维度**: 人才队伍、创新能力、组织文化、信息化水平

在返回的JSON中：
{{
    "key_facts": {{
        "财务维度": ["事实1", "事实2"],
        "客户/民生维度": ["事实1"],
        "内部运营维度": ["事实1", "事实2"],
        "学习与成长维度": ["事实1"]
    }},
    "insights": ["战略目标洞察1", "洞察2"]
}}

**特别要求**: 每个维度应包含可量化的目标或指标。"""

    elif "安索夫" in analysis_model or "Ansoff" in analysis_model or "增长矩阵" in analysis_model:
        instruction = """
**强制使用安索夫矩阵分析**：
从以下战略组合维度分析：

- **市场渗透 (现有市场×现有产品)**: 提升市场份额、深化客户关系
- **市场开发 (新市场×现有产品)**: 拓展地域、进入新细分市场
- **产品开发 (现有市场×���产品)**: 创新业务、新服务模式
- **多元化 (新市场×新产品)**: 全新业务领域、跨界融合

在返回的JSON中：
{{
    "key_facts": {{
        "市场渗透": ["事实1"],
        "市场开发": ["事实1"],
        "产品开发": ["事实1", "事实2"],
        "多元化": ["事实1"]
    }},
    "insights": ["增长战略洞察1", "洞察2"]
}}

**特别要求**: 侧重识别第二增长曲线，强调创新业务的拓展潜力。"""

    elif "7S" in analysis_model or "麦肯锡7S" in analysis_model or "麦肯锡" in analysis_model:
        instruction = """
**强制使用麦肯锡7S模型分析**：
从以下七个相互关联的要素分析：

- **Strategy (战略)**: 总体战略规划、竞争策略、发展方向
- **Structure (结构)**: 组织架构、权责划分、管理体系
- **Systems (制度)**: 管理制度、流程规范、信息系统
- **Shared Values (共同价值观)**: 核心理念、企业文化、使命愿景
- **Style (风格)**: 领导风格、管理方式、决策模式
- **Staff (员工)**: 人才队伍、能力素质、激励机制
- **Skills (技能)**: 核心能力、专业特长、竞争优势

在返回的JSON中：
{{
    "key_facts": {{
        "Strategy": ["事实1"],
        "Structure": ["事实1"],
        "Systems": ["事实1"],
        "Shared Values": ["事实1"],
        "Style": ["事实1"],
        "Staff": ["事实1"],
        "Skills": ["事实1"]
    }},
    "insights": ["组织保障洞察1", "洞察2"]
}}

**特别要求**: 强调各要素间的协调性和一致性，构建支撑战略的组织保障体系。"""

    elif "ESG" in analysis_model or "社会责任" in analysis_model or "产业链协同" in analysis_model:
        instruction = """
**强制使用ESG社会责任与产业链协同模型分析**：
从以下维度分析：

- **Environment (环境)**: 绿色发展、低碳转型、环境保护
- **Social (社会)**: 员工福祉、社区关系、公共安全、社会责任
- **Governance (治理)**: 公司治理、合规管理、风险控制、信息披露
- **产业链协同**: 上下游合作、产业生态、协同效应

在返回的JSON中：
{{
    "key_facts": {{
        "Environment": ["事实1"],
        "Social": ["事实1", "事实2"],
        "Governance": ["事实1"],
        "产业链协同": ["事实1", "事实2"]
    }},
    "insights": ["ESG与协同洞察1", "洞察2"]
}}

**特别要求**: 强调国企的社会责任担当和产业链龙头带动作用。"""

    else:
        # 通用分析（无特定模型）
        instruction = f"""
**使用常规分析方法**：
分析模型: {analysis_model}

请按照以下维度提取关键事实：
- 核心主题相关的事实
- 数据和统计信息
- 政策或法规信息
- 问题或挑战
- 机遇或优势

在返回的JSON中：
{{
    "key_facts": ["事实1", "事实2", "事实3", "事实4", "事实5"],
    "insights": ["洞察1", "洞察2"]
}}"""

    return instruction


def _parse_analysis_response(response: str, analysis_model: str = "") -> tuple[List[str], List[str]]:
    """
    解析LLM响应以提取key_facts和insights，支持结构化key_facts（如SWOT/PEST模型）。

    Args:
        response: Raw LLM response string
        analysis_model: The analysis model used (for handling structured responses)

    Returns:
        Tuple of (key_facts, insights) as lists of strings
        Note: If key_facts is structured (dict), it will be preserved as dict for downstream use

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    # Clean up response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    # Parse JSON
    try:
        result = json.loads(response)

        # Extract key_facts and insights
        key_facts = result.get("key_facts", [])
        insights = result.get("insights", [])

        # Handle structured key_facts (dict from models like PEST/SWOT/BCG)
        if isinstance(key_facts, dict):
            # Preserve the structured format for downstream nodes
            logger.info(f"Detected structured key_facts with keys: {list(key_facts.keys())}")
        elif isinstance(key_facts, list):
            # Convert list items to strings
            key_facts = [str(fact) for fact in key_facts]
        else:
            # Single item, convert to list
            key_facts = [str(key_facts)]

        # Ensure insights is a list of strings
        if not isinstance(insights, list):
            insights = [str(insights)]
        insights = [str(insight) for insight in insights]

        return key_facts, insights

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response content: {response[:500]}")
        raise ValueError(f"Invalid JSON response: {e}")


def _get_fallback_analysis(documents: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    """
    Generate fallback analysis when LLM fails.

    Args:
        documents: List of retrieved documents

    Returns:
        Tuple of (key_facts, insights) as lists of strings
    """
    logger.warning("Using fallback analysis")

    # Extract basic facts from documents
    key_facts = []
    insights = []

    if documents:
        # Count documents
        key_facts.append(f"Retrieved {len(documents)} relevant documents")

        # Extract source information
        sources = set()
        for doc in documents[:5]:  # Check first 5 docs
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "")

            # Handle empty/None source with improved fallback
            if source and str(source).strip():
                # Extract filename from path
                source_clean = str(source).strip()
                if '/' in source_clean:
                    source_clean = source_clean.split('/')[-1]
                elif '\\' in source_clean:
                    source_clean = source_clean.split('\\')[-1]
                sources.add(source_clean)
            else:
                # Try title as fallback
                title = metadata.get("title", "")
                if title and str(title).strip():
                    sources.add(f"文档: {str(title).strip()[:30]}")
                else:
                    # Use doc_type as fallback
                    doc_type = metadata.get("doc_type", "")
                    if doc_type:
                        sources.add(f"{doc_type}文档")
                    else:
                        sources.add("未命名文档")

        if sources:
            key_facts.append(f"Information来源于: {', '.join(list(sources)[:3])}")

        # Generate basic insights
        if len(documents) > 5:
            insights.append(f"Found substantial relevant documentation ({len(documents)} documents)")
        else:
            insights.append(f"Limited documentation available ({len(documents)} documents)")

        insights.append("Detailed analysis unavailable due to processing error")
    else:
        key_facts.append("No documents were retrieved for analysis")
        insights.append("Unable to generate insights without source documents")

    return key_facts, insights
