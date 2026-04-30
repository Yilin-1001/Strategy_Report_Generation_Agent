"""
Shared Prompts Module

All prompt-generating functions extracted from rag_project agent nodes.
These prompts are used identically across all 4 ablation experiment groups.

Sources:
- researcher.py: query generation prompts
- analyst.py: analysis prompts with strategic model injection
- writer.py: writing prompts with model-specific instructions
- strategist.py: blueprint generation prompts
- prep_chapter.py: context compression prompts
"""

from typing import Dict, List, Any, Optional


# ============================================================================
# RESEARCHER NODE PROMPTS (from researcher.py)
# ============================================================================

def generate_query_prompt(
    chapter_question: str,
    chapter_context: str,
    revision_hint: str = ""
) -> str:
    """
    Generate prompt for multi-query generation.
    Source: researcher.py:135-147

    Args:
        chapter_question: The main research question
        chapter_context: Optional context for the question
        revision_hint: Optional hint from reviewer for targeted retrieval

    Returns:
        Formatted prompt string for LLM
    """
    revision_section = ""
    if revision_hint:
        revision_section = f"""
IMPORTANT - Previous retrieval was insufficient:
{revision_hint}

Make sure your queries specifically address these gaps.
"""

    return f"""Given the following research question, generate 5-7 diverse and specific search queries that would help retrieve relevant documents.

Research Question: {chapter_question}

Context: {chapter_context if chapter_context else "No additional context provided"}
{revision_section}
Generate 5-7 search queries (one per line) that:
1. Rephrase the question in different ways
2. Include related keywords and concepts
3. Cover different aspects of the topic
4. Are specific enough to retrieve relevant documents

Queries:"""


def generate_retrieval_sufficiency_prompt(
    chapter_question: str,
    doc_preview: str,
    doc_count: int
) -> str:
    """
    Generate prompt to evaluate retrieval sufficiency.
    Source: researcher.py:264-272

    Args:
        chapter_question: The research question
        doc_preview: Preview of retrieved documents
        doc_count: Number of documents retrieved

    Returns:
        Formatted prompt string
    """
    return f"""评估以下检索结果是否能充分回答研究问题。

研究问题: {chapter_question}

检索到 {doc_count} 个文档:
{doc_preview}

请判断: 这些文档的信息是否足以支撑对该问题的深入分析？
只回答 YES 或 NO。"""


def generate_supplementary_query_prompt(
    chapter_question: str,
    existing_keywords: str,
    doc_count: int
) -> str:
    """
    Generate prompt for supplementary queries when initial retrieval insufficient.
    Source: researcher.py:298-304

    Args:
        chapter_question: The research question
        existing_keywords: Keywords from existing queries
        doc_count: Number of documents already retrieved

    Returns:
        Formatted prompt string
    """
    return f"""基于初始检索结果，生成 2 个补充搜索查询。

研究问题: {chapter_question}
已有查询: {existing_keywords}
已检索到 {doc_count} 个文档，但信息不够充分。

请生成 2 个不同角度的补充查询（每行一个），聚焦尚未覆盖的方面:"""


# ============================================================================
# ANALYST NODE PROMPTS (from analyst.py)
# ============================================================================

def get_model_instruction(analysis_model: str) -> str:
    """
    Generate strategic model-specific analysis instruction.
    Source: analyst.py:512-736

    Args:
        analysis_model: Analysis model name (e.g., "PEST模型", "SWOT分析")

    Returns:
        Model-specific instruction text
    """
    if not analysis_model:
        return "**通用分析**: 请按照常规逻辑提取关键事实和洞察。"

    if "PEST" in analysis_model:
        return """
**强制使用PEST模型框架分析**：
你必须从以下维度组织分析结果：

- **Political (政策/政治)**: 提取相关政策、法规、政府举措、政治环境变化
- **Economic (经济)**: 提取经济数据、市场趋势、财务影响、投资环境
- **Social (社会)**: 提取社会因素、人口结构、公众需求、社会态度
- **Technological (技术)**: 提取技术发展、创新应用、数字化转型、技术壁垒

在返回的JSON中，key_facts必须按照P-E-S-T分类返回：
{
    "key_facts": {
        "Political": ["政策事实1", "政策事实2"],
        "Economic": ["经济事实1", "经济事实2"],
        "Social": ["社会事实1"],
        "Technological": ["技术事实1", "技术事实2"]
    },
    "insights": ["洞察1", "洞察2"]
}

**特别要求**: 侧重P（政策）与E（经济）维度，关注国家战略、省级政策、财政支持等信息。"""

    elif "SWOT" in analysis_model or "SWOT分析" in analysis_model:
        return """
**强制使用SWOT模型框架分析**：
你必须从以下维度组织分析结果：

- **Strengths (优势)**: 内部优势资源、核心能力、竞争壁垒
- **Weaknesses (劣势)**: 内部不足、瓶颈问题、资源短板
- **Opportunities (机会)**: 外部机遇、有利条件、市场空间
- **Threats (威胁)**: 外部挑战、风险因素、竞争压力

在返回的JSON中，必须包含结构化的SWOT矩阵：
{
    "key_facts": {
        "Strengths": ["优势1", "优势2"],
        "Weaknesses": ["劣势1", "劣势2"],
        "Opportunities": ["机会1", "机会2"],
        "Threats": ["威胁1", "威胁2"]
    },
    "insights": ["基于SWOT的洞察1", "基于SWOT的洞察2"]
}

**特别要求**: 必须在每个维度下至少提取2-3项，确保SWOT矩阵的完整性和平衡性。"""

    elif "波士顿" in analysis_model or "BCG" in analysis_model or "现金牛" in analysis_model:
        return """
**强制使用BCG波士顿矩阵分析**：
将业务/产品/项目按照以下维度分类：

- **现金牛业务 (Cash Cow)**: 高市场份额、低增长 → 侧重精益化运营、稳健回报
- **明星业务 (Star)**: 高市场份额、高增长 → 侧重持续投资、扩大优势
- **问题业务 (Question Mark)**: 低市场份额、高增长 → 侧重战略选择、资源配置
- **瘦狗业务 (Dog)**: 低市场份额、低增长 → 侧重退出或转型

在返回的JSON中：
{
    "key_facts": {
        "现金牛业务": ["相关事实1", "相关事实2"],
        "明星业务": ["相关事实1"],
        "问题业务": ["相关事实1"],
        "瘦狗业务": ["相关事实1"]
    },
    "insights": ["业务组合洞察1", "业务组合洞察2"]
}

**特别要求**: 识别主业作为现金牛业务，强调其稳定性和对整体业务的支撑作用。"""

    elif "波特五力" in analysis_model or "五力模型" in analysis_model:
        return """
**强制使用波特五力模型分析**：
从以下五个竞争力量维度分析：

- **现有竞争者竞争强度**: 市场竞争格局、主要竞争对手、竞争策略
- **潜在进入者威胁**: 行业壁垒、准入门槛、新进入者可能性
- **替代品威胁**: 替代方案、替代技术、替代服务
- **供应商议价能力**: 供应商集中度、依赖程度、成本压力
- **买方议价能力**: 客户集中度、价格敏感度、需求变化

在返回的JSON中：
{
    "key_facts": {
        "现有竞争者": ["事实1"],
        "潜在进入者": ["事实1"],
        "替代品": ["事实1"],
        "供应商": ["事实1"],
        "买方": ["事实1"]
    },
    "insights": ["行业竞争态势洞察1", "洞察2"]
}"""

    elif "平衡计分卡" in analysis_model or "BSC" in analysis_model or "计分卡" in analysis_model:
        return """
**强制使用平衡计分卡(BSC)模型分析**：
从以下四个维度设定目标和分析：

- **财务维度**: 营收增长、成本控制、资产效率、盈利能力
- **客户/民生维度**: 客户满意度、公共服务质量、社会评价
- **内部运营维度**: 运营效率、项目管理、服务质量、安全保障
- **学习与成长维度**: 人才队伍、创新能力、组织文化、信息化水平

在返回的JSON中：
{
    "key_facts": {
        "财务维度": ["事实1", "事实2"],
        "客户/民生维度": ["事实1"],
        "内部运营维度": ["事实1", "事实2"],
        "学习与成长维度": ["事实1"]
    },
    "insights": ["战略目标洞察1", "洞察2"]
}

**特别要求**: 每个维度应包含可量化的目标或指标。"""

    elif "安索夫" in analysis_model or "Ansoff" in analysis_model or "增长矩阵" in analysis_model:
        return """
**强制使用安索夫矩阵分析**：
从以下战略组合维度分析：

- **市场渗透 (现有市场×现有产品)**: 提升市场份额、深化客户关系
- **市场开发 (新市场×现有产品)**: 拓展地域、进入新细分市场
- **产品开发 (现有市场×新产品)**: 创新业务、新服务模式
- **多元化 (新市场×新产品)**: 全新业务领域、跨界融合

在返回的JSON中：
{
    "key_facts": {
        "市场渗透": ["事实1"],
        "市场开发": ["事实1"],
        "产品开发": ["事实1", "事实2"],
        "多元化": ["事实1"]
    },
    "insights": ["增长战略洞察1", "洞察2"]
}

**特别要求**: 侧重识别第二增长曲线，强调创新业务的拓展潜力。"""

    elif "7S" in analysis_model or "麦肯锡7S" in analysis_model or "麦肯锡" in analysis_model:
        return """
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
{
    "key_facts": {
        "Strategy": ["事实1"],
        "Structure": ["事实1"],
        "Systems": ["事实1"],
        "Shared Values": ["事实1"],
        "Style": ["事实1"],
        "Staff": ["事实1"],
        "Skills": ["事实1"]
    },
    "insights": ["组织保障洞察1", "洞察2"]
}

**特别要求**: 强调各要素间的协调性和一致性，构建支撑战略的组织保障体系。"""

    elif "ESG" in analysis_model or "社会责任" in analysis_model or "产业链协同" in analysis_model:
        return """
**强制使用ESG社会责任与产业链协同模型分析**：
从以下维度分析：

- **Environment (环境)**: 绿色发展、低碳转型、环境保护
- **Social (社会)**: 呡工福祉、社区关系、公共安全、社会责任
- **Governance (治理)**: 公司治理、合规管理、风险控制、信息披露
- **产业链协同**: 上下游合作、产业生态、协同效应

在返回的JSON中：
{
    "key_facts": {
        "Environment": ["事实1"],
        "Social": ["事实1", "事实2"],
        "Governance": ["事实1"],
        "产业链协同": ["事实1", "事实2"]
    },
    "insights": ["ESG与协同洞察1", "洞察2"]
}

**特别要求**: 强调国企的社会责任担当和产业链龙头带动作用。"""

    else:
        # 通用分析（无特定模型）
        return f"""
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


def generate_analysis_prompt(
    chapter_question: str,
    chapter_context: str,
    document_summary: str,
    analysis_model: str,
    phase: str,
    revision_feedback: Optional[Dict] = None
) -> str:
    """
    Generate prompt for LLM analysis with strategic model injection.
    Source: analyst.py:404-509

    Args:
        chapter_question: The research question
        chapter_context: Optional context
        document_summary: Summary of retrieved documents
        analysis_model: Strategic analysis model to apply
        phase: Current phase (diagnosis/initiatives)
        revision_feedback: Revision feedback from reviewer

    Returns:
        Formatted prompt string
    """
    context_section = f"\nContext: {chapter_context}" if chapter_context else ""

    # Generate model-specific instruction
    model_instruction = get_model_instruction(analysis_model)

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

        dim_summary = ""
        if dimension_scores:
            dim_items = [f"   - {k}: {v}分" for k, v in dimension_scores.items() if v is not None]
            dim_summary = "\n维度评分:\n" + "\n".join(dim_items)

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


# ============================================================================
# WRITER NODE PROMPTS (from writer.py)
# ============================================================================

def get_model_writing_instruction(analysis_model: str, phase: str) -> str:
    """
    Generate model-specific writing instruction for the writer.
    Source: writer.py:673-742

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
        return base_instruction + f"""**写作要求**: 体现{analysis_model}的分析逻辑
- 在章节结构和内容中自然融入该分析模型的视角
- 使用对应的专业术语和框架
- 确保逻辑符合该模型的分析范式"""


def build_facts_section(key_facts: Any, analysis_model: str) -> str:
    """
    Build the key facts section for writing prompt.
    Source: writer.py:597-625

    Args:
        key_facts: Key facts (can be list or dict from analysis)
        analysis_model: Analysis model name

    Returns:
        Formatted facts section string
    """
    if not key_facts:
        return "\n关键事实: 无"

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
    return "\n关键事实 (Key Facts):\n" + "\n".join(f"- {fact}" for fact in key_facts)


def build_blueprint_constraint(strategic_blueprint: Optional[Dict]) -> str:
    """
    Build strategic blueprint constraint for initiatives phase.
    Source: writer.py:628-670

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


def generate_writing_prompt(
    chapter_title: str,
    chapter_question: str,
    chapter_context: str,
    key_facts: Any,
    insights: List[str],
    context_summary: str,
    document_summary: str,
    filename_mapping: Dict[str, str],
    analysis_model: str,
    phase: str,
    strategic_blueprint: Optional[Dict] = None,
    revision_feedback: Optional[Dict] = None
) -> str:
    """
    Generate prompt for chapter writing with strategic constraints.
    Source: writer.py:411-594

    Args:
        chapter_title: Title of the chapter
        chapter_question: The research question
        chapter_context: Optional context
        key_facts: Key facts from analysis
        insights: Insights from analysis
        context_summary: Rolling context summary (memory)
        document_summary: Summary of retrieved documents
        filename_mapping: Mapping of Document X to actual filenames
        analysis_model: Strategic analysis model used
        phase: Current phase (diagnosis/initiatives)
        strategic_blueprint: Strategic blueprint (for initiatives phase)
        revision_feedback: Revision feedback from reviewer

    Returns:
        Formatted prompt string
    """
    # Build facts section
    facts_section = build_facts_section(key_facts, analysis_model)

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
        context_parts.append(f"参考文档摘要:\n{document_summary}")

    context_section = "\n".join(context_parts) if context_parts else "无额外背景信息"

    # Build filename reference section
    filename_ref = ""
    if filename_mapping:
        filename_ref = "\n可用来源文件名列表 (引用时直接使用文件名):\n"
        for doc_ref, filename in filename_mapping.items():
            filename_ref += f"- {filename}\n"
    else:
        filename_ref = "\n注意: 无可用文件名映射，请在引用时标注[来源: 相关文档名]\n"

    # Build strategic blueprint constraint
    blueprint_constraint = build_blueprint_constraint(strategic_blueprint)

    # Build model-specific writing instruction
    model_instruction = get_model_writing_instruction(analysis_model, phase)

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

        dim_summary = ""
        if dimension_scores:
            dim_items = [f"   - {k}: {v}分" for k, v in dimension_scores.items() if v is not None]
            dim_summary = "\n维度评分:\n" + "\n".join(dim_items)

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

    # Build cross-chapter consistency instruction
    consistency_instruction = ""
    if context_summary:
        consistency_instruction = """
7. **跨章节一致性**:
   - 如果前序章节已提及具体数字（如投资额、增长率），必须使用相同数字
   - 不要重复前序章节已有的内容，而是在其基础上深化
   - 如果发现前序结论与当前分析矛盾，以当前分析为准但需说明变化原因
"""

    # Hard limit on word count
    hard_limit = """
## ⚠️ 硬性字数上限（必须严格遵守）

- **绝对上限**: 1800个中文字符，超过即为不合格输出
- **目标范围**: 1000-1800个中文字符
- **计数方式**: 统计正文中的中文字符数（不含标题、标记符号）
- **超标后果**: 超过1800字的输出将被系统直接截断，导致内容不完整
- **控制技巧**: 宁可精简到1200字，也不要超过1800字；每个小节控制在300-500字即可

再次强调：你的输出不得超过1800个中文字符！
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


# ============================================================================
# STRATEGIST NODE PROMPTS (from strategist.py)
# ============================================================================

def generate_swot_extraction_prompt(chapter_text: str) -> str:
    """
    Generate prompt to extract structured SWOT from chapter text.
    Source: strategist.py:189-213

    Args:
        chapter_text: Chapter text containing SWOT analysis

    Returns:
        Formatted prompt string
    """
    return f"""你是一位战略分析专家。请从以下章节文本中提取结构化的SWOT分析。

章节文本:
{chapter_text}

任务要求:
1. 仔细阅读文本，识别并提取SWOT四个维度的内容
2. 每个维度至少提取3-5项
3. 确保提取的内容是原文中明确提到的优势、劣势、机会、威胁
4. 返回JSON格式

返回JSON结构:
{{
    "Strengths": ["优势1", "优势2", "优势3", ...],
    "Weaknesses": ["劣势1", "劣势2", "劣势3", ...],
    "Opportunities": ["机会1", "机会2", "机会3", ...],
    "Threats": ["威胁1", "威胁2", "威胁3", ...]
}}

输出要求:
- 必须使用中文
- 每项应该简洁明确（1-2句话）
- 只返回JSON对象，不要包含其他文本

现在请提取SWOT分析:"""


def generate_blueprint_prompt(swot_data: Dict, user_input: str) -> str:
    """
    Generate prompt for strategic blueprint creation.
    Source: strategist.py:315-400

    Args:
        swot_data: Structured SWOT analysis
        user_input: Original user request

    Returns:
        Formatted prompt string
    """
    return f"""你是一位资深的国企战略规划专家。基于前期的诊断分析，现在需要制定省属国企的战略蓝图。

## 已识别的SWOT分析

**优势 (Strengths)**:
{chr(10).join(f"- {s}" for s in swot_data.get('Strengths', []) if s)}

**劣势 (Weaknesses)**:
{chr(10).join(f"- {w}" for w in swot_data.get('Weaknesses', []) if w)}

**机会 (Opportunities)**:
{chr(10).join(f"- {o}" for o in swot_data.get('Opportunities', []) if o)}

**威胁 (Threats)**:
{chr(10).join(f"- {t}" for t in swot_data.get('Threats', []) if t)}

## 任务要求

请使用TOWS矩阵分析法，生成完整的战略蓝图。返回JSON格式：

{{
    "mission": "一句话凝练核心使命（20-30字）",
    "swot_analysis": {{
        "strengths": ["优势1", "优势2", ...],
        "weaknesses": ["劣势1", "劣势2", ...],
        "opportunities": ["机会1", "机会2", ...],
        "threats": ["威胁1", "威胁2", ...]
    }},
    "tows_strategies": {{
        "SO": ["利用优势抓住机会的策略1（具体可执行）", "利用优势抓住机会的策略2"],
        "WO": ["弥补劣势抓住机会的策略1（具体可执行）"],
        "ST": ["利用优势应对威胁的策略1（具体可执行）"],
        "WT": ["减少劣势规避威胁的策略1（具体可执行）"]
    }},
    "strategic_pillars": [
        "战略支柱1：名称与描述（如：主业升级：夯实交通投资建设主阵地）",
        "战略支柱2：名称与描述（如：创新驱动：培育智慧绿色交通新动能）",
        "战略支柱3：名称与描述（如：产业协同：构建交旅融合新生态）",
        "战略支柱4：名称与描述（如：治理提升：完善现代企业制度）"
    ],
    "kpis": {{
        "财务维度": {{"营收增长率": "年增长8%", "资产负债率": "控制在65%以内"}},
        "客户/民生维度": {{"公众满意度": "提升至90分", "服务质量投诉率": "下降30%"}},
        "运营维度": {{"项目按期完工率": "达到95%", "安全事故率": "零重大事故"}},
        "学习成长维度": {{"员工培训覆盖率": "100%", "创新项目数量": "每年5项以上"}}
    }}
}}

## 输出要求

1. **所有内容必须使用中文**
2. **mission（核心使命）**:
   - 必须体现国企"服务国家战略、承担社会责任、推动高质量发展"的定位
   - 20-30字，高度凝练
   - 示例："服务交通强省战略，打造一流综合交通投资运营集团"
3. **TOWS策略**:
   - 必须具体可执行（避免空洞表述）
   - 每类策略至少2-3项
   - 体现内外部匹配的智慧
4. **战略支柱**:
   - 建议3-5个
   - 覆盖业务、创新、协同、治理等维度
   - 每个支柱应有明确的名称和简短描述
5. **KPIs**:
   - 必须符合SMART原则（具体、可衡量、可达成、相关、有时限）
   - 按平衡计分卡四个维度组织
   - 每个维度至少2-3个指标
   - 指标应包含目标值
6. **只返回JSON对象，不要包含任何其他文本**

## 参考背景

用户请求: {user_input}

现在生成战略蓝图:"""


def generate_diagnosis_compression_prompt(
    user_input: str,
    chapters_text: str
) -> str:
    """
    Generate prompt to compress diagnosis phase chapters.
    Source: strategist.py:52-73

    Args:
        user_input: Original user request
        chapters_text: Concatenated text of diagnosis chapters

    Returns:
        Formatted prompt string
    """
    return f"""请将以下诊断阶段的三章内容压缩为一份结构化综合摘要。

主题: {user_input}

要求:
1. 保留所有关键数据（投资额、增长率、政策指标等具体数字）
2. 保留所有政策名称、战略定位、核心结论
3. 重点保留SWOT分析相关内容（优势、劣势、机会、威胁）
4. 按以下结构输出:
   - 宏观环境要点（政策、经济）
   - 区域战略定位
   - 行业竞争态势
   - 内部优势与劣势
   - 外部机遇与威胁
   - 关键数据汇总（列出所有具体数字）
5. 2000字以内
6. 必须使用中文

诊断阶段内容:
{chapters_text}

请生成综合摘要:"""


# ============================================================================
# PREP CHAPTER NODE PROMPTS (from prep_chapter.py)
# ============================================================================

def generate_context_compression_prompt(
    existing_summary: str,
    new_knowledge: str
) -> str:
    """
    Generate prompt for rolling context compression.
    Source: prep_chapter.py:69-85

    Args:
        existing_summary: Existing context summary
        new_knowledge: New knowledge to compress

    Returns:
        Formatted prompt string
    """
    return f"""请将以下新旧知识合并为一份紧凑的滚动摘要。

要求:
1. 保留所有具体数字（金额、比例、增长率等）
2. 保留政策名称和战略定位
3. 保留关键结论和核心洞察
4. 删除冗余和重复
5. 控制在500字以内
6. 使用中文

已有摘要:
{existing_summary if existing_summary else '（无）'}

新增知识:
{new_knowledge}

请输出合并后的摘要:"""