# Agent 核心代码文档

> 本文档摘录 Agent 后端实现中最能体现系统设计的代码段落，分为两大部分：
> 1. **整体工作流与多层记忆架构的主管线** — `graph.py`、`state.py`、路由逻辑
> 2. **不同章节任务的提示词原始代码段落** — 各节点的 Prompt 模板与分析模型注入

---

## 第一部分：整体工作流与多层记忆架构

### 1.1 LangGraph 工作流构建（`graph.py`）

两阶段战略推演架构：**诊断阶段（第1-3章）→ 战略蓝图生成 → 推演阶段（第4-8章）**

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def create_report_graph():
    workflow = StateGraph(GraphState)

    # 初始化各节点的 LLM 管理器
    coordinator_llm = LLMManager("coordinator")
    researcher_llm = LLMManager("researcher")
    analyst_llm = LLMManager("analyst")
    writer_llm = LLMManager("writer")
    strategist_llm = LLMManager("strategist")
    archiver_llm = LLMManager("archiver")
    prep_llm = LLMManager("coordinator")

    retriever = RAGRetriever(agent_config_path="config/agent_config.yaml")

    # 添加所有节点
    workflow.add_node("coordinator", lambda state: coordinator_node(state, coordinator_llm))
    workflow.add_node("prepare_chapter", lambda state: prepare_chapter_node(state, prep_llm))
    workflow.add_node("researcher", lambda state: researcher_node(state, retriever, researcher_llm))
    workflow.add_node("analyst", lambda state: analyst_node(state, analyst_llm))
    workflow.add_node("writer", lambda state: writer_node(state, writer_llm))
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("strategist", lambda state: strategist_node(state, strategist_llm))
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("archiver", lambda state: archiver_node(state, archiver_llm))
    workflow.add_node("report_evaluator", report_evaluator_node)

    workflow.set_entry_point("coordinator")

    # 主线流程
    workflow.add_edge("coordinator", "prepare_chapter")
    workflow.add_edge("prepare_chapter", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", "human_review")
    workflow.add_edge("strategist", "human_review")

    # 条件路由：支持两阶段架构的分支决策
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "prepare_chapter": "prepare_chapter",   # 下一章 / 进入推演阶段
            "strategist": "strategist",              # 生成/重生成战略蓝图
            "researcher": "researcher",              # 数据修订
            "analyst": "analyst",                    # 逻辑修订
            "writer": "writer",                      # 写作修订
            "end": "archiver"                        # 终结报告
        }
    )

    # 终结路径
    workflow.add_edge("archiver", "report_evaluator")
    workflow.add_edge("report_evaluator", END)

    # 编译：启用 MemorySaver 持久化 + human_review 前中断
    checkpointer = MemorySaver()
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )
    return app
```

**工作流拓扑图**：

```
coordinator → prepare_chapter → researcher → analyst → writer → reviewer
       ↑                                                                ↓
       │                                                         human_review
       │                                                    ↙    ↓    ↘
       │                                          strategist  revise:*  end
       │                                              ↓                   ↓
       │                                         human_review         archiver → report_evaluator → END
       └───────────────────────────────────────────────┘
```

---

### 1.2 三层记忆架构状态定义（`state.py`）

```python
from typing import TypedDict, List, Dict, Annotated, Optional
import operator

class GraphState(TypedDict):
    """三层记忆架构:
    1. 长期记忆: Milvus RAG系统 (仅Researcher可访问)
    2. 短期工作区: chapter_scratchpad (章节专属沙盒，阅后即焚)
    3. 神圣上下文池: context_pool (仅存入审核通过的定稿)
    """

    # --- 输入层 ---
    user_input: str

    # --- 全局规划层 ---
    global_plan: List[Dict]       # 完整章节大纲（含 title, phase, analysis_model 元数据）
    current_chapter_index: int

    # --- 战略蓝图层（两阶段架构） ---
    strategic_blueprint: Optional[Dict]   # mission, swot_analysis, tows_strategies, pillars, kpis
    current_phase: str                    # "diagnosis" 或 "initiatives"

    # --- 上下文层 (长期/跨章节记忆) ---
    context_pool: Annotated[List[str], operator.add]  # operator.add 确保纯累加，不覆盖
    context_summary: str                               # 压缩后的全局上下文摘要

    # --- 当前章节层 (短期/工作区记忆) ---
    chapter_title: str
    chapter_question: str
    chapter_context: str
    chapter_scratchpad: Dict              # 本章结构化草稿本（阅后即焚）
    current_draft: str
    _pending_chapter_knowledge: Dict      # human_review 保存的上一章知识

    # --- 控制层 ---
    human_feedback: Dict
    review_decision: str                  # "approve", "revise:data", "revise:logic", "revise:writing", "finished"
    auto_revision_count: int
    llm_review_result: Optional[Dict]     # LLM 评审结果（score, issues, suggestion）

    # --- 输出层 ---
    final_report: str
    report_evaluation: Optional[Dict]     # 全文评审结果（5维度评分）
```

**记忆流转示意**：

```
                    ┌─────────────────────┐
  Milvus RAG ──────►│   长期记忆 (RAG)     │◄──── 仅 Researcher 可访问
  (持久化向量库)     │   document chunks    │
                    └─────────────────────┘

                    ┌─────────────────────┐
  reviewer通过 ────►│   神圣上下文池        │─────► 下一章 context_summary
  human approve     │   context_pool       │      (滚动压缩)
                    │   operator.add 累加   │
                    └─────────────────────┘

                    ┌─────────────────────┐
  prepare_chapter ─►│   短期工作区          │─────→ researcher / analyst / writer
  (清空)            │   chapter_scratchpad │       共享本章中间数据
                    │   (阅后即焚)          │
                    └─────────────────────┘
```

---

### 1.3 条件路由函数（`human_review.py` — `should_continue`）

```python
def should_continue(state: Dict[str, Any]) -> str:
    """增强的路由函数，支持两阶段战略推演架构的决策路由。"""
    decision = state.get("review_decision")
    current_index = state.get("current_chapter_index", 0)
    global_plan = state.get("global_plan", [])
    blueprint = state.get("strategic_blueprint", {})
    context_pool = state.get("context_pool", [])

    # 修订路由
    if decision == "revise:data":
        return "researcher"
    elif decision == "revise:logic":
        return "analyst"
    elif decision == "revise:writing":
        return "writer"
    elif decision == "finished":
        return "end"

    # 战略蓝图路由
    if decision == "revise_blueprint":
        return "strategist"
    if decision == "approve_blueprint":
        return "prepare_chapter"

    # 正常章节审批
    if decision == "approve" or decision is None:
        # 第三章完成后 → 触发战略蓝图生成
        if current_index == 2 and len(context_pool) >= 3:
            if not blueprint or not blueprint.get("approved"):
                return "strategist"
            else:
                if current_index < len(global_plan):
                    return "prepare_chapter"
                else:
                    return "end"

        # 正常流转到下一章
        if current_index < len(global_plan):
            return "prepare_chapter"
        else:
            return "end"

    return "end"
```

---

## 第二部分：不同章节任务的提示词原始代码段落

### 2.1 Coordinator 节点 — 固定8章大纲（`coordinator.py`）

不使用 LLM 动态生成，返回包含阶段和分析模型元数据的固定结构：

```python
global_plan = [
    {
        "title": "第一章：宏观政策环境与时代要求",
        "phase": "diagnosis",
        "analysis_model": "PEST模型 (侧重P-政策与E-经济维度)",
        "index": 0
    },
    {
        "title": "第二章：区域战略与'交通强省'建设剖析",
        "phase": "diagnosis",
        "analysis_model": "无特定模型，侧重省级政策承接与区域占位分析",
        "index": 1
    },
    {
        "title": "第三章：行业演进趋势与当前内部诊断",
        "phase": "diagnosis",
        "analysis_model": "波特五力模型与SWOT分析 (强制要求在分析结尾输出结构化的SWOT矩阵)",
        "index": 2
    },
    {
        "title": "第四章：总体战略思路与政策响应目标",
        "phase": "initiatives",
        "analysis_model": "平衡计分卡(BSC)模型 (从财务、客户/民生、内部运营、学习与成长四个维度设定目标)",
        "index": 3
    },
    {
        "title": "第五章：主责主业：高质量建设与保通保畅举措",
        "phase": "initiatives",
        "analysis_model": "BCG波士顿矩阵 (将主业作为'现金牛'业务，侧重精益化与稳健回报)",
        "index": 4
    },
    {
        "title": "第六章：创新驱动：绿色低碳与智慧交投建设",
        "phase": "initiatives",
        "analysis_model": "安索夫矩阵 (将创新业务作为新产品/新市场拓展，侧重第二增长曲线)",
        "index": 5
    },
    {
        "title": "第七章：产业协同：交旅融合与服务地方经济",
        "phase": "initiatives",
        "analysis_model": "产业链协同与ESG社会责任模型",
        "index": 6
    },
    {
        "title": "第八章：治理效能：深化国企改革与党建引领",
        "phase": "initiatives",
        "analysis_model": "麦肯锡7S模型 (从结构、制度、风格、员工、技能等维度构建组织保障)",
        "index": 7
    }
]
```

---

### 2.2 Researcher 节点 — 多查询生成与自适应检索（`researcher.py`）

**查询生成 Prompt**：

```python
prompt = f"""Given the following research question, generate 5-7 diverse and specific search queries that would help retrieve relevant documents.

Research Question: {chapter_question}

Context: {chapter_context if chapter_context else "No additional context provided"}
{revision_section}
Generate 5-7 search queries (one per line) that:
1. Rephrase the question in different ways
2. Include related keywords and concepts
3. Cover different aspects of the topic
4. Are specific enough to retrieve relevant documents

Queries:"""
```

**检索充分性评估 Prompt**：

```python
prompt = f"""评估以下检索结果是否能充分回答研究问题。

研究问题: {chapter_question}

检索到 {doc_count} 个文档:
{preview}

请判断: 这些文档的信息是否足以支撑对该问题的深入分析？
只回答 YES 或 NO。"""
```

**补充查询生成 Prompt**：

```python
prompt = f"""基于初始检索结果，生成 2 个补充搜索查询。

研究问题: {chapter_question}
已有查询: {existing_keywords}
已检索到 {len(docs)} 个文档，但信息不够充分。

请生成 2 个不同角度的补充查询（每行一个），聚焦尚未覆盖的方面:"""
```

---

### 2.3 Analyst 节点 — 战略分析模型注入（`analyst.py`）

**分析主 Prompt（含模型注入与修订反馈）**：

```python
prompt = f"""使用指定的战略分析模型分析以下检索到的文档。

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
```

**8种战略分析模型指令（`_get_model_instruction`）**：

#### PEST 模型（第一章）

```python
"""
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
```

#### SWOT 分析（第三章）

```python
"""
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
```

#### BCG 波士顿矩阵（第五章）

```python
"""
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
```

#### 安索夫矩阵（第六章）

```python
"""
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
```

#### ESG与产业链协同（第七章）

```python
"""
**强制使用ESG社会责任与产业链协同模型分析**：
从以下维度分析：

- **Environment (环境)**: 绿色发展、低碳转型、环境保护
- **Social (社会)**: 员工福祉、社区关系、公共安全、社会责任
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
```

#### 麦肯锡7S模型（第八章）

```python
"""
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
```

#### 平衡计分卡BSC（第四章）

```python
"""
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
```

#### 波特五力模型（第三章）

```python
"""
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
```

---

### 2.4 Writer 节点 — 章节写作提示词（`writer.py`）

**写作主 Prompt（含战略蓝图约束、模型指令、修订反馈）**：

```python
prompt = f"""你是一位资深的国企战略规划报告撰写专家。

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
```

**战略蓝图约束注入（推演阶段专用）**：

```python
def _build_blueprint_constraint(strategic_blueprint: Dict = None) -> str:
    constraint_parts = ["## 战略蓝图约束（本章节必须遵循）\n"]

    if mission:
        constraint_parts.append(f"**核心使命**: {mission}\n")
    if pillars:
        constraint_parts.append("**战略支柱**:")
        for i, pillar in enumerate(pillars, 1):
            constraint_parts.append(f"{i}. {pillar}")

    if kpis:
        constraint_parts.append("**关键绩效指标 (KPIs)**:")
        for dimension, metrics in kpis.items():
            ...

    constraint_parts.append("\n**强制要求**: 在撰写本章时，必须:")
    constraint_parts.append("1. 显式说明本章举措如何支撑上述核心使命")
    constraint_parts.append("2. 阐明本章内容与战略支柱的关系")
    constraint_parts.append("3. 确保提出的目标与KPI体系保持一致")
    constraint_parts.append("4. 使用'为支撑...使命'、'为实现...目标'、'落实...战略支柱'等表述")
```

**字数约束（利用近因效应）**：

```python
hard_limit = """
## ⚠️ 字数指导（写作参考）

- **目标范围**: 1200-2500个中文字符
- **最佳长度**: 1500-2200个中文字符
- **计数方式**: 统计正文中的中文字符数（不含标题、标记符号）
- **写作原则**: 宁可充分展开论点，也不要因字数限制而牺牲分析深度
- **控制技巧**: 开头概述2-3句话，每个分析维度300-500字，结尾总结1-2句话

注意：优先保证分析质量和论证完整性，字数在合理范围内即可。
"""
```

---

### 2.5 Reviewer 节点 — 章节级5维度评分（`reviewer.py`）

**诊断阶段评分标准**：

```python
_DIAGNOSIS_DIMENSIONS = """## 评分维度（每项0-20分，允许0.5分精度）— 诊断章节评估标准

### 维度一：模型运用与框架完整性 (0-20)
评估本章是否正确、完整地运用了指定分析模型（{analysis_model}）。
- 18-20: 模型框架运用精准、维度覆盖完整、各要素间逻辑衔接紧密
- 14-17: 模型框架基本正确，但维度覆盖不完整或分析深度不够
- 10-13: 模型框架使用有偏差，维度缺失或混淆，缺乏实质分析
- 0-9: 未使用指定模型或使用完全错误，仅堆砌信息

### 维度二：数据支撑与证据质量 (0-20)
### 维度三：内部逻辑与结构清晰度 (0-20)
### 维度四：内容深度与专业洞察 (0-20)
### 维度五：写作质量与规范表达 (0-20)
"""
```

**推演阶段评分标准**：

```python
_INITIATIVES_DIMENSIONS = """## 评分维度（每项0-20分，允许0.5分精度）— 推演章节评估标准

### 维度一：模型运用与战略设计严谨度 (0-20)
### 维度二：数据支撑与证据质量 (0-20)
### 维度三：内部逻辑与结构清晰度 (0-20)
### 维度四：内容深度与战略创新性 (0-20)
### 维度五：写作质量与规范表达 (0-20)
"""
```

**章节专属期望块**：

```python
_CHAPTER_EXPECTATIONS = {
    0: "本章使用PEST模型，期望政策(P)/经济(E)/社会(S)/技术(T)四维度完整覆盖，政策维度需引用具体法规文件，经济维度需有数据支撑。",
    1: "本章侧重省级政策承接与区域占位分析，期望将'交通强国'战略与省级'交通强省'目标逐级衔接，体现区域差异化定位。",
    2: "本章使用波特五力与SWOT模型，期望在分析结尾输出结构化的SWOT矩阵表格，五力分析需覆盖供应商/买方/新进入者/替代品/行业内竞争。",
    3: "本章使用平衡计分卡(BSC)模型，期望从财务、客户/民生、内部运营、学习与成长四个维度设定可量化目标与KPI。",
    4: "本章使用BCG波士顿矩阵，期望将主业明确为'现金牛'业务定位，提出精益化运营与稳健回报的具体举措。",
    5: "本章使用安索夫矩阵，期望绘制创新路线图（市场渗透/产品开发/市场开发/多元化），明确第二增长曲线的路径与时间表。",
    6: "本章使用产业链协同与ESG社会责任模型，期望在产业协同中体现'交通+'融合模式，ESG维度需覆盖环境/社会/治理三方面。",
    7: "本章使用麦肯锡7S模型，期望从七个维度构建组织保障体系。",
}
```

**评审 Prompt**：

```python
return f"""请评估以下战略规划报告章节，直接输出JSON，不要输出其他任何文字。

## 章节信息
标题: {chapter_title}
阶段: {phase_label}
分析模型: {analysis_model}
章节专属期望: {chapter_expectation}

{context_info}
{blueprint_info}

## 章节内容
{eval_text}

---

输出格式（严格遵守）:
{{"d1_score":0,"d1_analysis":"详细评语","d2_score":0,"d2_analysis":"详细评语",
  "d3_score":0,"d3_analysis":"详细评语","d4_score":0,"d4_analysis":"详细评语",
  "d5_score":0,"d5_analysis":"详细评语","total_score":0,
  "issues":["具体问题描述1","具体问题描述2"],
  "suggestions":"详细改进建议"}}

说明:
- d1=模型运用, d2=数据支撑, d3=内部逻辑, d4=内容深度, d5=写作质量
- 每项0-20分，total=五项之和
- 每个维度的analysis必须写100-200字的详细评语
- issues必须列出3-5个具体问题
- suggestions给出200字左右的总体改进建议"""
```

**阶段感知阈值与路由**：

```python
# 诊断阶段70分通过，推演阶段72分通过
approve_threshold = 72 if phase == "initiatives" else 70
suggestion = "approve" if total_score >= approve_threshold else "revise:writing"

if 50 <= total_score < approve_threshold:
    weakest = min(dimension_scores.items(), key=lambda x: x[1]["score"])
    if weakest[0] in ("model_application", "internal_logic"):
        suggestion = "revise:logic"
    elif weakest[0] in ("data_support", "content_depth"):
        suggestion = "revise:data"
    else:
        suggestion = "revise:writing"
```

---

### 2.6 Strategist 节点 — 战略蓝图生成（`strategist.py`）

**诊断阶段压缩 Prompt**：

```python
prompt = f"""请将以下诊断阶段的三章内容压缩为一份结构化综合摘要。

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
5. 3000字以内
6. 必须使用中文

诊断阶段内容:
{chapters_text}

请生成综合摘要:"""
```

**SWOT 提取 Prompt**：

```python
prompt = f"""你是一位战略分析专家。请从以下章节文本中提取结构化的SWOT分析。

章节文本:
{chapter_text}

任务要求:
1. 仔细阅读文本，识别并提取SWOT四个维度的内容
2. 每个维度至少提取3-5项
3. 确保提取的内容是原文中明确提到的优势、劣势、机会、威胁
4. 返回JSON格式

返回JSON结构:
{
    "Strengths": ["优势1", "优势2", "优势3", ...],
    "Weaknesses": ["劣势1", "劣势2", "劣势3", ...],
    "Opportunities": ["机会1", "机会2", "机会3", ...],
    "Threats": ["威胁1", "威胁2", "威胁3", ...]
}

输出要求:
- 必须使用中文
- 每项应该简洁明确（1-2句话）
- 只返回JSON对象，不要包含其他文本

现在请提取SWOT分析:"""
```

**TOWS 战略蓝图生成 Prompt**：

```python
prompt = f"""你是一位资深的国企战略规划专家。基于前期的诊断分析，现在需要制定省属国企的战略蓝图。

## 已识别的SWOT分析

**优势 (Strengths)**:
{swot_strengths}

**劣势 (Weaknesses)**:
{swot_weaknesses}

**机会 (Opportunities)**:
{swot_opportunities}

**威胁 (Threats)**:
{swot_threats}

## 任务要求

请使用TOWS矩阵分析法，生成完整的战略蓝图。返回JSON格式：

{
    "mission": "一句话凝练核心使命（20-30字）",
    "swot_analysis": { ... },
    "tows_strategies": {
        "SO": ["利用优势抓住机会的策略1（具体可执行）", ...],
        "WO": ["弥补劣势抓住机会的策略1（具体可执行）"],
        "ST": ["利用优势应对威胁的策略1（具体可执行）"],
        "WT": ["减少劣势规避威胁的策略1（具体可执行）"]
    },
    "strategic_pillars": [
        "战略支柱1：名称与描述",
        "战略支柱2：名称与描述",
        "战略支柱3：名称与描述",
        "战略支柱4：名称与描述"
    ],
    "kpis": {
        "财务维度": {{"净资产收益率(ROE)": "年均不低于5%", ...}},
        "客户/民生维度": {...},
        "运营维度": {...},
        "学习成长维度": {...}
    }
}

## 输出要求

1. **所有内容必须使用中文**
2. **mission（核心使命）**: 必须体现国企定位，20-30字，高度凝练
3. **TOWS策略**: 必须具体可执行，每类策略至少2-3项
4. **战略支柱**: 建议4个，覆盖业务升级、创新驱动、产业协同、治理提升
5. **KPIs**: 必须符合SMART原则，每个维度至少3个指标，必须包含量化数字
6. **只返回JSON对象，不要包含任何其他文本**

## 参考背景

用户请求: {user_input}

现在生成战略蓝图:"""
```

---

### 2.7 Prepare Chapter 节点 — 滚动上下文压缩与知识缺口检测（`prep_chapter.py`）

**知识压缩 Prompt**：

```python
prompt = f"""请将以下新旧知识合并为一份紧凑的滚动摘要。

要求:
1. 保留所有具体数字（金额、比例、增长率等）
2. 保留政策名称和战略定位
3. 保留关键结论和核心洞察
4. 删除冗余和重复
5. 控制在1000字以内
6. 使用中文

已有摘要:
{existing_summary if existing_summary else '（无）'}

新增知识:
{new_knowledge}

请输出合并后的摘要:"""
```

**知识缺口检测 Prompt**：

```python
prompt = f"""基于前序章节的压缩摘要，判断是否有下一章需要但尚未覆盖的关键信息。

下一章: {next_chapter_title}

前序章节摘要:
{context_summary}

如果存在知识缺口，用1-2句话描述需要补充检索的方向。如果没有明显缺口，返回 NONE。"""
```

---

### 2.8 Archiver 节点 — 执行摘要生成（`archiver.py`）

**执行摘要 Prompt**：

```python
prompt = f"""你是一位资深的国企公文写作专家。请为以下省属国企战略规划报告撰写执行摘要（Executive Summary）。

## 报告背景

**主题**: {user_input}

**核心使命**: {mission}

**战略支柱**:
{pillars}

**关键KPI**:
{kpis_formatted}

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
```

**跨章一致性检查 Prompt**：

```python
prompt = f"""检查以下报告各章节之间是否存在一致性问题。

{chapter_summaries}

请检查:
1. 跨章数据矛盾（同一指标不同数字）
2. 重大内容重复（整段相同）
3. 关键引用缺失（正文中引用但不存在的引用标记）

如果没有问题，返回 OK。如果有问题，每行列出一个问题。"""
```

---

### 2.9 Report Evaluator 节点 — 全文5维度评审（`report_evaluator.py`）

**全文评分标准 System Prompt**：

```python
REPORT_SCORING_SYSTEM_PROMPT = """你是一位拥有20年经验的国际顶尖战略咨询合伙人，专门评估战略规划报告。

你必须严格按JSON格式输出评分，不要输出任何其他文字、分析或解释。不要使用markdown代码块。直接输出纯JSON。

## 评分维度（每项0-20分，允许0.5分精度）

### 维度一：方法论运用与分析框架严谨度 (0-20)
评估各章节是否正确、深入地运用了指定的战略分析模型（PEST/SWOT/BCG/五力/BSC/安索夫/7S/ESG等）。
- 18-20: 模型框架运用精准、维度完整、各要素间逻辑衔接紧密
- 14-17: 模型框架基本正确，但维度覆盖不完整或分析深度不够
- 10-13: 模型框架使用有偏差，维度缺失或混淆
- 0-9: 未使用指定模型或使用完全错误

### 维度二：战略一致性与外部环境契合度 (0-20)
评估报告是否紧密结合政策导向（交通强国、交通强省、国企改革），对外部环境认知是否准确深入。
- 18-20: 政策引用精准、外部环境认知深刻、战略定位高度契合
- 14-17: 政策引用较准确但深度不足，或战略定位有偏差
- 10-13: 政策引用泛泛，缺乏针对性分析
- 0-9: 脱离政策背景，战略定位模糊

### 维度三：逻辑连贯性与战略闭环思维 (0-20)
评估诊断阶段→战略推演→实施举措之间是否形成完整闭环，章节间逻辑是否连贯。
- 18-20: 诊断-战略-举措完美闭环，问题-对策一一对应，章节间高度连贯
- 14-17: 基本形成闭环，但部分对策缺乏诊断依据，或章节间有脱节
- 10-13: 闭环不完整，诊断与举措脱节
- 0-9: 各章节独立无关联，无闭环思维

### 维度四：创新性与前瞻洞察力 (0-20)
- 18-20: 提出多维度原创性战略洞察，前瞻判断有力，创新举措兼具突破性与可行性
- 14-17: 有一定的原创洞察和前瞻分析，但创新深度不足或部分流于常规
- 10-13: 以文档信息复述为主，缺乏独立思考和前瞻性判断
- 0-9: 完全依赖文档搬运，无任何原创洞察或前瞻分析

### 维度五：隐性约束洞察与组织治理深度 (0-20)
评估报告是否识别了组织内部的隐性约束（利益相关方博弈、变革阻力、文化惯性），治理策略是否深入。
- 18-20: 深刻洞察组织摩擦力，提出具体可行的变革管理策略
- 14-17: 识别了部分组织约束，但应对策略偏宏观
- 10-13: 对组织约束的认知停留在表面
- 0-9: 忽视组织内部约束，治理建议空泛"""
```

**全文评审 User Prompt**：

```python
prompt = f"""请评估以下战略规划报告，直接输出JSON，不要输出其他任何文字。

报告内容:
{eval_text}

---

输出格式（严格遵守）:
{{"d1_score":0,"d1_analysis":"详细评语","d2_score":0,"d2_analysis":"详细评语",
  "d3_score":0,"d3_analysis":"详细评语","d4_score":0,"d4_analysis":"详细评语",
  "d5_score":0,"d5_analysis":"详细评语","total_score":0,
  "suggestions":"详细改进建议"}}

说明:
- d1=方法论, d2=战略一致, d3=逻辑闭环, d4=创新前瞻, d5=组织治理
- 每项0-20分，total=五项之和
- 每个维度的analysis必须写100-200字的详细评语
- suggestions给出200字左右的总体改进建议，按优先级排序"""
```

---

## 架构总结

| 节点 | 输入记忆 | 输出记忆 | 关键设计 |
|------|----------|----------|----------|
| **Coordinator** | `user_input` | `global_plan`, `current_phase` | 固定8章结构+元数据（phase, analysis_model） |
| **Prepare Chapter** | `context_pool`, `_pending_knowledge` | `context_summary`（滚动压缩）, `chapter_scratchpad`（清空重建） | 阅后即焚 + 知识缺口检测 + 蓝图注入 |
| **Researcher** | `chapter_question`, Milvus RAG | `chapter_scratchpad[retrieved_docs]` | 多查询生成 + 自适应检索 + 去重 |
| **Analyst** | `chapter_scratchpad`, `analysis_model` | `chapter_scratchpad[key_facts, insights]` | 8种战略模型注入 + 结构化JSON输出 + 质量自校验 |
| **Writer** | `chapter_scratchpad`, `strategic_blueprint` | `current_draft` | 模型写作指令 + 蓝图约束 + 引用后处理 + 自修订 |
| **Reviewer** | `current_draft`, `phase` | `llm_review_result` | 阶段感知5维度评分 + 章节专属期望 + 智能修订路由 |
| **Strategist** | `context_pool[0:3]` | `strategic_blueprint` | 3章压缩 → SWOT提取 → TOWS分析 → 蓝图生成 |
| **Human Review** | `review_decision` | `context_pool`（累加）, 路由决策 | 审批写入上下文池 / 修订反馈注入 / 蓝图审批门控 |
| **Archiver** | `context_pool`, `strategic_blueprint` | `final_report` | 去重 + 执行摘要 + 蓝图附录 + 一致性检查 |
| **Report Evaluator** | `final_report` | `report_evaluation` | 全文5维度评分（与消融实验一致） |
