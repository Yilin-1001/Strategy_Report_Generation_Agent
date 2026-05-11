# 核心实现：基于多智能体协作的战略规划报告生成系统

> 本文档详细阐述系统后端各核心模块的设计原理与实现细节，内容可直接用于毕业论文的"系统实现"章节。

---

## 目录

1. [LangGraph 工作流拓扑构建](#1-langgraph-工作流拓扑构建)
2. [全局状态与三层记忆架构](#2-全局状态与三层记忆架构)
3. [Coordinator 节点——固定八章大纲生成](#3-coordinator-节点固定八章大纲生成)
4. [Prepare Chapter 节点——章节状态隔离与滚动压缩](#4-prepare-chapter-节点章节状态隔离与滚动压缩)
5. [Researcher 节点——多查询自适应检索](#5-researcher-节点多查询自适应检索)
6. [Analyst 节点——战略分析模型动态注入](#6-analyst-节点战略分析模型动态注入)
7. [Writer 节点——约束驱动的章节草稿生成](#7-writer-节点约束驱动的章节草稿生成)
8. [Reviewer 节点——五维度 LLM 评审](#8-reviewer-节点五维度-llm-评审)
9. [Human Review 节点——人机协同审核与智能路由](#9-human-review-节点人机协同审核与智能路由)
10. [Strategist 节点——TOWS 战略蓝图生成](#10-strategist-节点tows-战略蓝图生成)
11. [Archiver 节点——终稿归档与一致性校验](#11-archiver-节点终稿归档与一致性校验)

---

## 1. LangGraph 工作流拓扑构建

### 1.1 设计背景

本系统采用 LangGraph 框架构建多智能体协作工作流。LangGraph 提供了基于有向无环图（DAG）的状态机抽象，支持节点注册、边连接、条件路由和执行挂起等能力，天然适用于需要"多步推理 + 人工介入"的报告生成场景。

### 1.2 核心代码

```python
def create_report_graph():
    workflow = StateGraph(GraphState)

    # ── 注册节点 ──
    workflow.add_node("coordinator",      lambda s: coordinator_node(s, coordinator_llm))
    workflow.add_node("prepare_chapter",  lambda s: prepare_chapter_node(s, prep_llm))
    workflow.add_node("researcher",       lambda s: researcher_node(s, retriever, researcher_llm))
    workflow.add_node("analyst",          lambda s: analyst_node(s, analyst_llm))
    workflow.add_node("writer",           lambda s: writer_node(s, writer_llm))
    workflow.add_node("reviewer",         reviewer_node)
    workflow.add_node("human_review",     human_review_node)
    workflow.add_node("strategist",       lambda s: strategist_node(s, strategist_llm))
    workflow.add_node("archiver",         lambda s: archiver_node(s, archiver_llm))

    # ── 线性主流程 ──
    workflow.set_entry_point("coordinator")
    workflow.add_edge("coordinator",     "prepare_chapter")
    workflow.add_edge("prepare_chapter", "researcher")
    workflow.add_edge("researcher",      "analyst")
    workflow.add_edge("analyst",         "writer")
    workflow.add_edge("writer",          "reviewer")
    workflow.add_edge("reviewer",        "human_review")
    workflow.add_edge("strategist",      "human_review")

    # ── 条件路由：人工审核后的分支决策 ──
    workflow.add_conditional_edges("human_review", should_continue, {
        "prepare_chapter": "prepare_chapter",
        "strategist":      "strategist",
        "researcher":      "researcher",
        "analyst":         "analyst",
        "writer":          "writer",
        "end":             "archiver",
    })

    # ── 编译：启用断点挂起机制 ──
    app = workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"]
    )
    return app
```

### 1.3 详细解释

**节点注册阶段**：系统通过 `add_node` 将 9 个功能节点注册到 StateGraph 中。每个节点对应一个具体的智能体角色（如 Researcher 负责检索、Analyst 负责分析等）。节点函数通过 `lambda` 表达式注入各自所需的外部依赖（如 LLM 管理器、RAG 检索器），实现了依赖注入与关注点分离。

**线性主链**：从 `coordinator` 到 `human_review` 的 7 条有向边构成了报告生成的线性主流程——"规划 → 准备 → 检索 → 分析 → 写作 → 评审 → 人工审核"。这种管道式（Pipeline）设计确保了每个章节的生成过程严格遵循"先检索后分析、先分析后写作"的认知顺序。

**条件路由**：`add_conditional_edges` 以 `should_continue` 函数为路由决策器，根据人工审核的结果动态选择后续路径。该机制支持五种分支：继续下一章（`prepare_chapter`）、生成战略蓝图（`strategist`）、数据/逻辑/写作修订（分别回到 `researcher`/`analyst`/`writer`）、以及终稿归档（`archiver`）。

**断点挂起**：`interrupt_before=["human_review"]` 使工作流在每次到达人工审核节点前自动挂起，等待用户通过 Gradio 前端提交审核决策后，再通过 `MemorySaver` 中的检查点恢复执行。这一机制是实现"人机协同"（Human-in-the-Loop）的核心基础。

**拓扑特征总结**：

| 拓扑特征 | 实现方式 | 设计意图 |
|---------|---------|---------|
| 线性主链 | 7条 `add_edge` | 保证认知顺序的严格性 |
| 条件分支 | `add_conditional_edges` | 支持修订回退与阶段切换 |
| 断点挂起 | `interrupt_before` | 实现人机协同审核 |
| 状态持久化 | `MemorySaver` | 支持工作流暂停与恢复 |

---

## 2. 全局状态与三层记忆架构

### 2.1 设计背景

在多智能体系统中，状态管理是核心挑战之一。本系统采用 LangGraph 的 `TypedDict` 状态模式，将工作流的全局状态定义为具有明确类型注解的结构化字典。更重要的是，本系统创新性地提出了**三层记忆架构**，通过不同生命周期和访问权限的记忆层，解决了长文档生成过程中的信息遗忘和上下文污染问题。

### 2.2 核心代码

```python
from typing import TypedDict, List, Dict, Annotated, Optional
import operator

class GraphState(TypedDict):
    """三层记忆架构:
       长期记忆 → Milvus RAG（仅 Researcher 可访问）
       短期工作区 → chapter_scratchpad（章节沙盒，阅后即焚）
       神圣上下文池 → context_pool（仅存入审核通过的定稿）
    """

    # ── 输入层 ──
    user_input: str

    # ── 全局规划层 ──
    global_plan: List[Dict]          # 8章大纲（含 title, phase, analysis_model）
    current_chapter_index: int

    # ── 两阶段架构层 ──
    strategic_blueprint: Optional[Dict]
    current_phase: str               # "diagnosis" | "initiatives"

    # ── 跨章节记忆（operator.add 确保只追加不覆盖）──
    context_pool: Annotated[List[str], operator.add]
    context_summary: str             # 滚动压缩摘要

    # ── 当前章节工作区 ──
    chapter_title: str
    chapter_question: str
    chapter_context: str
    chapter_scratchpad: Dict         # 每章清空
    current_draft: str
    _pending_chapter_knowledge: Dict

    # ── 控制层 ──
    human_feedback: Dict
    review_decision: str
    auto_revision_count: int
    llm_review_result: Optional[Dict]

    # ── 输出层 ──
    final_report: str
    report_evaluation: Optional[Dict]
```

### 2.3 详细解释

**三层记忆架构的设计动机**：在传统的单状态字典方案中，所有中间结果（检索文档、分析结论、草稿文本）共享同一命名空间，容易导致：（1）前一章节的中间数据污染后续章节的生成；（2）随着章节推进，状态字典膨胀，超出 LLM 上下文窗口限制；（3）无法区分"已验证的结论"与"未确认的推测"。

本系统通过三层记忆架构解决上述问题：

**第一层——长期记忆（Milvus RAG）**：外部知识库通过 Milvus 向量数据库持久化存储，仅 Researcher 节点拥有访问权限。这种设计确保了知识检索的专一性——其他节点（如 Writer）不会绕过检索环节直接访问原始文档，从而避免了幻觉（Hallucination）问题。

**第二层——短期工作区（`chapter_scratchpad`）**：该字段作为当前章节的临时沙盒，存储检索结果（`retrieved_docs`）、关键事实（`key_facts`）、分析洞察（`insights`）等中间产物。关键的设计决策是：**每个新章节开始时，`prepare_chapter` 节点会清空该字段**（即"阅后即焚"原则）。这确保了章节间的状态隔离——前一章的中间数据不会泄漏到后续章节。

**第三层——神圣上下文池（`context_pool`）**：该字段使用 `Annotated[List[str], operator.add]` 类型注解。`operator.add` 是一个 reducer 函数，其语义是：当节点返回 `context_pool` 的更新时，LangGraph 运行时会自动将新值追加（`+`）到现有列表中，而非覆盖。这意味着：（1）只有经过人工审核批准（`approve`）的章节定稿才会被追加到该池中；（2）一旦追加，内容不可被后续节点修改或删除；（3）所有后续章节均可读取该池中的已审核内容作为上下文参考。

**滚动上下文压缩（`context_summary`）**：随着章节推进，`context_pool` 中累积的内容会越来越长。为避免超出 LLM 的上下文窗口，系统在每个章节完成后，通过 LLM 将前序章节的关键知识压缩为约 1000 字的滚动摘要（`context_summary`），保留关键数据、政策名称和核心结论，剔除冗余细节。

**`_pending_chapter_knowledge` 的过渡机制**：当 `human_review` 节点批准一章时，系统会将该章的 `key_facts` 和 `insights` 暂存到 `_pending_chapter_knowledge` 中。随后 `prepare_chapter` 节点读取该暂存数据，调用 LLM 进行滚动压缩后存入 `context_summary`，再清空 `chapter_scratchpad`。这一"暂存→压缩→清空"的三步序列确保了知识不会在状态隔离过程中丢失。

---

## 3. Coordinator 节点——固定八章大纲生成

### 3.1 设计背景

Coordinator 是工作流的入口节点，负责将用户的原始请求转化为结构化的章节大纲。本系统采用了**固定八章结构**的方案，每章绑定特定的战略分析模型和阶段标识，这一设计基于省属国企战略规划报告的领域约束——报告的结构和使用的分析框架具有高度规范性。

### 3.2 核心代码

```python
def coordinator_node(state: Dict, llm_manager) -> Dict:
    user_input = state.get("user_input", "")

    global_plan = [
        {"title": "第一章：宏观政策环境与时代要求",
         "phase": "diagnosis",
         "analysis_model": "PEST模型 (侧重P-政策与E-经济维度)", "index": 0},

        {"title": "第二章：区域战略与'交通强省'建设剖析",
         "phase": "diagnosis",
         "analysis_model": "无特定模型，侧重省级政策承接与区域占位分析", "index": 1},

        {"title": "第三章：行业演进趋势与当前内部诊断",
         "phase": "diagnosis",
         "analysis_model": "波特五力模型与SWOT分析", "index": 2},

        {"title": "第四章：总体战略思路与政策响应目标",
         "phase": "initiatives",
         "analysis_model": "平衡计分卡(BSC)模型", "index": 3},

        {"title": "第五章：主责主业：高质量建设与保通保畅举措",
         "phase": "initiatives",
         "analysis_model": "BCG波士顿矩阵", "index": 4},

        {"title": "第六章：创新驱动：绿色低碳与智慧交投建设",
         "phase": "initiatives",
         "analysis_model": "安索夫矩阵", "index": 5},

        {"title": "第七章：产业协同：交旅融合与服务地方经济",
         "phase": "initiatives",
         "analysis_model": "产业链协同与ESG社会责任模型", "index": 6},

        {"title": "第八章：治理效能：深化国企改革与党建引领",
         "phase": "initiatives",
         "analysis_model": "麦肯锡7S模型", "index": 7},
    ]

    return {
        "global_plan": global_plan,
        "current_chapter_index": 0,
        "current_phase": "diagnosis"
    }
```

### 3.3 详细解释

**两阶段架构**：八章大纲被显式划分为两个阶段——**诊断阶段**（Diagnosis，第 1-3 章）和**推演阶段**（Initiatives，第 4-8 章）。诊断阶段侧重现状分析、环境扫描和问题识别；推演阶段侧重战略举措设计、实施路径规划和资源配置。两阶段之间通过 Strategist 节点生成"战略蓝图"作为桥梁，确保推演阶段的战略举措与诊断阶段的发现保持逻辑一致性。

**模型绑定**：每章通过 `analysis_model` 元数据绑定一个战略分析框架。这种设计使得后续的 Analyst 节点和 Writer 节点能够根据当前章节的元数据动态调整行为——Analyst 会按指定模型的结构化维度提取关键事实，Writer 会按指定模型的逻辑框架组织文本结构。模型与章节的对应关系如下：

| 章节 | 阶段 | 战略分析模型 | 分析目标 |
|------|------|------------|---------|
| 第一章 | Diagnosis | PEST 模型 | 宏观政策环境扫描 |
| 第二章 | Diagnosis | 政策承接分析 | 省级战略定位 |
| 第三章 | Diagnosis | 波特五力 + SWOT | 行业竞争与内部诊断 |
| 第四章 | Initiatives | 平衡计分卡 (BSC) | 战略目标设定 |
| 第五章 | Initiatives | BCG 波士顿矩阵 | 主业经营策略 |
| 第六章 | Initiatives | 安索夫矩阵 | 创新增长路径 |
| 第七章 | Initiatives | ESG + 产业链协同 | 社会责任与生态构建 |
| 第八章 | Initiatives | 麦肯锡 7S | 组织保障体系 |

**固定结构的选择理由**：虽然 LLM 具备动态生成大纲的能力，但本系统选择固定结构的原因有三：（1）省属国企战略规划报告具有强规范性和行业惯例，章节结构不宜随意变化；（2）固定结构允许预先设计每章的分析模型与评估标准，确保评审的公平性和可比较性；（3）固定结构简化了工作流的状态管理——系统可以精确控制每个阶段的触发条件和路由逻辑。

---

## 4. Prepare Chapter 节点——章节状态隔离与滚动压缩

### 4.1 设计背景

`prepare_chapter` 是每个章节迭代的第一步（在 Coordinator 之后），负责为新章节的生成准备干净的工作环境。该节点实现了两个关键机制：**滚动上下文压缩**和**章节状态隔离**。

### 4.2 核心代码

```python
def prepare_chapter_node(state: Dict, llm_manager=None) -> Dict:
    # Step 1: 读取上一章的知识（由 human_review 暂存）
    pending_knowledge = state.get("_pending_chapter_knowledge", {})
    current_summary = state.get("context_summary", "")

    # Step 2: 滚动压缩——将上一章的 key_facts/insights 合并到 context_summary
    context_summary_update = {}
    if pending_knowledge and pending_knowledge.get("key_facts"):
        compressed = _compress_chapter_knowledge(
            chapter_title=pending_knowledge["title"],
            key_facts=pending_knowledge["key_facts"],
            insights=pending_knowledge.get("insights", []),
            existing_summary=current_summary,
            llm_manager=llm_manager
        )
        context_summary_update = {"context_summary": compressed}

    # Step 3: 设置新章节状态
    global_plan = state["global_plan"]
    current_index = state["current_chapter_index"]
    chapter_metadata = global_plan[current_index]
    chapter_title = chapter_metadata["title"]

    # CRITICAL: 清空 chapter_scratchpad（状态隔离）
    chapter_scratchpad = {}

    # 推演阶段特殊处理：注入战略蓝图约束
    if chapter_metadata["phase"] == "initiatives" and state.get("current_phase") == "initiatives":
        blueprint = state.get("strategic_blueprint", {})
        if blueprint.get("approved", False):
            chapter_scratchpad["strategic_blueprint"] = blueprint

    # Step 4: 知识缺口检测（Agent 化决策点）
    if context_summary_update.get("context_summary"):
        gap_hint = _detect_knowledge_gaps(
            next_chapter_title=chapter_title,
            context_summary=context_summary_update["context_summary"],
            llm_manager=llm_manager
        )
        if gap_hint:
            chapter_context += f"\n\n[知识缺口提示]: {gap_hint}"

    return {
        "chapter_title": chapter_title,
        "chapter_question": extract_question(chapter_title),
        "chapter_context": chapter_context,
        "chapter_scratchpad": chapter_scratchpad,  # 干净的工作区
        "current_draft": "",
        **context_summary_update
    }
```

### 4.3 详细解释

**滚动上下文压缩机制**：`_compress_chapter_knowledge` 函数采用 LLM 驱动的增量压缩策略。当上一章被批准后，该章的关键事实（`key_facts`）和洞察（`insights`）会被传递给 LLM，与现有的 `context_summary` 合并为一份新的紧凑摘要。LLM 被指示保留所有具体数字（金额、比例、增长率）、政策名称和战略定位，同时删除冗余和重复，将输出控制在 1000 字以内。这种"增量压缩"策略的数学直觉类似于概率论中的贝叶斯更新——每新增一章信息，就对全局认知进行一次后验修正。

**章节状态隔离**：`chapter_scratchpad = {}` 这一行看似简单，实则承载了核心设计哲学。通过在每个章节开始时清空工作区，系统确保了：（1）前一章的检索文档、分析结论等中间产物不会影响当前章节；（2）当前章节的所有中间结果都是独立产生的，不存在跨章污染；（3）如果用户要求修订前一章，修订过程不受后续章节中间产物的干扰。

**推演阶段的蓝图注入**：当系统进入 Initiatives 阶段（第 4-8 章），且战略蓝图已被人工批准时，`prepare_chapter` 会将蓝图注入到新章节的 `chapter_scratchpad` 中。这使得后续的 Analyst 和 Writer 节点能够感知战略蓝图的约束——确保推演阶段的每一章都与整体战略方向保持一致。

**知识缺口检测**：`_detect_knowledge_gaps` 是一个 Agent 化的决策点。它使用 LLM 判断前序章节的压缩摘要是否已覆盖下一章所需的关键信息。如果检测到知识缺口，系统会在 `chapter_context` 中附加提示信息，引导 Researcher 节点在检索时针对性地补充缺失信息。这一机制体现了"主动式信息获取"（Proactive Information Seeking）的设计理念。

---

## 5. Researcher 节点——多查询自适应检索

### 5.1 设计背景

Researcher 节点是系统与外部知识库（Milvus）之间的唯一接口，负责根据当前章节的研究问题检索相关文档。该节点实现了**多查询检索**（Multi-Query Retrieval）和**自适应补充检索**（Adaptive Supplementary Retrieval）两阶段策略，以平衡检索的召回率（Recall）与精确率（Precision）。

### 5.2 核心代码

```python
def researcher_node(state, retriever, llm_manager):
    question = state["chapter_question"]
    scratchpad = state["chapter_scratchpad"]

    # ① LLM 生成 5 个多样化查询
    queries = _generate_queries(question, context, llm_manager)  # 返回 5-7 个查询

    # ② 多查询并行检索
    all_docs = []
    for query in queries:
        docs = retriever.search(query, top_k=20)  # 混合检索（稠密 + BM25）
        all_docs.extend(docs)

    # ③ SHA-256 文档去重
    unique_docs = _deduplicate_documents(all_docs)

    # ④ 自适应补充检索：LLM 评估信息充足性
    if len(unique_docs) < 10 or not _evaluate_retrieval_sufficiency(question, unique_docs, llm_manager):
        supp_queries = _generate_supplementary_queries(question, queries, unique_docs, llm_manager)
        for query in supp_queries:
            all_docs.extend(retriever.search(query, top_k=20))
        unique_docs = _deduplicate_documents(all_docs)

    # ⑤ 按相关性得分排序，取 Top-20
    unique_docs.sort(key=lambda d: d.get("score", 0), reverse=True)
    top_docs = unique_docs[:20]

    scratchpad["queries"] = queries
    scratchpad["retrieved_docs"] = top_docs
    return {"chapter_scratchpad": scratchpad}
```

### 5.3 详细解释

**多查询生成（Multi-Query Generation）**：单一查询往往无法覆盖研究问题的所有方面。本系统首先利用 LLM 将章节研究问题转化为 5-7 个不同角度的搜索查询，这些查询从不同侧面重述问题（如同义改写、关键词替换、子问题分解）。这种策略的理论基础来自信息检索领域的**查询扩展**（Query Expansion）研究——通过多样化的查询表达提高检索召回率。

**文档去重**：多个查询可能返回相同的文档片段。系统使用 SHA-256 哈希对文档文本进行指纹计算，通过 `set` 集合实现 O(1) 时间复杂度的去重判断。去重是必要的，因为 Milvus 的混合检索策略（稠密向量 + BM25 稀疏向量）可能对同一文档片段产生多次命中。

**自适应补充检索（Adaptive Retrieval）**：这是本系统的核心创新之一。在初始检索完成后，系统并不直接进入分析阶段，而是通过 LLM 对检索结果进行**充足性评估**。LLM 会收到研究问题和前 5 个文档的摘要（各 200 字），判断这些文档的信息是否足以支撑深入分析。如果 LLM 判断"不充足"（返回 NO），系统会生成 2 个补充查询，从不同角度再次检索。这种"检索→评估→补充"的循环确保了 Analyst 节点能够获得充分的信息输入。

**Top-20 截断**：最终结果按相关性得分降序排列，截取前 20 个文档。这一截断基于实际经验——20 个文档的文本量（约 3 万字符）恰好能够在保留充分信息的同时不超出下游 LLM 的上下文窗口限制。

**检索策略的混合性**：底层 `retriever.search` 方法实现了稠密向量检索（Dense Retrieval，基于 Embedding 相似度）和稀疏检索（Sparse Retrieval，基于 BM25 关键词匹配）的混合策略，通过倒数排名融合（Reciprocal Rank Fusion, RRF）算法合并两路结果。这种混合策略兼顾了语义相似性匹配和关键词精确匹配的优势。

---

## 6. Analyst 节点——战略分析模型动态注入

### 6.1 设计背景

Analyst 节点是系统中体现"领域知识驱动"设计的核心模块。该节点接收 Researcher 检索到的文档，根据当前章节绑定的战略分析模型，动态构建结构化的分析 prompt，引导 LLM 按照指定模型的分析框架提取关键事实和生成洞察。

### 6.2 核心代码

```python
def analyst_node(state, llm_manager):
    index = state["current_chapter_index"]
    model = state["global_plan"][index]["analysis_model"]  # 如 "SWOT分析"
    docs  = state["chapter_scratchpad"]["retrieved_docs"]

    # ① 首尾智能提取，控制 Token 预算
    doc_summary = _generate_document_summary_v2(docs, total_budget=30000)

    # ② 动态注入战略模型指令
    model_instruction = _get_model_instruction(model)
    # 例 SWOT → 强制按 S/W/O/T 四维度返回 JSON
    # 例 PEST → 强制按 P/E/S/T 四维度返回 JSON
    # 例 BCG  → 强制按 现金牛/明星/问题/瘦狗 返回 JSON

    # ③ LLM 结构化分析
    key_facts, insights = _extract_facts_and_insights(
        question, doc_summary, model_instruction, llm_manager
    )

    # ④ 质量自校验：维度完整性检查 + 低温度重试
    validation = _validate_analysis(key_facts, insights, model)
    if not validation["valid"]:
        key_facts, insights = _extract_facts_and_insights(..., temperature=0.3)

    return {"chapter_scratchpad": {
        **scratchpad,
        "key_facts": key_facts,          # 按模型维度组织的结构化事实
        "insights": insights,            # 深度分析洞察
        "analysis_model_used": model     # 记录使用的模型
    }}
```

### 6.3 详细解释

**首尾智能提取（`_smart_extract`）**：当文档文本超出字符预算时，系统采用"首尾提取"策略——保留文档的前 60% 和后 40%（中间用省略标记替代），而非简单的头部截断。这一策略基于经验观察：学术和政策文档的关键信息通常集中在开头（摘要、结论）和结尾（总结、展望），中间段落多为论证过程和支撑细节。提取函数根据文档总数动态分配单文档预算（`total_budget / num_docs`），确保所有文档都能获得公平的展示机会。

**模型指令动态注入（`_get_model_instruction`）**：这是 Analyst 节点的核心设计。该函数根据 `analysis_model` 字段的值，动态生成与模型对应的结构化分析指令。以 SWOT 模型为例，指令会强制要求 LLM 将提取的关键事实按照 `Strengths`（优势）、`Weaknesses`（劣势）、`Opportunities`（机会）、`Threats`（威胁）四个维度分类组织，并以 JSON 格式返回。系统共内置了 8 种战略分析模型的指令模板：

```
PEST       → {Political: [...], Economic: [...], Social: [...], Technological: [...]}
SWOT       → {Strengths: [...], Weaknesses: [...], Opportunities: [...], Threats: [...]}
BCG        → {现金牛业务: [...], 明星业务: [...], 问题业务: [...], 瘦狗业务: [...]}
Porter 5F  → {现有竞争者: [...], 潜在进入者: [...], 替代品: [...], 供应商: [...], 买方: [...]}
BSC        → {财务维度: [...], 客户/民生维度: [...], 内部运营维度: [...], 学习与成长维度: [...]}
Ansoff     → {市场渗透: [...], 市场开发: [...], 产品开发: [...], 多元化: [...]}
7S         → {Strategy: [...], Structure: [...], Systems: [...], Shared Values: [...], ...}
ESG        → {Environment: [...], Social: [...], Governance: [...], 产业链协同: [...]}
```

**结构化输出的意义**：通过强制 LLM 按模型维度返回 JSON 格式的结果，系统实现了分析过程的**可验证性**（Verifiability）和**可比较性**（Comparability）。例如，可以明确检查 SWOT 四个维度是否各至少包含 2 项事实，从而量化评估分析的完整性。

**质量自校验（`_validate_analysis`）**：在 LLM 返回分析结果后，系统执行规则驱动的质量校验，检查维度完整性和数量充足性。例如，对于 SWOT 模型，要求四个维度中至少有三个维度有内容，且总事实数不少于 3 项。如果校验不通过，系统会以更低的温度（temperature=0.3）重新调用 LLM 分析——低温度意味着 LLM 的输出更加确定性和保守，减少随机性导致的维度遗漏。

---

## 7. Writer 节点——约束驱动的章节草稿生成

### 7.1 设计背景

Writer 节点将 Analyst 提取的关键事实和分析洞察合成为完整的章节草稿。该节点的核心挑战在于：如何在保证内容专业性的同时，确保草稿遵循指定战略分析模型的结构框架，并与前序章节保持一致性。

### 7.2 核心代码

```python
def writer_node(state, llm_manager):
    scratchpad = state["chapter_scratchpad"]
    key_facts = scratchpad["key_facts"]
    insights = scratchpad["insights"]
    model = scratchpad["analysis_model_used"]
    blueprint = scratchpad.get("strategic_blueprint")

    # ① 构建 prompt：注入事实、洞察、模型约束、蓝图约束
    prompt = _generate_writing_prompt(
        chapter_title, key_facts, insights,
        analysis_model=model,
        strategic_blueprint=blueprint,  # 推演阶段注入
        context_summary=state["context_summary"]  # 前序章节摘要
    )

    # ② LLM 生成草稿（较高温度以获得更自然的表达）
    draft = llm_manager.invoke(prompt, temperature=0.7)

    # ③ 自校验：检查字数和引用
    validation = _validate_draft(draft)
    if validation["needs_revision"]:
        draft = _revise_draft(draft, validation, llm_manager)

    # ④ 后处理：将 "Document X" 替换为实际文件名
    draft = _replace_document_x_with_filenames(draft, filename_mapping)

    return {"current_draft": draft}
```

### 7.3 详细解释

**多约束 Prompt 构建**：Writer 节点的 Prompt 由多个约束模块叠加构成：

1. **事实与洞察模块**：将 Analyst 提取的 `key_facts` 和 `insights` 以结构化方式呈现。对于 SWOT 等模型产生的字典格式事实，会按维度分组展示；对于列表格式的事实，以编号列表呈现。

2. **战略模型写作指令（`_get_model_writing_instruction`）**：与 Analyst 的模型指令对应，Writer 也有针对每个模型的写作指令。例如，SWOT 模型要求"主体部分应按照优势-劣势-机会-威胁的结构组织"，BCG 模型要求"按现金牛/明星/问题/瘦狗分类阐述"。

3. **战略蓝图约束（`_build_blueprint_constraint`）**：仅推演阶段生效。该模块将战略蓝图的核心使命、战略支柱和 KPI 指标注入 Prompt，并明确要求 LLM"显式说明本章举措如何支撑上述核心使命"。

4. **跨章节一致性指令**：当 `context_summary` 存在时，Prompt 会加入一致性约束——"如果前序章节已提及具体数字，必须使用相同数字"。

**草稿自校验**：`_validate_draft` 检查草稿的中文字符数（要求 1000-2500 字）和引用数量（至少 1 个 `[来源: ...]` 格式的引用）。如果字数超标或缺少引用，系统会调用 `_revise_draft` 让 LLM 进行精简修订。这种"生成→校验→修订"的循环体现了自我反思（Self-Reflection）的 Agent 设计模式。

**引用后处理**：LLM 在生成草稿时可能使用"Document X"等占位符引用。系统通过 `_replace_document_x_with_filenames` 将这些占位符替换为实际的知识库文件名（如"江西省交通投资集团2023年度报告"），使最终报告的引用格式符合学术规范。

---

## 8. Reviewer 节点——五维度 LLM 评审

### 8.1 设计背景

Reviewer 节点实现了基于 LLM 的自动化章节质量评估。该节点使用与生成节点不同的独立 LLM 实例进行评审，避免自我评估偏差。评审采用五维度评分标准，每个维度 0-20 分（满分 100），并根据诊断/推演阶段采用差异化的评估标准。

### 8.2 核心代码

```python
def reviewer_node(state):
    draft = state["current_draft"]
    phase = state["global_plan"][state["current_chapter_index"]]["phase"]
    model = state["global_plan"][state["current_chapter_index"]]["analysis_model"]

    # 构建阶段自适应的系统提示词
    system_prompt = _build_system_prompt(phase, model)

    # 5 维度评审（每项 0-20 分）
    result = _call_reviewer_llm(system_prompt, draft)
    # result 包含: d1_score, d2_score, d3_score, d4_score, d5_score

    # 阶段感知阈值：诊断≥70 通过，推演≥72 通过
    threshold = 72 if phase == "initiatives" else 70
    if result["score"] >= threshold:
        result["suggestion"] = "approve"
    else:
        # 自动定位最弱维度 → 路由到对应修订节点
        weakest = min(result["dimension_scores"].items(), key=lambda x: x[1]["score"])
        if weakest[0] in ("model_application", "internal_logic"):
            result["suggestion"] = "revise:logic"     # 回到 Analyst
        elif weakest[0] in ("data_support", "content_depth"):
            result["suggestion"] = "revise:data"      # 回到 Researcher
        else:
            result["suggestion"] = "revise:writing"   # 回到 Writer

    return {"llm_review_result": result}
```

### 8.3 详细解释

**独立评审实例**：系统通过 `_init_reviewer` 初始化一个独立的 OpenAI 客户端连接到不同的 LLM 服务（如 SiliconFlow/Kimi），使用与生成节点不同的模型进行评审。这一设计遵循了"避免自我评估"的原则——如果由同一模型既生成又评审，评审结果可能存在确认偏差（Confirmation Bias）。

**五维度评分标准**：评审从五个相互独立的维度评估章节质量：

| 维度 | 英文标识 | 评估焦点 |
|------|---------|---------|
| 模型运用与框架完整性 | `model_application` | 是否正确、完整地运用了指定分析模型 |
| 数据支撑与证据质量 | `data_support` | 论据是否有充分的数据和引用支撑 |
| 内部逻辑与结构清晰度 | `internal_logic` | 段落间是否有逻辑递进，结构是否清晰 |
| 内容深度与专业水准 | `content_depth` | 是否超越信息复述，提供深度分析洞察 |
| 写作质量与规范表达 | `writing_quality` | 语言是否专业通顺，格式是否符合公文风格 |

**阶段自适应评估标准**：系统为诊断阶段和推演阶段分别设计了不同的评分描述。例如，"模型运用"维度在诊断阶段侧重"框架完整性"，而在推演阶段侧重"战略设计严谨度"；"内容深度"维度在诊断阶段侧重"从数据中提炼洞察"，在推演阶段侧重"战略创新性"。这种差异化评估反映了两个阶段的不同目标——诊断阶段追求"准确描述现状"，推演阶段追求"创造性设计未来"。

**阶段感知阈值**：诊断阶段的通过阈值为 70 分，推演阶段为 72 分。推演阶段采用更高阈值的原因在于：战略举措的设计质量对企业决策具有直接影响，需要更严格的质量把关。

**最弱维度自动路由**：当评分低于阈值时，系统会自动识别得分最低的维度，并根据维度类型映射到对应的修订节点——模型运用或逻辑维度薄弱则路由到 Analyst（`revise:logic`），数据支撑或内容深度不足则路由到 Researcher（`revise:data`），写作质量问题则路由到 Writer（`revise:writing`）。这一机制实现了"精准修订"——只重做最需要改进的环节，而非全盘推翻重来。

**章节专属期望**：`_CHAPTER_EXPECTATIONS` 字典为 8 个章节分别定义了具体的评估指引（如第一章期望"PEST 四维度完整覆盖且政策维度引用具体法规文件"），使评审更加精准和有针对性。

---

## 9. Human Review 节点——人机协同审核与智能路由

### 9.1 设计背景

Human Review 节点是工作流中唯一需要人工介入的环节。该节点承担两项职责：（1）根据用户的审核决策更新工作流状态（如将批准的草稿移入上下文池）；（2）通过路由函数 `should_continue` 决定工作流的下一步走向。

### 9.2 核心代码

**状态更新逻辑：**

```python
def human_review_node(state):
    decision = state["review_decision"]

    # 批准：将定稿追加到 context_pool，清空工作区
    if decision == "approve":
        full_chapter = f"# {chapter_title}\n\n{current_draft}"
        updated_state = {
            "context_pool": [full_chapter],  # operator.add 会追加
            "chapter_scratchpad": {},
            "_pending_chapter_knowledge": {
                "title": chapter_title,
                "key_facts": scratchpad["key_facts"],
                "insights": scratchpad["insights"]
            }
        }
        # 第三章特殊处理：暂不递增索引，等待蓝图审批
        if current_index == 2:
            updated_state["_last_approved_index"] = 2
        else:
            updated_state["current_chapter_index"] = current_index + 1
        return updated_state

    # 战略蓝图批准：进入推演阶段
    if decision == "approve_blueprint":
        blueprint["approved"] = True
        return {
            "current_phase": "initiatives",
            "strategic_blueprint": blueprint,
            "chapter_scratchpad": {},
            "current_chapter_index": current_index + 1
        }

    # 修订：将评审反馈注入 scratchpad
    if decision.startswith("revise:"):
        scratchpad["revision_feedback"] = {
            "decision": decision,
            "issues": llm_review["issues"],
            "score": llm_review["score"],
            "improvement_hints": llm_review["improvement_hints"]
        }
        return {
            "auto_revision_count": current_count + 1,
            "chapter_scratchpad": scratchpad
        }
```

**路由函数：**

```python
def should_continue(state) -> str:
    decision = state["review_decision"]
    index = state["current_chapter_index"]
    blueprint = state.get("strategic_blueprint", {})
    pool_len = len(state.get("context_pool", []))

    # 修订路由
    if decision == "revise:data":    return "researcher"
    if decision == "revise:logic":   return "analyst"
    if decision == "revise:writing": return "writer"
    if decision == "revise_blueprint": return "strategist"

    # 蓝图批准 → 进入推演阶段
    if decision == "approve_blueprint": return "prepare_chapter"

    # 诊断三章完成 → 触发战略蓝图生成
    if decision == "approve" and index == 2 and pool_len >= 3:
        if not blueprint.get("approved"):
            return "strategist"

    # 正常流转
    if decision == "approve":
        if index < len(state["global_plan"]):
            return "prepare_chapter"
        return "end"

    return "end"
```

### 9.3 详细解释

**`context_pool` 的追加语义**：`human_review_node` 在处理"批准"操作时，仅返回 `[full_chapter]`（一个只包含当前章节的列表），而非整个 `context_pool`。这是因为 `context_pool` 的类型注解 `Annotated[List[str], operator.add]` 使得 LangGraph 运行时自动将返回值追加到现有列表。如果返回完整列表，会导致内容重复。这一设计要求开发者深刻理解 LangGraph 的 reducer 机制。

**修订反馈的结构化注入**：当用户选择修订时，系统将 LLM 评审结果（评分、问题列表、各维度评语、改进建议）和用户的手动批注合并为结构化的 `revision_feedback` 字典，注入到 `chapter_scratchpad` 中。下游的 Researcher/Analyst/Writer 节点会读取各自角色的 `improvement_hints`（如 `researcher` 对应数据检索建议、`analyst` 对应分析逻辑建议、`writer` 对应写作改进建议），实现"精准修订"。

**第三章的特殊处理**：当第三章（诊断阶段最后一章）被批准后，系统**不递增** `current_chapter_index`，而是将索引保持在 2，等待战略蓝图的审批。路由函数 `should_continue` 在检测到 `index == 2 && context_pool >= 3 && blueprint 未批准` 时，将工作流路由到 `strategist` 节点生成战略蓝图。蓝图经人工批准后（`approve_blueprint`），索引才递增到 3，正式进入推演阶段。这一"暂停-生成蓝图-审批-继续"的四步序列是两阶段架构的核心控制流。

**修订历史的追溯性**：每次修订操作都会在 `revision_history` 中记录版本号、时间戳、操作类型（补充数据/重新分析/重写内容）、评审分数和草稿预览。这不仅为用户提供了修订历史的可视化追踪，也为系统的消融实验（Ablation Study）提供了定量数据。

---

## 10. Strategist 节点——TOWS 战略蓝图生成

### 10.1 设计背景

Strategist 节点在诊断阶段（前三章）完成后触发，负责将诊断阶段的分析结论（特别是 SWOT 分析）转化为结构化的战略蓝图。该蓝图包含核心使命、TOWS 战略组合、战略支柱和量化 KPI，是连接诊断阶段与推演阶段的核心桥梁。

### 10.2 核心代码

```python
def strategist_node(state, llm_manager):
    context_pool = state["context_pool"]  # 前三章已完成内容

    # ① 将诊断阶段三章压缩为结构化综合摘要
    diagnosis_summary = _compress_diagnosis(context_pool, user_input, llm_manager)

    # ② 从综合摘要中提取结构化 SWOT 矩阵
    swot = _extract_swot_from_chapter(diagnosis_summary, llm_manager)
    # → {"Strengths": [...], "Weaknesses": [...],
    #    "Opportunities": [...], "Threats": [...]}

    # ③ TOWS 矩阵分析 → 生成完整战略蓝图
    blueprint = _generate_strategic_blueprint(swot, user_input, llm_manager)
    # 包含:
    #   mission:           "服务交通强省战略，打造一流国有资本投资运营平台..."
    #   swot_analysis:     {strengths, weaknesses, opportunities, threats}
    #   tows_strategies:   {SO: [...], WO: [...], ST: [...], WT: [...]}
    #   strategic_pillars: ["战略支柱1：...", "战略支柱2：...", ...]
    #   kpis:              {财务维度: {...}, 客户/民生维度: {...}, ...}

    blueprint["approved"] = False  # 待人工审核

    return {"strategic_blueprint": blueprint, "current_draft": ""}
```

### 10.3 详细解释

**诊断阶段压缩**：`_compress_diagnosis` 将前三章的全部内容（而非仅第三章）传递给 LLM，生成一份约 3000 字的结构化综合摘要。摘要按"宏观环境要点→区域战略定位→行业竞争态势→内部优势与劣势→外部机遇与威胁→关键数据汇总"的固定结构组织。这一步骤至关重要——它确保了蓝图生成是基于诊断阶段的全部发现，而非仅依赖 SWOT 分析章节。

**SWOT 提取**：`_extract_swot_from_chapter` 通过 LLM 从综合摘要中识别并提取四个维度的内容。LLM 被要求为每个维度提取 3-5 项，确保矩阵的完整性和平衡性。输出为严格的 JSON 格式，便于后续的结构化处理。

**TOWS 矩阵分析**：TOWS 矩阵是 SWOT 分析的战略推演延伸，通过将内部因素（S/W）与外部因素（O/T）进行交叉组合，生成四类战略选项：

- **SO 策略**（优势-机会）：利用内部优势抓住外部机遇（增长型策略）
- **WO 策略**（劣势-机会）：弥补内部劣势以利用外部机遇（扭转型策略）
- **ST 策略**（优势-威胁）：利用内部优势应对外部威胁（多元化策略）
- **WT 策略**（劣势-威胁）：减少内部劣势规避外部威胁（防御型策略）

**战略支柱与 KPI 的推导**：基于 TOWS 战略选项，LLM 进一步推导出 3-5 个战略支柱（覆盖业务升级、创新驱动、产业协同、治理提升等维度）和按平衡计分卡四维度组织的量化 KPI。KPI 被要求符合 SMART 原则（具体、可衡量、可达成、相关、有时限），每个维度至少包含 3 个带有具体目标值的指标。

**蓝图的约束力**：生成的蓝图以 `approved: False` 状态返回，经人工审核批准后，会在后续的 `prepare_chapter` 节点中被注入到推演阶段每一章的工作区中。这意味着第 4-8 章的每一个战略举措都必须与蓝图的使命、支柱和 KPI 保持一致，从而确保整份报告的逻辑连贯性。

---

## 11. Archiver 节点——终稿归档与一致性校验

### 11.1 设计背景

Archiver 是工作流的最后一个处理节点，在所有 8 个章节完成并通过人工审核后触发。该节点负责将散落的章节内容组装为完整的最终报告，并生成执行摘要和战略蓝图附录。

### 11.2 核心代码

```python
def archiver_node(state, llm_manager):
    context_pool = state["context_pool"]
    blueprint = state.get("strategic_blueprint", {})

    # 0. 章节去重：保留每个章节的最后审核版本
    context_pool = _deduplicate_chapters(context_pool)

    # 1. 生成封面
    cover = _create_cover(user_input)

    # 2. 生成执行摘要（1000字国企公文语态）
    executive_summary = _generate_executive_summary(
        context_pool, blueprint, user_input, llm_manager
    )

    # 3. 生成目录
    toc = _create_table_of_contents(context_pool)

    # 4. 合并章节
    chapters = "\n\n---\n\n".join(context_pool)

    # 5. 战略蓝图附录
    appendix = _create_blueprint_appendix(blueprint)

    # 6. 跨章一致性校验
    issues = _validate_report_consistency(context_pool, llm_manager)

    full_report = cover + executive_summary + toc + chapters + appendix
    return {"final_report": full_report}
```

### 11.3 详细解释

**章节去重**：在修订流程中，同一章节可能有多个版本被追加到 `context_pool`。`_deduplicate_chapters` 以章节标题为唯一键，仅保留每个章节的最后审核版本，确保报告中不会出现重复内容。

**执行摘要生成**：`_generate_executive_summary` 使用 LLM 将整份报告浓缩为约 1000 字的执行摘要，采用国企公文语态（如"深入贯彻"、"全面落实"等规范表述）。摘要由三部分构成：开篇（政策背景与时代要求，100-150 字）、主体（战略重点阐述，700-850 字）、结尾（愿景与承诺，50-100 字）。LLM 接收所有章节的简要概述（标题 + 前 300 字）和战略蓝图信息作为输入。

**跨章一致性校验**：`_validate_report_consistency` 是 Archiver 节点的一个 Agent 化决策点。它将各章节的摘要提交给 LLM，检查是否存在跨章数据矛盾（如同一指标在不同章节中数字不一致）、重大内容重复或关键引用缺失。如果发现问题，系统会将一致性审查备注附加到报告末尾，供最终人工审核参考。

**引用修复**：`_fix_generic_citations` 扫描报告中残留的通用引用标记（如 `[来源: 来源文档_3]`），通过在 Milvus 中检索引用上下文附近的文本来定位实际的知识库源文件名，将通用标记替换为具体的文件名，提升报告的可信度和可追溯性。

**最终报告结构**：

```
封面（标题、日期、主题）
执行摘要（~1000字，国企公文语态）
目录
第一章：宏观政策环境与时代要求
第二章：区域战略与'交通强省'建设剖析
第三章：行业演进趋势与当前内部诊断
第四章：总体战略思路与政策响应目标
第五章：主责主业：高质量建设与保通保畅举措
第六章：创新驱动：绿色低碳与智慧交投建设
第七章：产业协同：交旅融合与服务地方经济
第八章：治理效能：深化国企改革与党建引领
附录：战略蓝图详述（SWOT矩阵、TOWS策略、战略支柱、KPIs）
一致性审查备注（如有）
```

---

## 附录：工作流完整执行时序

```
用户输入请求
    │
    ▼
┌─────────────── Coordinator ──────────────┐
│  生成固定8章大纲（含 phase + model 元数据） │
└──────────────────┬────────────────────────┘
                   │
    ══════════════ 每章循环（i = 0..7）══════════════
    ║                                                  ║
    ▼                                                  ║
┌── Prepare Chapter ──┐                               ║
│ 滚动压缩前章知识      │                               ║
│ 清空 scratchpad      │                               ║
│ 注入蓝图（推演阶段）  │                               ║
└────────┬────────────┘                               ║
         │                                            ║
         ▼                                            ║
┌── Researcher ──┐                                   ║
│ 多查询检索      │                                   ║
│ 自适应补充检索  │                                   ║
│ SHA-256 去重    │                                   ║
└────────┬────────┘                                   ║
         │                                            ║
         ▼                                            ║
┌── Analyst ──┐                                      ║
│ 模型动态注入  │                                      ║
│ 结构化分析    │                                      ║
│ 质量自校验    │                                      ║
└────────┬────────┘                                   ║
         │                                            ║
         ▼                                            ║
┌── Writer ──┐                                       ║
│ 多约束生成    │                                      ║
│ 字数/引用校验 │                                      ║
│ 引用后处理    │                                      ║
└────────┬────────┘                                   ║
         │                                            ║
         ▼                                            ║
┌── Reviewer ──┐                                     ║
│ 5维独立评审    │                                     ║
│ 最弱维度路由   │                                     ║
└────────┬────────┘                                   ║
         │                                            ║
         ▼                                            ║
┌── Human Review ──┐ ╌╌╌ 断点挂起 ╌╌╌╌               ║
│ 等待用户决策      │ ←─── Gradio 前端交互             ║
│ approve → 下一章  │                                 ║
│ revise:* → 修订   │ ──→ 回到对应节点                ║
└────────┬──────────┘                                ║
         │                                            ║
    ══════╧════════════════════════════════════════════╛
         │
         │  ← 第三章完成后触发 →
         ▼
┌── Strategist ──┐
│ 诊断压缩        │
│ SWOT 提取       │
│ TOWS 矩阵分析   │
│ 蓝图生成        │
└────────┬────────┘
         │
         ▼
┌── Human Review ──┐ ╌╌╌ 蓝图审核 ╌╌╌╌
│ approve_blueprint │ → 进入推演阶段
│ revise_blueprint  │ → 重新生成蓝图
└──────────────────┘

... 继续第4-8章循环 ...

         │
         ▼  （所有章节完成）
┌── Archiver ──┐
│ 章节去重      │
│ 执行摘要生成  │
│ 跨章一致性校验│
│ 引用修复      │
└────────┬──────┘
         │
         ▼
┌── Report Evaluator ──┐
│ 全文5维度评审          │
└────────┬──────────────┘
         │
         ▼
    最终报告输出
```
