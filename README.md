# RAG Project -- 基于检索增强生成的企业战略报告系统

本项目是一个企业级 RAG (Retrieval-Augmented Generation) 系统，集成了完整的文档处理管线和基于 LangGraph 的多智能体报告生成框架。系统支持从文档摄入、向量化存储到多章节战略报告全自动生成，并内置人工审核机制以确保输出质量。

## 目录

- [系统架构](#系统架构)
- [RAG 模块](#rag-模块)
  - [文档加载与分块](#1-文档加载与分块)
  - [向量化嵌入](#2-向量化嵌入)
  - [向量存储与检索](#3-向量存储与检索)
  - [重排序](#4-重排序reranking)
  - [管线入口](#5-管线入口)
- [Agent 模块](#agent-模块)
  - [设计理念](#1-设计理念)
  - [LangGraph 工作流构建](#2-langgraph-工作流构建)
  - [三层记忆架构](#3-三层记忆架构)
  - [Scratchpad 机制](#4-scratchpad-机制章节级临时工作区)
  - [滚动上下文压缩](#5-滚动上下文压缩)
  - [工作流节点详解](#6-工作流节点详解)
  - [两阶段战略推演](#7-两阶段战略推演架构)
  - [战略模型注入系统](#8-战略模型注入系统)
  - [条件路由与修订反馈](#9-条件路由与修订反馈注入)
  - [独立评审与质量控制](#10-独立评审与质量控制)
  - [人工审核与 LLM 协同](#11-人工审核与-llm-协同)
- [配置说明](#配置说明)
- [部署与运行](#部署与运行)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 系统架构

系统由两个核心模块组成：

```
                          +-----------------------+
                          |    用户请求 / CLI      |
                          +-----------+-----------+
                                      |
                    +-----------------v--------------------+
                    |          Agent 模块 (LangGraph)       |
                    |                                      |
                    |  Coordinator -> Researcher -> Analyst|
                    |       -> Writer -> Human Review      |
                    |       -> Strategist -> Archiver      |
                    +-----------------+--------------------+
                                      |
                                      | RAG 检索
                    +-----------------v---------------------+
                    |           RAG 模块                     |
                    |                                       |
                    |  文档加载 -> 分块 -> 嵌入 -> Milvus      |
                    |                 检索 -> 重排序 -> 返回   |
                    +---------------------------------------+
```

- **RAG 模块**：负责文档的加载、分块、向量化存储和语义检索，为 Agent 提供知识库支撑。
- **Agent 模块**：基于 LangGraph 编排多智能体工作流，实现从规划、检索、分析、撰写到审核的完整报告生成流程。

---

## RAG 模块

RAG 模块实现了从原始文档到可检索向量的完整处理管线，支持多种文档格式和混合检索策略。

### 1. 文档加载与分块

**模块路径**: `rag_project/data_loader/`

| 组件 | 文件 | 职责 |
|------|------|------|
| 文档类型检测 | `document_type_detector.py` | 根据文件扩展名和内容特征识别文档类型 (TXT/PDF/DOCX)，返回对应的 LangChain Loader |
| 可配置分块器 | `configurable_splitter.py` | 基于 YAML 配置按文档类型执行差异化的分块策略 |
| 元数据提取 | `metadata_extractor.py` | 提取文档元信息（标题、发布日期、页码、标签等），支持 V2 增强标签构建 |
| 分块存储 | `chunk_storage.py` | 将分块结果持久化到 JSON 文件，支持断点续处理 |

**分块策略配置** (`config/chunking_config.yaml`):

| 文档类型 | 分块大小 | 重叠长度 | 特点 |
|----------|---------|----------|------|
| 新闻 (TXT) | 512 tokens | 50 | 短文本，快速检索 |
| PDF 文档 | 1000 tokens | 200 | 保留语义完整性，支持页面感知分块 |
| 规章制度 (DOCX) | 1000 tokens | 150 | 保留章节结构 |
| 默认 | 800 tokens | 100 | 通用策略 |

高级选项包括：语义分块 (semantic chunking)、表格保留 (preserve_tables) 以及表格独立分块 (table_as_separate_chunk)。

### 2. 向量化嵌入

**模块路径**: `rag_project/embeddings/`

系统支持两种嵌入模式，通过 `config/milvus_config.yaml` 中的 `embedding.mode` 字段切换：

| 模式 | 实现文件 | 说明 |
|------|---------|------|
| **本地模型** | `embedding_model.py` | 加载 BAAI/bge-m3 模型到本地 GPU/CPU，维度 1024，支持 GPU 加速和批量处理 |
| **API 服务** | `embedding_client.py` | 调用 SiliconFlow 等第三方 API，启动速度快，无需本地 GPU |

核心参数：

- 模型：BAAI/bge-m3 (1024 维)
- 批处理大小：32
- 最大长度：8192 tokens
- 支持 L2 归一化 (normalize_embeddings)

### 3. 向量存储与检索

**模块路径**: `rag_project/storage/`

| 管理器 | 文件 | 检索策略 | 对应管线 |
|--------|------|---------|---------|
| `MilvusManager` | `milvus_manager.py` | 纯稠密向量搜索 | `RAGPipeline` |
| `MilvusHybridManager` | `milvus_hybrid_manager.py` | 稠密向量 + BM25 文本搜索 | `HybridRAGPipeline` |

**Milvus 配置**:

- 版本：Milvus 2.6 (通过 Docker 部署)
- 索引类型：HNSW (M=16, efConstruction=256)
- 相似度度量：IP (内积)
- 检索参数：top_k=50, ef=128
- 元数据字段：id, vector, text, doc_type, source, publish_date, page_number, title

**混合检索** (Hybrid Retrieval) 支持两种融合策略：

- **RRF (Reciprocal Rank Fusion)**：基于排名倒数的融合，默认策略
- **加权融合 (Weighted)**：可配置稠密/稀疏权重比例 (默认 dense=0.7, sparse=0.3)

### 4. 重排序 (Reranking)

**模块路径**: `rag_project/reranker/`

| 组件 | 文件 | 说明 |
|------|------|------|
| 基类 | `reranker_base.py` | 定义重排序器接口 |
| SiliconFlow 重排序器 | `siliconflow_reranker.py` | 调用 SiliconFlow BGE-reranker API 对初筛结果进行精排 |

### 5. 管线入口

**标准管线** (`rag_project/pipeline.py` -- `RAGPipeline`):

```bash
# 索引文档
python rag_project/main.py index <文件或目录路径> --chunks-output data/chunks.json

# 搜索文档
python rag_project/main.py search "查询内容" --top-k 10 --doc-type pdf
```

**混合管线** (`rag_project/pipeline_hybrid.py` -- `HybridRAGPipeline`):

混合管线是标准管线的替代方案，使用 `MilvusHybridManager` 实现稠密向量与 BM25 文本搜索的联合检索，适用于需要兼顾语义相似和关键词匹配的场景。

---

## Agent 模块

Agent 模块是本系统的核心创新，基于 LangGraph 构建了一个面向企业战略规划的多智能体工作流。以下详细阐述其设计思路和关键机制。

### 1. 设计理念

本系统的 Agent 模块围绕以下核心设计原则构建：

**分离关注点 (Separation of Concerns)**

报告生成被拆解为四个专职节点：Researcher 负责信息检索，Analyst 负责结构化分析，Writer 负责文本撰写，Strategist 负责战略推演。每个节点只做一件事，通过 LangGraph 的状态图传递中间产物。这种设计使得每个环节可以独立优化和替换，且修订反馈可以精确路由到对应的节点。

**有限记忆与上下文窗口管理**

LLM 的上下文窗口是有限资源。系统通过三层记忆架构（详见下文）和滚动压缩机制，确保每个节点在调用 LLM 时只接收与其任务相关的上下文，避免信息过载和注意力稀释。

**两阶段战略推演**

系统不是逐章独立生成，而是先完成诊断阶段（第1-3章），再从中推导出战略蓝图，最后在蓝图的约束下完成推演阶段（第4-8章）。这使得后续章节与前期诊断结论保持逻辑一致性。

**Human-in-the-Loop with LLM Pre-Review**

每章完成后不是直接由人工判断质量，而是先用独立评审模型（不同厂商、不同模型）给出结构化评分和改进建议，人工在此基础上做最终决策。这降低了人工审核的认知负担，同时避免了单一模型的自我评估偏差。

### 2. LangGraph 工作流构建

**核心文件**: `rag_project/agent/graph.py`

工作流基于 LangGraph 的 `StateGraph` 构建，使用 `GraphState` (TypedDict) 作为全局状态 schema。图的构建过程如下：

```
(1) 定义 StateGraph(GraphState)

(2) 注册节点 (add_node):
    coordinator       <- lambda state: coordinator_node(state, coordinator_llm)
    prepare_chapter   <- lambda state: prepare_chapter_node(state, prep_llm)
    researcher        <- lambda state: researcher_node(state, retriever, researcher_llm)
    analyst           <- lambda state: analyst_node(state, analyst_llm)
    writer            <- lambda state: writer_node(state, writer_llm)
    strategist        <- lambda state: strategist_node(state, strategist_llm)
    human_review      <- human_review_node (纯状态处理，无 LLM)
    archiver          <- lambda state: archiver_node(state, archiver_llm)

(3) 设置入口点: set_entry_point("coordinator")

(4) 添加确定性边 (add_edge):
    coordinator     -> prepare_chapter
    prepare_chapter -> researcher
    researcher      -> analyst
    analyst         -> writer
    writer          -> human_review
    strategist      -> human_review
    archiver        -> END

(5) 添加条件边 (add_conditional_edges):
    human_review -> should_continue(state) -> {
        "prepare_chapter" : prepare_chapter  // 下一章
        "strategist"      : strategist        // 生成蓝图
        "researcher"      : researcher        // revise:data
        "analyst"         : analyst           // revise:logic
        "writer"          : writer            // revise:writing
        "end"             : archiver          // 归档
    }

(6) 编译:
    checkpointer = MemorySaver()
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]    // 暂停等待人工输入
    )
```

关键设计：

- **依赖注入**：每个节点的 LLM 实例和 Retriever 实例通过 lambda 闭包注入，节点函数本身保持纯粹的签名 `(state) -> state_update`，便于测试和替换。
- **MemorySaver**：使用 LangGraph 内置的内存检查点，支持工作流的暂停和恢复。配合 `interrupt_before=["human_review"]` 实现人工审核断点。
- **条件路由**：`should_continue` 函数根据 `review_decision`、`current_chapter_index`、`context_pool` 长度和 `strategic_blueprint` 状态进行多路分支决策。

### 3. 三层记忆架构

**核心文件**: `rag_project/agent/state.py`

GraphState 采用三层记忆架构管理整个工作流的状态。这一设计解决了一个核心问题：在多章节生成过程中，如何在 LLM 的有限上下文窗口内有效传递跨章节信息。

```
+================================================================+
|  第一层：外部知识库 (Milvus)                                      
|  - 仅 Researcher 节点可访问                                       
|  - 存储全部文档的向量表示                                          
|  - 通过语义检索按需获取，不占用 state 空间                          
+================================================================+
          |
          | 检索结果
          v
+================================================================+
|  第二层：全局状态 (GraphState 持久字段)                            
|                                                                  
|  context_pool : List[str]         <- 累加型，存储已通过章节全文      
|  context_summary : str            <- 滚动压缩摘要 (~500字)         
|  strategic_blueprint : Dict       <- 战略蓝图 (mission/SWOT/TOWS)  
|  global_plan : List[Dict]         <- 8章大纲及元数据                
+================================================================+
          |
          | prepare_chapter 提取子集
          v
+================================================================+
|  第三层：章节工作区 (chapter_scratchpad + 临时字段)                
|                                                                  
|  chapter_scratchpad : Dict        <- 本章结构化中间产物            
|  current_draft : str              <- Writer 生成的章节正文         
|  chapter_title / chapter_question <- 当前章节元信息               
|                                                                
|  【阅后即焚】：每章开始时由 prepare_chapter 清空重置                 
+================================================================+
```

**各层字段说明**：

| 层级 | 字段 | 类型 | 生命周期 | 说明 |
|------|------|------|---------|------|
| 全局规划 | `global_plan` | `List[Dict]` | 全局 | 8章大纲，每章含 title, phase, analysis_model, index |
| 全局规划 | `current_chapter_index` | `int` | 全局 | 当前章节索引，随审核通过递增 |
| 战略蓝图 | `strategic_blueprint` | `Optional[Dict]` | 第3章后 | 包含 mission, swot_analysis, tows_strategies, strategic_pillars, kpis, approved |
| 战略蓝图 | `current_phase` | `str` | 全局 | "diagnosis" 或 "initiatives" |
| 长期记忆 | `context_pool` | `Annotated[List[str], operator.add]` | 累加 | 已审核通过的章节原文，使用 operator.add 确保纯追加不覆盖 |
| 长期记忆 | `context_summary` | `str` | 滚动更新 | 前序章节的压缩摘要，由 prepare_chapter 滚动更新 |
| 短期工作区 | `chapter_scratchpad` | `Dict` | 每章重置 | 本章的中间产物（查询、文档、事实、洞察等） |
| 短期工作区 | `current_draft` | `str` | 每章重置 | Writer 生成的当前章节正文 |
| 控制 | `review_decision` | `str` | 每轮审核 | "approve" / "revise:data" / "revise:logic" / "revise:writing" |
| 控制 | `auto_revision_count` | `int` | 每章重置 | 自动修订次数，上限为1，防止死循环 |
| 控制 | `llm_review_result` | `Optional[Dict]` | 每轮审核 | LLM 预审结果（分数、问题、建议） |
| 传递 | `_pending_chapter_knowledge` | `Dict` | 跨章节 | 上一章的知识快照，供 prepare_chapter 压缩 |
| 输出 | `final_report` | `str` | 最终 | Archiver 组装的完整报告 |

**`context_pool` 的累加语义**：

`context_pool` 使用 `Annotated[List[str], operator.add]` 声明。这意味着每个节点返回 `{"context_pool": [new_item]}` 时，LangGraph 会自动执行追加而非覆盖。这一设计保证了：

- human_review 节点只需返回 `{"context_pool": [current_draft]}`，无需读取或复制已有列表
- 即使修订路由重新进入 analyst/writer，也不会因为状态更新丢失已通过的章节
- 修订时不可能意外覆盖之前的章节内容

### 4. Scratchpad 机制（章节级临时工作区）

`chapter_scratchpad` 是本系统的一个核心设计模式。它是一个字典类型的临时工作区，在单个章节的生成周期内承载各节点之间的结构化中间产物。

**生命周期**：

```
prepare_chapter          -- 清空 scratchpad = {}
       |
       v
researcher               -- 写入 scratchpad["queries"], scratchpad["retrieved_docs"]
       |
       v
analyst                  -- 读取 scratchpad["retrieved_docs"]
                         -- 写入 scratchpad["document_summary"]
                         -- 写入 scratchpad["key_facts"]      (可能是 list 或 dict)
                         -- 写入 scratchpad["insights"]
                         -- 写入 scratchpad["analysis_model_used"]
       |
       v
writer                   -- 读取 scratchpad["key_facts"], scratchpad["insights"]
                         -- 读取 scratchpad["document_summary"]
                         -- 输出到 current_draft (不写回 scratchpad)
       |
       v
human_review (approve)   -- 从 scratchpad 提取知识保存到 _pending_chapter_knowledge
                         -- 清空 scratchpad = {}
       |
       v
prepare_chapter (下一章)  -- 从 _pending_chapter_knowledge 压缩到 context_summary
                         -- 清空 scratchpad = {}
```

**Scratchpad 在各节点的数据流**：

```
chapter_scratchpad = {
    // --- 由 Researcher 写入 ---
    "queries": [                       // 生成的5-7个检索查询
        "2024年交通投资政策",
        "江西省交通基础设施建设规划",
        ...
    ],
    "retrieved_docs": [                // 去重后的检索文档 (top 20)
        {"text": "...", "score": 0.85, "metadata": {...}},
        ...
    ],

    // --- 由 Analyst 写入 ---
    "document_summary": "Document 1 [...]\nDocument 2 [...]",  // 文档摘要 (v2首尾提取)
    "key_facts": {                     // 结构化关键事实 (模型驱动)
        "Political": ["政策事实1", "政策事实2"],
        "Economic": ["经济事实1"],
        ...
    },
    "insights": ["洞察1", "洞察2"],   // 分析洞察
    "analysis_model_used": "PEST模型", // 记录使用的分析模型

    // --- 由 Human Review 写入 (修订时) ---
    "revision_feedback": {
        "decision": "revise:logic",
        "issues": ["SWOT维度不完整"],
        "score": 55,
        "dimension_scores": {...},
        "improvement_hints": {
            "researcher": "补充市场竞争数据",
            "analyst": "补全SWOT四个维度",
            "writer": "加强公文语态"
        },
        "previous_draft_summary": "..."  // 上一轮草稿摘要
    },

    // --- 推演阶段由 Prepare Chapter 注入 ---
    "strategic_blueprint": {...}       // 仅推演阶段存在
}
```

**设计要点**：

- **阅后即焚**：每章开始时 `prepare_chapter` 将 scratchpad 重置为 `{}`，确保章节间不会产生状态污染。上一章的知识在重置前被提取并压缩到 `context_summary` 中。
- **节点间解耦**：Researcher 不知道 Analyst 如何使用检索结果，Analyst 不知道 Writer 如何组织文本。它们只通过 scratchpad 的约定字段通信。
- **修订反馈注入**：当 human_review 决定修订时，修订反馈被注入 scratchpad 的 `revision_feedback` 字段。下游节点（researcher/analyst/writer）在执行时读取该字段，针对性地改进工作。

### 5. 滚动上下文压缩

随着章节推进，`context_pool` 中的已完成章节不断累加。如果将所有历史章节原文传递给后续章节的 Writer 节点，会超出 LLM 的上下文窗口。系统通过滚动压缩机制解决这一问题。

**压缩流程**：

```
章节 N 审核通过
       |
       v
human_review:
  scratchpad["key_facts"] + scratchpad["insights"]
  -> 保存到 _pending_chapter_knowledge
       |
       v
prepare_chapter (章节 N+1):
  1. 读取 _pending_chapter_knowledge
  2. 调用 _compress_chapter_knowledge():
     - 将 key_facts 按分类结构化（如 SWOT 的四个象限）
     - 将 insights 合并
     - 调用 LLM 将新旧摘要合并为 ~500 字的滚动摘要
     - 保留关键数据（金额、比例、增长率）
     - 保留政策名称和战略定位
     - LLM 失败时回退到简单拼接 + 尾部截断
  3. 更新 context_summary
  4. 调用 _detect_knowledge_gaps():
     - 使用 LLM 检测前序章节是否存在下一章需要的知识缺口
     - 如有缺口，将提示附加到 chapter_context
  5. 清空 scratchpad，开始新一章
```

**知识缺口检测**：

`prepare_chapter` 在设置新章节状态时，会调用 `_detect_knowledge_gaps()` 检测前序章节摘要是否已覆盖下一章所需的关键信息。如果存在缺口，返回补充检索方向的提示文本，附加到 `chapter_context` 中供 Researcher 参考。

### 6. 工作流节点详解

#### Coordinator -- 大纲生成

**文件**: `nodes/coordinator.py`

Coordinator 是工作流的入口节点。当前实现采用硬编码的 8 章结构，每章附带 `phase`（诊断/推演）和 `analysis_model`（战略分析模型）元数据。这种设计适用于省属国企政策导向型战略规划报告，确保大纲的结构性和可预测性。

返回的 `global_plan` 结构示例：

```python
[
    {"title": "第一章：宏观政策环境与时代要求", "phase": "diagnosis", "analysis_model": "PEST模型", "index": 0},
    {"title": "第三章：行业演进趋势与当前内部诊断", "phase": "diagnosis", "analysis_model": "波特五力模型与SWOT分析", "index": 2},
    {"title": "第五章：主责主业：高质量建设与保通保畅举措", "phase": "initiatives", "analysis_model": "BCG波士顿矩阵", "index": 4},
    ...
]
```

#### Prepare Chapter -- 章节状态初始化

**文件**: `nodes/prep_chapter.py`

Prepare Chapter 是每章循环的起点，承担三项关键职责：

1. **滚动压缩**：从 `_pending_chapter_knowledge` 中提取上一章的知识，通过 LLM 压缩合并到 `context_summary`
2. **知识缺口检测**：使用 LLM 判断前序章节是否覆盖下一章所需信息，生成补充检索提示
3. **状态隔离**：清空 `chapter_scratchpad` 和 `current_draft`，确保每章从干净状态开始

推演阶段的特殊处理：当 `phase == "initiatives"` 且战略蓝图已批准时，自动将蓝图注入 scratchpad，使后续节点可以在蓝图的约束下工作。

#### Researcher -- 自适应多查询检索

**文件**: `nodes/researcher.py`

Researcher 节点实现了自适应的多查询检索策略：

1. **初始查询生成**：基于 `chapter_question` 和 `chapter_context`，使用 LLM 生成 5-7 个多样化的检索查询
2. **多路检索**：对每个查询执行独立的 RAG 检索（top_k=20），扩大召回率
3. **去重**：基于 SHA-256 文本哈希去重，避免同一文档被重复召回
4. **充分性评估**：使用 LLM 评估检索结果是否足以支撑该章节的深入分析
5. **补充检索**：如果信息不足，自动生成 2 个补充查询，再次检索并去重
6. **排序截断**：按相似度分数排序，保留 top 20 文档

修订时的特殊处理：如果 scratchpad 中存在 `revision_feedback` 且 decision 为 `revise:data`，Researcher 会读取其中的 `improvement_hints["researcher"]`，针对性生成补充查询。

#### Analyst -- 战略模型驱动分析

**文件**: `nodes/analyst.py`

Analyst 是本系统区别于通用 RAG 系统的关键节点。它不是简单地总结检索结果，而是根据章节元数据中的 `analysis_model` 字段，强制 LLM 按照指定的战略分析框架进行结构化分析。

核心流程：

1. **文档摘要生成 (v2)**：对检索文档进行首尾提取（60% 首部 + 40% 尾部），根据总字符预算动态分配每文档的提取量
2. **模型注入**：根据 `analysis_model` 选择对应的 prompt 模板（PEST/SWOT/BCG/BSC/安索夫/7S/ESG 等），强制 LLM 按模型维度分类返回 `key_facts`
3. **质量自检**：对分析结果执行 `_validate_analysis()`，检查事实数量、模型维度完整性等。不合格时以更低温度重试一次
4. **结果写入 scratchpad**：`key_facts`（可能是 list 或 dict）、`insights`、`analysis_model_used`

#### Writer -- 约束撰写

**文件**: `nodes/writer.py`

Writer 在 Analyst 产出的结构化分析基础上，生成符合要求的章节正文。

核心设计：

1. **多源上下文注入**：Writer 的 prompt 同时包含 `key_facts`（分析产物）、`insights`（洞察）、`context_summary`（滚动摘要）、`document_summary`（文档原文摘要）和战略蓝图约束
2. **跨章节一致性约束**：当 `context_summary` 存在时，prompt 中会加入一致性指令，要求使用前序章节中相同的数据，避免矛盾
3. **引用后处理**：Writer 生成的草稿中的通用引用标记（"Document X"、"来源文档_X"）会被自动替换为实际文件名
4. **自校验与精简**：对生成的草稿执行 `_validate_draft()`，检查字数（1000-1800字）和引用数量。超标时调用 LLM 精简，移除改写说明等元信息

推演阶段特殊处理：当处于 initiatives 阶段时，prompt 中注入 `_build_blueprint_constraint()`，包含核心使命、战略支柱和 KPI，并强制要求章节显式说明如何支撑战略目标。

#### Strategist -- 战略蓝图生成

**文件**: `nodes/strategist.py`

Strategist 在诊断阶段（第1-3章）全部完成后被触发，生成战略蓝图。

核心流程：

1. **诊断阶段压缩**：调用 `_compress_diagnosis()` 将前3章内容压缩为 ~2000 字的结构化综合摘要，保留关键数据和 SWOT 相关内容
2. **SWOT 提取**：从压缩摘要中提取结构化的 SWOT 四象限数据
3. **TOWS 矩阵推导**：基于 SWOT 数据，通过 LLM 生成 SO/WO/ST/WT 四象限战略选项
4. **蓝图生成**：推导核心使命（20-30字）、3-5 个战略支柱、BSC 四维度的 KPI 体系
5. **标记未批准**：`strategic_blueprint["approved"] = False`，等待人工审核

#### Human Review -- 状态路由枢纽

**文件**: `nodes/human_review.py`

Human Review 是整个工作流的路由枢纽。它不调用任何 LLM，纯粹做状态处理和路由决策。

关键职责：

1. **章节批准**：将 `current_draft` 添加到 `context_pool`，从 scratchpad 提取知识保存到 `_pending_chapter_knowledge`，递增 `current_chapter_index`
2. **蓝图批准**：设置 `current_phase = "initiatives"`，标记 `strategic_blueprint["approved"] = True`
3. **修订路由**：不修改章节索引，将修订反馈（含 LLM 评审的问题、分数、维度评分、改进建议）注入 scratchpad 的 `revision_feedback` 字段
4. **防重复**：通过 `_last_approved_index` 防止同一章节被重复添加到 context_pool

路由逻辑（`should_continue` 函数）：

```python
if decision == "revise:data":     return "researcher"
if decision == "revise:logic":    return "analyst"
if decision == "revise:writing":  return "writer"
if decision == "revise_blueprint": return "strategist"
if decision == "approve_blueprint": return "prepare_chapter"
if decision == "approve":
    if 第3章完成 and 蓝图未生成:  return "strategist"
    if 还有下一章:                return "prepare_chapter"
    else:                         return "end"
```

#### Archiver -- 报告归档

**文件**: `nodes/archiver.py`

Archiver 是工作流的终点节点，负责将所有章节组装为最终报告。

1. **章节去重**：按标题去重 context_pool，保留每个章节的最新版本
2. **封面生成**：包含报告标题、生成时间和主题
3. **执行摘要**：调用 LLM 生成 ~1000 字的执行摘要，使用国企公文语态
4. **目录生成**：从各章节提取标题
5. **蓝图附录**：将 SWOT、TOWS、战略支柱、KPI 格式化为附录
6. **引用修复**：通过 Milvus 检索将残留的通用引用替换为实际文件名
7. **跨章一致性检查**：调用 LLM 检查跨章数据矛盾、内容重复和引用缺失

最终报告结构：封面 -> 执行摘要 -> 目录 -> 第1-8章 -> 战略蓝图附录 -> 一致性审查备注

### 7. 两阶段战略推演架构

系统采用"诊断-推演"两阶段架构，这是本系统区别于通用报告生成工具的核心设计。该架构确保战略报告不是各章节的简单拼接，而是从诊断到推演的逻辑递进。

```
阶段一：诊断 (Diagnosis)                     阶段二：推演 (Initiatives)
================================           ================================
第1章  PEST模型 (宏观政策环境)                 第4章  BSC平衡计分卡 (总体战略思路)
第2章  区域政策承接分析                        第5章  BCG波士顿矩阵 (主责主业)
第3章  波特五力 + SWOT (内部诊断)              第6章  安索夫矩阵 (创新驱动)
        |                                   第7章  ESG+产业链协同 (产业协同)
        v                                   第8章  麦肯锡7S (治理效能)
  +-------------------+
  |  SWOT提取         
  |  TOWS矩阵推导                            所有推演阶段章节受以下约束：
  |  使命/支柱/KPI                           -核心使命对齐
  |  战略蓝图生成                             -战略支柱支撑
  +-------------------+                     -KPI 体系一致
        |                                   -必须使用"为支撑...使命"等表述
        v
  +-------------------+
  |  人工审核蓝图     
  |  approve / revise 
  +-------------------+
```

**阶段转换的状态管理**：

第3章审核通过后，`human_review_node` 检测到 `current_chapter_index == 2` 且 `context_pool >= 3`，不递增索引，而是将 `review_decision` 设为 approve 并返回。`should_continue` 路由到 `strategist`，生成蓝图后再次进入 `human_review` 等待蓝图审核。蓝图通过后（`approve_blueprint`），索引递增到 3，`current_phase` 切换为 `"initiatives"`，系统进入推演阶段。

### 8. 战略模型注入系统

Analyst 和 Writer 节点都支持战略模型注入。系统预置了以下分析模型，每种模型定义了完整的分析维度和输出结构：

| 模型 | 分析维度 | 应用章节 | key_facts 结构 |
|------|---------|---------|---------------|
| PEST | Political, Economic, Social, Technological | 第1章 | `{"Political": [...], "Economic": [...], ...}` |
| 波特五力 | 现有竞争者, 潜在进入者, 替代品, 供应商, 买方 | 第3章 | `{"现有竞争者": [...], "潜在进入者": [...], ...}` |
| SWOT | Strengths, Weaknesses, Opportunities, Threats | 第3章 | `{"Strengths": [...], "Weaknesses": [...], ...}` |
| BSC | 财务, 客户/民生, 内部运营, 学习与成长 | 第4章 | `{"财务维度": [...], "客户/民生维度": [...], ...}` |
| BCG | 现金牛, 明星, 问题, 瘦狗 | 第5章 | `{"现金牛业务": [...], "明星业务": [...], ...}` |
| 安索夫 | 市场渗透, 市场开发, 产品开发, 多元化 | 第6章 | `{"市场渗透": [...], "市场开发": [...], ...}` |
| 麦肯锡7S | Strategy, Structure, Systems, Shared Values, Style, Staff, Skills | 第8章 | `{"Strategy": [...], "Structure": [...], ...}` |
| ESG+协同 | Environment, Social, Governance, 产业链协同 | 第7章 | `{"Environment": [...], "Social": [...], ...}` |

**注入方式**：Analyst 的 `_get_model_instruction()` 函数根据 `analysis_model` 字符串匹配对应的 prompt 模板，将分析维度的名称、定义和要求的 JSON 输出结构注入到 LLM prompt 中。这确保了 LLM 的输出是结构化的、按模型维度分类的 `key_facts`，而非自由文本总结。

**质量校验**：Analyst 的 `_validate_analysis()` 会检查模型维度的完整性。例如，SWOT 模型要求至少覆盖 4 个维度中的 3 个，否则触发重试。

### 9. 条件路由与修订反馈注入

**条件路由** (`should_continue` 函数) 是工作流的核心调度器，基于当前状态做出以下决策：

```
human_review 的 review_decision
       |
       +-- "revise:data"      -> researcher     (回到检索)
       +-- "revise:logic"     -> analyst        (回到分析)
       +-- "revise:writing"   -> writer         (回到撰写)
       +-- "revise_blueprint" -> strategist     (重新生成蓝图)
       +-- "approve_blueprint" -> prepare_chapter (进入推演阶段)
       +-- "approve"          -> 根据 context_pool 长度和蓝图状态决定:
       |     +-- 第3章完成 + 蓝图未生成 -> strategist
       |     +-- 蓝图已批准 + 有更多章节 -> prepare_chapter
       |     +-- 最后一章完成           -> archiver (END)
       +-- "finished"         -> archiver (END)
```

**修订反馈注入机制**：

当 human_review 节点处理修订决策时（`revise:data`/`revise:logic`/`revise:writing`），它会将 LLM 预审结果和人工反馈合并注入到 `chapter_scratchpad["revision_feedback"]` 中：

```python
chapter_scratchpad["revision_feedback"] = {
    "decision": "revise:logic",
    "comments": "用户手动输入的修改意见",          # 人工反馈
    "issues": ["SWOT维度不完整", "分析太浅"],       # LLM 评审发现的问题
    "score": 55,                                    # LLM 评审总分
    "dimension_scores": {"analysis_depth": 8, ...}, # LLM 维度评分
    "improvement_hints": {                          # LLM 针对各节点的改进建议
        "researcher": "补充市场竞争数据",
        "analyst": "补全SWOT四个维度的分析",
        "writer": "加强国企公文语态"
    },
    "previous_draft_summary": "上一轮草稿前800字"   # 供参考的上下文
}
```

下游节点在执行时读取 `revision_feedback`：

- **Researcher** 读取 `improvement_hints["researcher"]`，生成更有针对性的补充查询
- **Analyst** 将评审分数、维度评分、问题和改进方向全部注入 prompt，要求针对问题重新分析
- **Writer** 同理，读取 `improvement_hints["writer"]`，在修订 prompt 中展示上一轮草稿摘要和具体问题

用户的文字反馈会被合并到对应角色节点的 hint 中（前缀 `[用户指令]`），确保人工意见能精确传达。

**自动修订保护**：`auto_revision_count` 限制每章最多自动修订 1 次。即使 LLM 评审分数低于阈值，也不会无限循环修订。

### 10. 独立评审与质量控制

**核心文件**: `rag_project/agent/cli.py` (`_llm_evaluate_draft` 方法)

系统使用独立于主 LLM 的评审模型 (SiliconFlow Qwen3.5-122B) 进行质量评估，避免自我评估偏差。

**评审维度** (满分 100 分)：

| 维度 | 满分 | 评估内容 |
|------|------|---------|
| 主题契合度 (topic_relevance) | 15 | 内容是否紧扣章节标题和研究问题 |
| 分析深度 (analysis_depth) | 20 | 战略分析模型各维度的覆盖程度和洞察深度 |
| 写作专业度 (writing_quality) | 15 | 是否符合国企公文语态和报告规范 |
| 引用充分性 (citation_sufficiency) | 15 | 引用数量、是否为真实文件名、位置是否恰当 |
| 内容真实性 (groundedness) | 20 | 数据和政策名称是否可溯源，幻觉程度 |
| 上下文连贯性 (context_coherence) | 15 | 与前序章节的数据一致性，是否存在矛盾或重复 |

**阶段差异化评估**：

- **诊断阶段**：侧重信息覆盖度、分析模型完整性、数据准确性
- **推演阶段**：侧重蓝图对齐度（是否引用核心使命/战略支柱）、举措可执行性、与前序诊断结论的呼应

**评分规则**：

- 总分 >= 70 分 -> 建议 approve
- 总分 50-69 分 -> 根据主要问题类型建议 revise:data / revise:logic / revise:writing
- 总分 < 50 分 -> 建议 revise:writing
- 第一章无前序上下文，上下文连贯性维度自动满分

**针对性改进建议**：

评审结果中包含 `improvement_hints`，为 Researcher、Analyst、Writer 三个节点分别提供具体的改进方向（例如"补充市场竞争数据"或"SWOT 维度不完整"）。这些提示会通过修订反馈注入机制传递到对应节点。

### 11. 人工审核与 LLM 协同

**核心文件**: `rag_project/agent/cli.py`

系统提供两种审核模式，均集成了 LLM 预审：

**交互模式流程**：

```
Writer 生成草稿
       |
       v
LLM 预审 (SiliconFlow Qwen)
  -> 生成总分、维度评分表、问题列表、改进建议
       |
       v
CLI 展示:
  - 章节草稿全文
  - 质检报告 (总分 + 6维评分表 + 状态标识)
  - 问题摘要
  - 各节点改进建议
       |
       v
用户选择:
  1. Approve    -> 通过，进入下一章
  2. Revise     -> 手动指定修订类型 (data/logic/writing) + 修改指令
  3. LLM建议    -> 采纳 LLM 的评分驱动建议
  4. Skip       -> 跳过本章
       |
       v
可选: 编辑评审字段
  - 修改 suggestion (approve/revise:data/revise:logic/revise:writing)
  - 修改 issues 列表
  - 修改 researcher/analyst/writer 的改进建议
  - 支持多字段同时编辑
  - 确认前可回退重编辑
```

**自动模式逻辑**：

```
LLM 评分 >= 60 或已自动修订过 -> approve
LLM 评分 < 60 且未自动修订    -> 采纳 LLM suggestion，自动路由到对应修订节点
```

**蓝图审核**：

战略蓝图有独立的审核界面，展示核心使命、SWOT 分析、TOWS 策略、战略支柱和 KPIs，提供 Approve 和 Revise（附修改意见）两个选项。

**执行摘要质检**：

报告生成完成后，系统额外对执行摘要进行独立评审。不合格时使用评审反馈重新生成执行摘要，确保最终报告质量。

---

## 配置说明

所有配置通过 YAML 文件管理，位于 `config/` 目录：

| 配置文件 | 说明 |
|----------|------|
| `milvus_config.yaml` | Milvus 连接、集合、索引、搜索参数，以及嵌入模型配置 |
| `chunking_config.yaml` | 按文档类型配置分块策略 (大小、重叠、分隔符) |
| `agent_config.yaml` | LLM 提供商、评审模型、各 Agent 温度/Token/系统提示词、检索策略 |
| `reranker_config.yaml` | 重排序模型配置 |

**Agent 配置要点** (`config/agent_config.yaml`):

- 主 LLM：DeepSeek (deepseek-chat)，通过 `DEEPSEEK_API_KEY` 环境变量鉴权
- 评审模型：SiliconFlow Qwen3.5-122B，通过 `SILICONFLOW_API_KEY` 环境变量鉴权
- 检索策略：hybrid (稠密 + BM25)，使用 RRF 融合
- 各 Agent 可独立配置 temperature 和 max_tokens

---

## 部署与运行

### 环境要求

- Python 3.10+
- Docker (用于 Milvus 部署)
- NVIDIA GPU (推荐，用于本地嵌入模型加速)

### 安装步骤

**1. 启动 Milvus 向量数据库**

```bash
docker-compose up -d
```

等待约30秒，确认容器运行正常：

```bash
docker ps | grep milvus
```

**2. 安装依赖**

```bash
# RAG 核心依赖
pip install -r requirements.txt

# Agent 模块依赖
pip install -r requirements-agent.txt
```

**3. 配置 API Key**

```bash
# Windows CMD
set DEEPSEEK_API_KEY=sk-your-api-key
set SILICONFLOW_API_KEY=sk-your-api-key

# Linux / macOS
export DEEPSEEK_API_KEY=sk-your-api-key
export SILICONFLOW_API_KEY=sk-your-api-key
```

**4. 索引文档**

```bash
python rag_project/main.py index <文件或目录路径> --chunks-output data/chunks.json
```

**5. 生成报告**

```bash
# 交互模式 (推荐)
python scripts/run_agent_report.py "生成2026年江西交通投资集团战略规划报告"

# 自动模式 (快速测试)
python scripts/run_agent_report.py "生成2026年江西交通投资集团战略规划报告" --auto

# 自定义输出路径
python scripts/run_agent_report.py "生成年度总结报告" --output reports/2024_summary.md
```

**6. 文档搜索 (仅 RAG 管线)**

```bash
python rag_project/main.py search "查询内容" --top-k 10 --doc-type pdf
```

---

## 项目结构

```
RAG Project/
|
|-- config/                          # 配置文件
|   |-- milvus_config.yaml           #   Milvus 和嵌入模型配置
|   |-- chunking_config.yaml         #   分块策略配置
|   |-- agent_config.yaml            #   Agent 和 LLM 配置
|   |-- reranker_config.yaml         #   重排序模型配置
|
|-- rag_project/                     # 核心 Python 包
|   |-- main.py                      #   RAG 管线 CLI 入口
|   |-- pipeline.py                  #   标准 RAG 管线
|   |-- pipeline_hybrid.py           #   混合检索 RAG 管线
|   |
|   |-- data_loader/                 #   文档加载与分块
|   |   |-- document_type_detector.py
|   |   |-- configurable_splitter.py
|   |   |-- metadata_extractor.py
|   |   |-- chunk_storage.py
|   |
|   |-- embeddings/                  #   向量化嵌入
|   |   |-- embedding_model.py       #     本地 BGE-M3 模型
|   |   |-- embedding_client.py      #     API 嵌入服务客户端
|   |
|   |-- storage/                     #   向量存储
|   |   |-- milvus_manager.py        #     纯稠密检索管理器
|   |   |-- milvus_hybrid_manager.py #     混合检索管理器 (Dense+BM25)
|   |
|   |-- reranker/                    #   重排序模块
|   |   |-- reranker_base.py
|   |   |-- siliconflow_reranker.py
|   |
|   |-- agent/                       #   多智能体报告生成模块
|   |   |-- graph.py                 #     LangGraph 工作流构建器
|   |   |-- state.py                 #     全局状态定义 (GraphState)
|   |   |-- cli.py                   #     Agent CLI 交互界面
|   |   |-- llm_manager.py           #     LLM 调用管理器
|   |   |-- retriever.py             #     RAG 检索适配器
|   |   |-- nodes/                   #     工作流节点
|   |       |-- coordinator.py       #       大纲生成
|   |       |-- prep_chapter.py      #       章节准备
|   |       |-- researcher.py        #       多查询检索
|   |       |-- analyst.py           #       战略模型分析
|   |       |-- writer.py            #       章节撰写
|   |       |-- strategist.py        #       战略蓝图生成
|   |       |-- human_review.py      #       人工审核
|   |       |-- archiver.py          #       报告归档
|   |
|   |-- utils/                       #   工具模块
|       |-- logger.py
|
|-- scripts/                         # 运行脚本
|   |-- run_agent_report.py          #   报告生成启动脚本
|
|-- docs/                            # 文档
|
|-- docker-compose.yml               # Milvus Docker 部署配置
|-- requirements.txt                 # RAG 核心依赖
|-- requirements-agent.txt           # Agent 模块依赖
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 工作流编排 | LangGraph |
| LLM 框架 | LangChain |
| 主 LLM | DeepSeek (deepseek-chat) |
| 评审 LLM | SiliconFlow Qwen3.5-122B |
| 嵌入模型 | BAAI/bge-m3 (1024维) |
| 向量数据库 | Milvus 2.6 (HNSW 索引) |
| 重排序 | SiliconFlow BGE-reranker |
| 文档解析 | unstructured, pypdf, python-docx, pdfplumber |
| 中文处理 | jieba |
| 容器化 | Docker Compose (Milvus + etcd + MinIO) |
| 语言 | Python 3.10+ |
