# Agent 系统架构文档

## 目录
- [系统概述](#系统概述)
- [两阶段战略架构](#两阶段战略架构)
- [设计原则](#设计原则)
- [Agent 职责](#agent-职责)
- [引用系统设计](#引用系统设计)
- [战略分析模型体系](#战略分析模型体系)
- [路由逻辑](#路由逻辑)
- [技术栈](#技术栈)
- [扩展指南](#扩展指南)

---

## 系统概述

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户界面层                               │
│                   (CLI / Python API)                           │
│          python scripts/run_agent_report.py "请求" [-a] [-o]   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                LangGraph 工作流 (两阶段战略架构)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    StateGraph                            │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │              GraphState (全局状态)                  │  │  │
│  │  │  • user_input              (用户原始请求)           │  │  │
│  │  │  • global_plan             (8章结构化大纲,含元数据)  │  │  │
│  │  │  • strategic_blueprint     (战略蓝图:使命/TOWS/KPI) │  │  │
│  │  │  • current_phase           (diagnosis / initiatives)│  │  │
│  │  │  • context_pool            (已审核章节,只增不改)     │  │  │
│  │  │  • chapter_scratchpad      (当前章节工作区,阅后即焚) │  │  │
│  │  │  • current_draft           (当前章节草稿)           │  │  │
│  │  │  • review_decision         (审核决策)               │  │  │
│  │  │  • final_report            (最终完整报告)           │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  节点 (Agents):                                          │  │
│  │                                                          │  │
│  │  ┌────────────┐   ┌────────────┐   ┌─────────────┐      │  │
│  │  │ 协调器     │──▶│ 章节准备   │──▶│  研究员     │      │  │
│  │  │Coordinator │   │PrepChapter │   │ Researcher  │      │  │
│  │  │ (固定8章)  │   │(蓝图注入)  │   │(多查询检索) │      │  │
│  │  └────────────┘   └────────────┘   └─────────────┘      │  │
│  │                                             │            │  │
│  │                                             ▼            │  │
│  │  ┌────────────┐   ┌────────────┐   ┌─────────────┐      │  │
│  │  │  归档器    │◀──│ 人工审核   │◀──│  分析员     │      │  │
│  │  │ Archiver   │   │HumanReview │   │  Analyst    │      │  │
│  │  │(摘要+引用) │   │(章节+蓝图) │   │(模型注入)   │      │  │
│  │  └────────────┘   └──────┬─────┘   └─────────────┘      │  │
│  │                          │              ▲                 │  │
│  │  ┌────────────┐          │              │                 │  │
│  │  │ 战略规划师 │◀─────────┘              │                 │  │
│  │  │ Strategist │  (第3章后生成蓝图)      │                 │  │
│  │  │(SWOT→TOWS) │                         │                 │  │
│  │  └────────────┘                          │                 │  │
│  │                                   ┌─────────────┐         │  │
│  │                                   │   写作员    │─────────┘  │
│  │                                   │   Writer    │            │
│  │                                   │(引用后处理) │            │
│  │                                   └─────────────┘            │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬──────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   LLM 管理器    │  │  RAG 检索器     │  │  检查点管理     │
│ (DeepSeek API)  │  │   (Milvus)      │  │ (MemorySaver)   │
│ 6种Agent配置    │  │ BGE-M3嵌入      │  │ 断点恢复        │
│ 独立温度/Token  │  │ 多查询+去重     │  │                  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│  DeepSeek API   │  │  Milvus 向量库  │
│   (LLM服务)     │  │  (HNSW索引)     │
│                 │  │  3,422条向量     │
└─────────────────┘  └─────────────────┘
```

### 核心组件

#### 1. LangGraph 工作流引擎
- **文件:** `rag_project/agent/graph.py`
- **目的:** 编排多 Agent 工作流（两阶段战略架构）
- **核心功能:**
  - 通过 StateGraph 进行状态管理
  - 节点间的条件路由（含蓝图审核路由）
  - 人工干预的断点设置（`interrupt_before=["human_review"]`）
  - 基于检查点的持久化（MemorySaver）
  - 两阶段流程控制（诊断 → 蓝图 → 推演）

#### 2. 状态管理
- **文件:** `rag_project/agent/state.py`
- **目的:** 维护全局工作流状态
- **架构:** 三层记忆系统 + 战略记忆层
  - **长期记忆:** Milvus RAG（持久化知识库，3,422条文档chunks）
  - **工作记忆:** `chapter_scratchpad`（每章节独立沙盒，阅后即焚）
  - **已批准记忆:** `context_pool`（仅存入审核通过的定稿，只能累加）
  - **战略记忆:** `strategic_blueprint`（TOWS分析、使命、KPI，推演阶段的约束契约）

#### 3. Agent 节点
- **目录:** `rag_project/agent/nodes/`
- **8个专业化节点:**
  - `coordinator.py` - 固定8章大纲生成（含阶段元数据和分析模型）
  - `prep_chapter.py` - 章节状态初始化 + 推演阶段蓝图注入
  - `researcher.py` - 多查询语义检索 + 文档去重
  - `analyst.py` - 战略模型注入分析（PEST/SWOT/BCG等8种模型）
  - `writer.py` - 内容生成 + 引用后处理（Document X → 真实文件名）
  - `strategist.py` - 战略蓝图生成（SWOT → TOWS → 使命/支柱/KPI）
  - `human_review.py` - 双模式审核（章节审核 + 蓝图审核）+ 路由函数
  - `archiver.py` - 执行摘要生成 + 引用修复 + 报告汇编

#### 4. LLM 管理器
- **文件:** `rag_project/agent/llm_manager.py`
- **目的:** 管理 LLM API 交互，6种Agent独立配置
- **配置:**

| Agent | 温度 | 最大Token | 用途 |
|-------|------|-----------|------|
| coordinator | 0.3 | 2048 | 稳定生成大纲 |
| researcher | 0.1 | 4096 | 精确生成查询 |
| analyst | 0.5 | 3072 | 平衡分析 |
| writer | 0.7 | 4096 | 创意写作 |
| strategist | 0.5 | 4096 | 战略推导 |
| archiver | 0.5 | 3072 | 摘要生成 |

#### 5. RAG 检索器
- **文件:** `rag_project/agent/retriever.py`
- **目的:** 知识库查询，封装 RAGPipeline
- **功能:**
  - 基于 BGE-M3 嵌入的语义搜索（1024维）
  - 多查询并行检索（Researcher节点生成3-5个查询）
  - SHA256文本去重
  - Top-K结果返回（默认top_k=20）

---

## 两阶段战略架构

### 概述

Agent 系统实现了**两阶段战略推导架构**，专为省属国企战略规划设计：

1. **诊断阶段（第 1-3 章）:** 宏观环境 → 区域战略 → 内部诊断（SWOT）
2. **战略蓝图生成:** TOWS矩阵分析 → 使命制定 → 支柱/KPI设定
3. **推演阶段（第 4-8 章）:** 在蓝图约束下的战略举措

### 固定8章结构

Coordinator 不再使用 LLM 动态生成大纲，而是返回包含完整元数据的固定结构：

```python
global_plan = [
    # === 诊断阶段 (Diagnosis) ===
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
        "analysis_model": "波特五力模型与SWOT分析 (强制要求结构化SWOT矩阵)",
        "index": 2
    },
    # 第3章后 → 战略规划师生成蓝图 → 人工审核蓝图
    # === 推演阶段 (Initiatives) ===
    {
        "title": "第四章：总体战略思路与政策响应目标",
        "phase": "initiatives",
        "analysis_model": "平衡计分卡(BSC)模型 (财务/民生/运营/学习四维度)",
        "index": 3
    },
    {
        "title": "第五章：主责主业：高质量建设与保通保畅举措",
        "phase": "initiatives",
        "analysis_model": "BCG波士顿矩阵 (主业作为'现金牛'业务)",
        "index": 4
    },
    {
        "title": "第六章：创新驱动：绿色低碳与智慧交投建设",
        "phase": "initiatives",
        "analysis_model": "安索夫矩阵 (新产品/新市场拓展,第二增长曲线)",
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
        "analysis_model": "麦肯锡7S模型 (结构/制度/风格/员工/技能等)",
        "index": 7
    }
]
```

### 两阶段工作流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    两阶段工作流                                 │
└─────────────────────────────────────────────────────────────────┘

     第一阶段：诊断阶段                第二阶段：推演阶段
┌────────────────────────────┐    ┌────────────────────────────┐
│  第1章：PEST宏观环境分析    │    │  第4章：BSC战略目标        │
│  第2章：区域战略定位        │    │  第5章：BCG主责主业举措    │
│  第3章：波特五力+SWOT诊断   │    │  第6章：安索夫创新驱动     │
└───────────┬────────────────┘    │  第7章：ESG产业协同        │
            │                      │  第8章：7S治理效能         │
            ▼                      └─────────────┬──────────────┘
┌───────────────────────┐                      │
│   战略规划师节点       │                      │
│   - 从第3章提取SWOT    │                      │
│   - TOWS矩阵分析       │                      │
│   - 推导核心使命       │                      │
│   - 生成战略支柱       │                      │
│   - BSC维度KPI设定     │                      │
└───────────┬───────────┘                      │
            │                                   │
            ▼                                   │
┌───────────────────────┐                      │
│   蓝图审核 (人工批准)  │                      │
└───────────┬───────────┘                      │
            │                                   │
        是否批准?                               │
            │                                   │
      ┌─────┴─────┐                            │
      │           │                            │
     是           否                            │
      │           └────────────────┐           │
      ▼                  (重新生成)             │
   进入第4章                                   │
      │          (蓝图注入到每章上下文)         │
      └────────────────────────────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  归档器节点       │
         │  - 执行摘要生成   │
         │  - 引用修复       │
         │  - 完整报告汇编   │
         │  - 蓝图附录       │
         └──────────────────┘
```

### 状态字段

```python
class GraphState(TypedDict):
    # --- 输入层 ---
    user_input: str                     # 用户的原始请求

    # --- 全局规划层 ---
    global_plan: List[Dict]             # 8章结构化大纲 (含title, phase, analysis_model)
    current_chapter_index: int          # 当前执行的章节索引 (0-based)

    # --- 战略蓝图层（两阶段架构） ---
    strategic_blueprint: Optional[Dict] # 战略蓝图 (mission, swot_analysis, tows_strategies,
                                        #          strategic_pillars, kpis, approved)
    current_phase: str                  # "diagnosis" 或 "initiatives"

    # --- 上下文层 (长期/跨章节记忆) ---
    context_pool: Annotated[List[str], operator.add]  # 已审核通过的章节 (只能累加)
    context_summary: str                # 压缩后的全局上下文摘要

    # --- 当前章节层 (短期/工作区记忆) ---
    chapter_title: str                  # 当前章节名
    chapter_question: str               # 当前章节的研究问题
    chapter_context: str                # 当前章节的上下文
    chapter_scratchpad: Dict            # 本章结构化草稿本 (阅后即焚)
    current_draft: str                  # Writer生成的当前草稿

    # --- 控制层 ---
    human_feedback: Dict                # 人类结构化反馈
    review_decision: str                # "approve", "revise:data", "revise:logic",
                                        # "revise:writing", "approve_blueprint",
                                        # "revise_blueprint", "finished"

    # --- 输出层 ---
    final_report: str                   # Archiver生成的最终完整报告
```

---

## 设计原则

### 1. 四层状态隔离

```
第一层：长期记忆（Milvus RAG）
    ↓ 仅 Researcher 可访问
    存储内容：3,422个文档chunks
    特点：永久保存，所有章节共享

第二层：短期工作区（chapter_scratchpad）
    ↓ 当前章节专用，阅后即焚
    存储内容：queries, retrieved_docs, key_facts, insights,
              document_summary, analysis_model_used
    特点：每章结束后物理清空为 {}

第三层：神圣上下文池（context_pool）
    ↓ 累积存储，只能添加
    存储内容：人工审核通过的章节完整内容
    特点：使用 operator.add 注解，确保纯累加不覆盖

第四层：战略记忆（strategic_blueprint）
    ↓ 推演阶段的约束契约
    存储内容：使命、SWOT、TOWS策略、战略支柱、KPI
    特点：第3章后生成，人工批准后约束所有推演章节
```

**各节点访问权限：**

| 节点 | Milvus RAG | Scratchpad | Context Pool | Blueprint |
|------|:----------:|:----------:|:------------:|:---------:|
| Researcher | 读+写 | 写 | - | - |
| Analyst | - | 读+写 | - | - |
| Writer | - | 读 | - | 读 |
| Strategist | - | - | 读(前3章) | 写 |
| Archiver | - | - | 读(全部) | 读 |

### 2. 逐章生成 + 蓝图检查点

```python
# 诊断阶段（第1-3章）
for chapter_index in [0, 1, 2]:
    prepare_chapter → researcher → analyst → writer → human_review

# 战略蓝图生成（第3章后自动触发）
strategist_node(context_pool[0:3])  → strategic_blueprint

# 蓝图审核
if human_approve(blueprint):
    blueprint["approved"] = True
else:
    strategist_node(context_pool, feedback)  # 重新生成

# 推演阶段（第4-8章，带蓝图约束）
for chapter_index in [3, 4, 5, 6, 7]:
    prepare_chapter(注入蓝图) → researcher → analyst → writer → human_review

# 最终报告
archiver_node(context_pool, blueprint) → final_report
```

### 3. 人机协同（HITL）

#### 中断机制
```python
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"]  # 在此处暂停
)
```

#### 双模式审核

**章节审核（每章后）：**
| 决策 | 路由目标 | 说明 |
|------|----------|------|
| `approve` | 下一章/strategist/archiver | 通过，进入下一环节 |
| `revise:data` | researcher | 数据不足，重新检索 |
| `revise:logic` | analyst | 逻辑问题，重新分析 |
| `revise:writing` | writer | 文笔问题，重新撰写 |

**蓝图审核（第3章后）：**
| 决策 | 路由目标 | 说明 |
|------|----------|------|
| `approve_blueprint` | prepare_chapter | 蓝图通过，进入推演阶段 |
| `revise_blueprint` | strategist | 重新生成蓝图 |

---

## Agent 职责

### 1. Coordinator（协调器）

**文件:** `rag_project/agent/nodes/coordinator.py`

**目的:** 生成固定的8章结构化大纲

**实现方式:** 不调用LLM，直接返回预定义的8章结构

**输出:**
```python
{
    "global_plan": [
        {"title": "第一章：...", "phase": "diagnosis", "analysis_model": "PEST模型...", "index": 0},
        # ... 8章
    ],
    "current_chapter_index": 0,
    "current_phase": "diagnosis"
}
```

**为什么不用LLM生成？**
- 省属国企战略规划有标准格式要求
- 避免LLM生成不一致的大纲结构
- 每章的分析模型需精确匹配战略框架

---

### 2. Prepare Chapter（章节准备）

**文件:** `rag_project/agent/nodes/prep_chapter.py`

**目的:** 初始化章节状态 + 推演阶段蓝图注入

**核心操作:**
1. 从 `global_plan[index]` 提取章节标题和元数据
2. 清空 `chapter_scratchpad = {}`（状态隔离）
3. 清空 `current_draft = ""`（确保干净起点）
4. 生成 `chapter_question`（从标题转换）
5. **推演阶段特殊处理:** 如果 `phase == "initiatives"` 且蓝图已批准，将蓝图注入到 `chapter_scratchpad`

**蓝图注入逻辑:**
```python
if is_initiatives_phase and blueprint and blueprint.get("approved"):
    chapter_scratchpad["strategic_blueprint"] = strategic_blueprint
```

---

### 3. Researcher（研究员）

**文件:** `rag_project/agent/nodes/researcher.py`

**目的:** 多查询语义检索

**工作流程:**
```
chapter_question → LLM生成3-5个查询 → Milvus检索每查询top_k=20
→ SHA256文本去重 → 按score降序 → 保留top 20
```

**状态读写:**
- 读: `chapter_question`, `chapter_context`, `chapter_scratchpad`
- 写: `chapter_scratchpad["queries"]`, `chapter_scratchpad["retrieved_docs"]`

---

### 4. Analyst（分析员）

**文件:** `rag_project/agent/nodes/analyst.py`

**目的:** 战略模型注入分析，提取关键事实和洞察

**核心创新:** 根据每章的 `analysis_model` 字段动态注入对应的战略分析框架

**工作流程:**
```
retrieved_docs → 文档摘要(限10篇,含来源标注) → LLM分析(注入战略模型)
→ 结构化key_facts + insights
```

**支持的8种战略分析模型:**

| 模型 | 适用章节 | key_facts结构 |
|------|----------|---------------|
| PEST模型 | 第1章 | `{"Political": [], "Economic": [], "Social": [], "Technological": []}` |
| 通用分析 | 第2章 | `["事实1", "事实2", ...]` |
| 波特五力+SWOT | 第3章 | `{"现有竞争者": [], "潜在进入者": [], ...}` |
| 平衡计分卡(BSC) | 第4章 | `{"财务维度": [], "客户/民生维度": [], ...}` |
| BCG波士顿矩阵 | 第5章 | `{"现金牛业务": [], "明星业务": [], ...}` |
| 安索夫矩阵 | 第6章 | `{"市场渗透": [], "市场开发": [], ...}` |
| ESG+产业链 | 第7章 | `{"Environment": [], "Social": [], ...}` |
| 麦肯锡7S | 第8章 | `{"Strategy": [], "Structure": [], ...}` |

**文档摘要格式（含来源标注）:**
```
Document 1 [来源: 中国交通年鉴2021_merged, 第15页]:
文本内容预览...

Document 2 [来源: 省交通运输厅年度工作报告]:
文本内容预览...
```

**状态读写:**
- 读: `chapter_question`, `chapter_context`, `chapter_scratchpad`, `global_plan[index]`
- 写: `chapter_scratchpad["document_summary"]`, `["key_facts"]`, `["insights"]`, `["analysis_model_used"]`

---

### 5. Writer（写作员）

**文件:** `rag_project/agent/nodes/writer.py`

**目的:** 章节内容生成 + 引用后处理

**工作流程:**
```
key_facts + insights + document_summary + analysis_model
→ LLM写作(注入模型写作指令+蓝图约束)
→ 引用后处理 (Document X → 真实文件名)
→ current_draft
```

**引用后处理机制（关键）:**

Writer 在生成草稿后执行自动引用替换：
```python
# Step 1: 从 document_summary 提取映射
mapping = {"Document 1": "中国交通年鉴2021_merged", "Document 2": "省交通运输厅报告", ...}

# Step 2: 替换草稿中的通用引用
"[来源: Document 1, 第15页]" → "[来源: 中国交通年鉴2021_merged, 第15页]"
"[来源: 来源文档_3]" → "[来源: 真实文件名]"
```

**推演阶段增强:**
- Writer 会接收 `strategic_blueprint` 约束
- 写作提示词要求章节内容必须支撑战略支柱和KPI

**状态读写:**
- 读: `chapter_title`, `chapter_question`, `chapter_context`, `chapter_scratchpad`, `global_plan[index]`, `strategic_blueprint`
- 写: `current_draft`

---

### 6. Strategist（战略规划师）

**文件:** `rag_project/agent/nodes/strategist.py`

**目的:** 从诊断阶段生成战略蓝图

**触发条件:** 第3章（index=2）批准后自动触发

**工作流程:**
```
context_pool[0:3] → 从第3章提取SWOT → TOWS矩阵分析
→ 推导使命/支柱/KPI → strategic_blueprint (approved=False)
```

**蓝图结构:**
```python
{
    "mission": "服务交通强省战略，打造一流综合交通投资运营集团",
    "swot_analysis": {
        "Strengths": ["省属国企平台优势", ...],
        "Weaknesses": ["创新能力有待提升", ...],
        "Opportunities": ["交通强省战略机遇", ...],
        "Threats": ["经济下行压力", ...]
    },
    "tows_strategies": {
        "SO": ["利用省属平台优势，抢抓交通强省战略机遇", ...],
        "WO": ["通过战略合作弥补创新短板", ...],
        "ST": ["强化风险防控应对经济下行", ...],
        "WT": ["深化改革提升组织韧性", ...]
    },
    "strategic_pillars": [
        "战略支柱1：主业提质 - 夯实交通投资建设主阵地",
        "战略支柱2：创新驱动 - 培育智慧绿色交通新动能",
        "战略支柱3：产业协同 - 构建交旅融合新生态",
        "战略支柱4：治理提升 - 完善现代企业制度"
    ],
    "kpis": {
        "财务维度": {"营收增长率": "年增长8%", ...},
        "客户/民生维度": {"公众满意度": "提升至90分", ...},
        "运营维度": {"项目按期完工率": "达到95%", ...},
        "学习成长维度": {"员工培训覆盖率": "100%", ...}
    },
    "approved": False
}
```

**状态读写:**
- 读: `context_pool`（前3章）, `user_input`
- 写: `strategic_blueprint`, `current_draft`（清空为""）

---

### 7. Human Review（人工审核）

**文件:** `rag_project/agent/nodes/human_review.py`

**目的:** 双模式审核（章节审核 + 蓝图审核）+ 路由函数

**章节审核逻辑:**
```python
if decision == "approve":
    # 强制使用 global_plan 中的标题（替换LLM可能生成的错误标题）
    full_chapter = f"# {correct_title_from_plan}\n\n{draft_content}"
    context_pool.append(full_chapter)  # 通过 operator.add 累加
    chapter_scratchpad = {}  # 阅后即焚

    if current_index == 2:
        # 第3章完成，不递增index，等待蓝图审核
        # 路由到 strategist
    else:
        current_chapter_index += 1  # 进入下一章
```

**蓝图审核逻辑:**
```python
if decision == "approve_blueprint":
    strategic_blueprint["approved"] = True
    current_phase = "initiatives"
    current_chapter_index += 1  # 从第3章 → 第4章

elif decision == "revise_blueprint":
    # 保持状态不变，路由回 strategist 重新生成
```

**标题保护机制:** Human Review 节点会**强制替换** LLM 生成的章节标题为 `global_plan` 中的正确标题，防止章节编号错误（如第3章生成"第二章"标题的问题）。

**should_continue 路由函数:**
```python
def should_continue(state) -> str:
    decision = state.get("review_decision")

    # 修订路由
    if decision == "revise:data":    return "researcher"
    if decision == "revise:logic":   return "analyst"
    if decision == "revise:writing": return "writer"
    if decision == "revise_blueprint": return "strategist"
    if decision == "finished":       return "end"

    # 蓝图批准 → 进入推演阶段
    if decision == "approve_blueprint": return "prepare_chapter"

    # 章节批准
    if decision == "approve":
        if current_index == 2:
            if not blueprint_approved:
                return "strategist"    # 第3章后生成蓝图
            else:
                return "prepare_chapter"
        elif has_more_chapters:
            return "prepare_chapter"
        else:
            return "end"  # → archiver

    return "end"
```

---

### 8. Archiver（归档器）

**文件:** `rag_project/agent/nodes/archiver.py`

**目的:** 最终报告汇编，含执行摘要、引用修复、蓝图附录

**工作流程:**
```
context_pool(8章) + strategic_blueprint
→ 封面 → 执行摘要(LLM生成,1000字) → 目录 → 8章内容
→ 蓝图附录 → 引用修复(来源文档_X → 真实文件名)
→ final_report
```

**报告结构:**
```markdown
# 江西交通投资集团战略规划报告
**生成时间**: YYYY年MM月DD日  |  **主题**: 用户请求

---

# 执行摘要
[LLM生成的1000字国企公文语态摘要]

---

## 目录
1. 第一章：...
2. 第二章：...
...

---

[第一章完整内容]
---
[第二章完整内容]
...
---
[第八章完整内容]

---

# 附录：战略蓝图详述
## 核心使命 / SWOT矩阵 / TOWS战略组合 / 战略支柱 / KPIs
```

**引用修复机制:**

Archiver 在最终报告中执行二级引用修复：
```python
# 搜索报告中的 [来源: 来源文档_X] 引用
# 提取引用附近的文本作为查询
# 通过 Milvus 检索找到真实来源文件名
# 替换: [来源: 来源文档_3] → [来源: 江西省交通工作会议纪要]
```

---

## 引用系统设计

### 三级引用保障机制

```
第一级：Analyst节点（来源标注）
  ↓ 分析文档时生成格式: "Document X [来源: 真实文件名, 第Y页]"
  ↓ 使用 fallback chain: source → title → doc_type → "来源文档_X"

第二级：Writer节点（引用后处理）
  ↓ 从 document_summary 提取 Document→文件名 映射
  ↓ 正则替换: [来源: Document X] → [来源: 真实文件名]
  ↓ 过滤掉通用文件名（来源文档_X）不进行替换

第三级：Archiver节点（最终修复）
  ↓ 搜索残余的 [来源: 来源文档_X] 引用
  ↓ 使用引用附近文本查询 Milvus 获取真实来源
  ↓ 正则替换所有残余通用引用
```

### 引用格式

最终报告中的引用格式：
- 有页码: `[来源: 文件名, 第X页]`
- 无页码: `[来源: 文件名]`

### 引用质量验证

```bash
# 运行报告生成后分析引用质量
python -c "
import re
report = open('output/report.md', 'r', encoding='utf-8').read()
all_citations = re.findall(r'\[来源:\s*[^\]]+\]', report)
generic = re.findall(r'\[来源:\s*来源文档_\d+[^\]]*\]', report)
real = len(all_citations) - len(generic)
print(f'总引用: {len(all_citations)}, 真实文件名: {real}, 通用引用: {len(generic)}')
print(f'质量率: {100*real/max(len(all_citations),1):.1f}%')
"
```

---

## 战略分析模型体系

### 模型注入流程

```
Coordinator → global_plan[index]["analysis_model"] = "PEST模型"
                                    ↓
Prepare Chapter → 传递给 scratchpad
                                    ↓
Analyst → _get_model_instruction("PEST模型") → 生成模型特定的分析指令
        → LLM按PEST框架返回结构化 key_facts
                                    ↓
Writer → _get_model_writing_instruction("PEST模型") → 生成模型特定的写作指令
       → 章节内容体现PEST四维度结构
```

### 模型指令示例

**PEST模型（第1章）:**
- 分析指令: 按Political/Economic/Social/Technological四维度组织
- 写作指令: 使用小标题明确区分政策环境/经济影响/社会因素/技术发展

**BCG波士顿矩阵（第5章）:**
- 分析指令: 按现金牛/明星/问题/瘦狗业务分类
- 写作指令: 识别主业为现金牛业务，强调稳定回报

**麦肯锡7S（第8章）:**
- 分析指令: 按Strategy/Structure/Systems/Shared Values/Style/Staff/Skills七维度
- 写作指令: 强调各要素间的协调性，构建组织保障体系

---

## 路由逻辑

### 完整路由图

```
                    human_review
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      revise:data    revise:logic   revise:writing
           │             │             │
           ▼             ▼             ▼
      researcher     analyst        writer
           │             │             │
           └─────────────┴─────────────┘
                         │
                    (回到 human_review)
                         │
                    approve
                         │
              ┌──────────┴──────────┐
              │                     │
         index=2               index≠2
              │                     │
              ▼                     ▼
         蓝图未批准?            has_more?
         ┌───┴───┐           ┌───┴───┐
        是      否           是      否
         │       │           │       │
         ▼       ▼           ▼       ▼
    strategist  prepare    prepare  archiver
    (生成蓝图)  _chapter   _chapter  (最终报告)
              │           │
              ▼           ▼
         human_review  human_review
         (蓝图审核)    (下一章)

    approve_blueprint ──→ prepare_chapter (进入推演)
    revise_blueprint  ──→ strategist (重新生成)
```

---

## 技术栈

| 组件 | 技术 | 版本 | 目的 |
|------|------|------|------|
| **工作流引擎** | LangGraph | Latest | 多Agent编排、状态管理、断点控制 |
| **状态管理** | StateGraph | Latest | 全局状态、条件路由、检查点恢复 |
| **LLM** | DeepSeek API | - | 语言模型（6种Agent配置） |
| **LLM客户端** | OpenAI SDK | 1.x | API接口封装 |
| **向量数据库** | Milvus | 2.3+ | 语义搜索、HNSW索引 |
| **嵌入模型** | BGE-M3 (SiliconFlow API) | - | 1024维文本向量化 |
| **配置管理** | PyYAML | 6.x | Agent和Milvus配置 |
| **日志记录** | Python logging | Stdlib | 结构化调试日志 |
| **环境变量** | python-dotenv | 1.x | API Key管理 |

---

## 扩展指南

### 添加新的战略分析模型

1. 在 `analyst.py` 的 `_get_model_instruction()` 中添加模型指令
2. 在 `writer.py` 的 `_get_model_writing_instruction()` 中添加写作指令
3. 在 `coordinator.py` 的 `global_plan` 中更新对应章节的 `analysis_model`

### 添加新的章节

1. 在 `coordinator.py` 中扩展 `global_plan` 列表
2. 确保每章的 `phase` 和 `analysis_model` 正确设置
3. 更新路由逻辑中 Chapter 3 的特殊处理（如需在诊断阶段末尾触发蓝图）

### 修改报告结构

1. 封面/目录: 修改 `archiver.py` 的 `_create_cover()` / `_create_table_of_contents()`
2. 执行摘要: 修改 `_generate_executive_summary()` 的提示词
3. 蓝图附录: 修改 `_create_blueprint_appendix()` 的格式

---

## 版本历史

- **v3.0** (2026-04-04): 战略分析模型体系 + 引用系统
  - Analyst节点支持8种战略分析模型注入（PEST/SWOT/BCG/波特五力/BSC/安索夫/7S/ESG）
  - Writer节点实现引用后处理（Document X → 真实文件名）
  - Archiver节点实现二级引用修复（Milvus查询补全）
  - Coordinator改为固定8章结构，不再使用LLM生成大纲
  - 引用质量达100%（测试验证80/80引用均为真实文件名）

- **v2.0** (2026-03-31): 两阶段战略架构
  - 添加战略规划师Agent用于蓝图生成
  - 添加 strategic_blueprint 到状态
  - 添加 current_phase 跟踪
  - 增强协调器以支持阶段元数据
  - 增强准备章节以支持蓝图注入
  - 增强归档器以支持执行摘要
  - 更新路由逻辑以支持蓝图审核

- **v1.0** (2026-03-28): 初始Agent系统
  - 基于LangGraph的多Agent工作流
  - HITL支持
  - 三层记忆架构
  - 7个专业化Agent