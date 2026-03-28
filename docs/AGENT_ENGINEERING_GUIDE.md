# Agent系统架构与逻辑详解

## 📚 目录
1. [系统概述](#系统概述)
2. [核心架构](#核心架构)
3. [工作流程](#工作流程)
4. [组件详解](#组件详解)
5. [数据流](#数据流)
6. [运行指南](#运行指南)

---

## 系统概述

### 什么是Agent系统？

这是一个基于**LangGraph**的多智能体协作系统，用于自动生成江西交通投资集团董事会的战略规划报告。

### 核心思想

模拟顶级咨询公司（如麦肯锡）的报告生成流程：
```
理解需求 → 制定大纲 → 分章调研 → 逐章撰写 → 人工审核 → 汇总成册
```

### 四大核心机制

1. **章节化生成** - 一次只写一章，避免长文本混乱
2. **状态隔离** - 每章独立工作空间，互不干扰
3. **人工介入** - 关键节点暂停，由人工审核把关
4. **结构化分析** - 检索→分析→写作，环环相扣

---

## 核心架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户请求                                │
│            "生成2024年江西交通投资政策解读报告"              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph工作流引擎                         │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Coordinator  │───▶│Prepare_Chapter│───▶│  Researcher  │ │
│  │  协调智能体   │    │   章节准备    │    │  检索智能体   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                        ↓   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Archiver    │◀───│Human Review  │◀───│   Analyst    │ │
│  │  归档节点    │    │  人工审核     │    │  分析智能体   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                      ↑                     │
│                              ┌──────────────┐              │
│                              │   Writer     │──────────────┘
│                              │  写作智能体   │
│                              └──────────────┘
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    最终报告(Markdown)                        │
│            2024年江西交通投资政策解读报告.md                  │
└─────────────────────────────────────────────────────────────┘
```

### 三层记忆架构

这是系统的核心设计，确保数据不会混乱：

```python
# 第一层：长期记忆（知识库）
Milvus向量数据库
    ↓ 仅Researcher可访问
    存储内容：3,426个文档chunks
    特点：永久保存，所有章节共享

# 第二层：短期工作区（章节草稿本）
chapter_scratchpad = {
    "queries": ["查询1", "查询2", ...],
    "retrieved_docs": [...],
    "key_facts": ["事实1", "事实2", ...],
    "insights": ["洞察1", "洞察2", ...]
}
    ↓ 当前章节专用
    存储内容：检索结果、分析数据
    特点：⚠️ 阅后即焚，每章结束后清空为 {}

# 第三层：神圣上下文池（已通过章节）
context_pool = [
    "# 第一章 行业概述\n\n内容...",
    "# 第二章 政策环境\n\n内容...",
    ...
]
    ↓ 累积存储
    存储内容：人工审核通过的章节
    特点：只能添加，不能修改，用于最终报告
```

---

## 工作流程

### 完整执行流程

```
第1步：协调规划
├─ 输入：用户请求 "生成2024年交通投资政策解读报告"
├─ 处理：Coordinator调用LLM生成报告大纲
└─ 输出：["行业概述", "政策环境", "现状分析", "问题挑战", "战略建议"]

第2步：准备第一章
├─ 操作：Prepare Chapter初始化状态
├─ 关键：清空chapter_scratchpad = {}
└─ 设置：chapter_title = "行业概述"

第3步：检索研究
├─ 操作：Researcher生成3-5个查询词
├─ 查询示例：["2024交通投资", "江西交通政策", "交通基础设施"]
├─ 检索：从Milvus获取相关文档
└─ 输出：retrieved_docs（top 20，去重）

第4步：分析洞察
├─ 操作：Analyst分析检索到的文档
├─ 提取：key_facts（5-10个关键事实）
├─ 生成：insights（3-5个商业洞察）
└─ 输出：更新chapter_scratchpad

第5步：撰写草稿
├─ 操作：Writer根据facts和insights生成章节
├─ 约束：不能直接读取retrieved_docs
├─ 生成：800-1200字的专业报告
└─ 输出：current_draft

第6步：人工审核
├─ 显示：打印current_draft供审核
├─ 选项：
│   1. approve → 加入context_pool，进入下一章
│   2. revise:data → 返回Researcher重新检索
│   3. revise:logic → 返回Analyst重新分析
│   4. revise:writing → 返回Writer重新撰写
└─ 决策：用户选择操作

第7步：循环迭代
├─ 如果approve且还有章节 → 返回第2步
├─ 如果approve且最后一章 → 进入第8步
└─ 如果revise → 返回对应节点重做

第8步：归档完成
├─ 操作：Archiver合并所有章节
├─ 添加：封面（标题、时间、主题）
├─ 添加：页脚（章节目录）
└─ 输出：final_report.md
```

### 状态流转示例

```python
# 初始状态
state = {
    "user_input": "生成2024年交通投资政策解读报告",
    "global_plan": [],
    "current_chapter_index": 0,
    "context_pool": [],
    "chapter_scratchpad": {},
    "current_draft": ""
}

# Coordinator处理后
state["global_plan"] = ["行业概述", "政策环境", "现状分析"]

# Prepare Chapter后（第一章）
state["chapter_title"] = "行业概述"
state["chapter_scratchpad"] = {}  # ⚠️ 必须清空

# Researcher处理后
state["chapter_scratchpad"] = {
    "queries": ["2024交通投资", "江西交通政策"],
    "retrieved_docs": [doc1, doc2, doc3, ...]  # 20个文档
}

# Analyst处理后
state["chapter_scratchpad"] = {
    "queries": [...],
    "retrieved_docs": [...],
    "key_facts": ["2024年投资增长15%", "政策支持加强", ...],
    "insights": ["投资反映行业活力", "政策红利释放", ...]
}

# Writer处理后
state["current_draft"] = "# 行业概述\n\n2024年以来，交通..."

# Human Review（approve）后
state["context_pool"] = ["# 行业概述\n\n2024年以来，交通..."]
state["chapter_scratchpad"] = {}  # ⚠️ 再次清空
state["current_chapter_index"] = 1
state["current_draft"] = ""

# ... 重复以上流程处理第二章、第三章 ...

# 所有章节完成后
state["final_report"] = "# 江西交通投资集团战略规划报告\n\n## 一、行业概述\n\n..."
```

---

## 组件详解

### 1. GraphState（状态管理器）

**作用**：定义整个工作流的数据结构

**位置**：`rag_project/agent/state.py`

**关键字段**：
```python
class GraphState(TypedDict):
    # 输入层
    user_input: str                    # 用户的原始请求

    # 全局规划层
    global_plan: List[str]             # 章节大纲
    current_chapter_index: int         # 当前第几章

    # 上下文层（长期记忆）
    context_pool: List[str]            # 已通过的章节 ⚠️ 只能累加
    context_summary: str               # 前文摘要（给Writer参考）

    # 当前章节层（短期工作区）
    chapter_title: str                 # 当前章节名
    chapter_scratchpad: Dict           # 本章草稿本 ⚠️ 阅后即焚
    current_draft: str                 # 当前生成的草稿

    # 控制层
    human_feedback: Dict               # 人类的反馈
```

**重要特性**：
- `context_pool`使用`operator.add`注解，确保只能累加不能覆盖
- `chapter_scratchpad`每章结束后必须物理清空为`{}`

---

### 2. LLM Manager（大模型管理器）

**作用**：统一管理所有Agent对DeepSeek API的调用

**位置**：`rag_project/agent/llm_manager.py`

**核心功能**：
```python
class LLMManager:
    def invoke(self, prompt: str, agent_type: str) -> str:
        # 根据agent_type选择不同的温度和系统提示词
        # agent_type可选值：
        # - coordinator: temp=0.3（稳定，生成大纲）
        # - researcher: temp=0.1（精确，生成查询）
        # - analyst: temp=0.5（平衡，分析数据）
        # - writer: temp=0.7（创意，撰写报告）
```

**为什么需要4种配置？**
- **Coordinator**：需要确定性，大纲结构要稳定
- **Researcher**：查询词要准确，不能太发散
- **Analyst**：需要适度创造性，但不能偏离事实
- **Writer**：需要最高创造性，文笔要好

---

### 3. RAG Retriever（检索器）

**作用**：封装现有的RAG Pipeline，提供检索能力

**位置**：`rag_project/agent/retriever.py`

**关键方法**：
```python
class RAGRetriever:
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        # 单次检索
        # 返回：[{"text": "...", "score": 0.95, "metadata": {...}}, ...]

    def search_multiple(self, queries: List[str], top_k: int = 5) -> Dict:
        # 批量检索
        # 返回：{"查询1": [结果...], "查询2": [结果...]}
```

**为什么需要这个包装器？**
- 复用现有代码，不重复造轮子
- 提供统一的接口给Agent使用
- 便于后续扩展（如添加缓存）

---

### 4. Coordinator（协调智能体）

**作用**：理解用户需求，生成报告大纲

**位置**：`rag_project/agent/nodes/coordinator.py`

**工作原理**：
```
输入：用户请求
    ↓
调用LLM（coordinator模式，temp=0.3）
    ↓
Prompt: "请生成5-8章的大纲，包含行业背景、政策环境、现状分析..."
    ↓
LLM输出：["行业概述", "政策环境", "现状分析", "问题挑战", "战略建议"]
    ↓
验证：是列表？章节数合理？
    ↓
更新状态：global_plan = [...], current_chapter_index = 0
```

**降级方案**：
如果LLM解析失败，使用默认大纲：
```python
[
    "行业概述与背景",
    "政策环境分析",
    "现状评估",
    "问题与挑战",
    "战略建议",
    "实施路径",
    "风险评估",
    "结论与建议"
]
```

---

### 5. Prepare Chapter（章节准备）

**作用**：每章开始前，初始化独立状态

**位置**：`rag_project/agent/nodes/prep_chapter.py`

**核心操作**：
```python
def prepare_chapter_node(state):
    # 1. 获取当前章节标题
    chapter_title = state["global_plan"][state["current_chapter_index"]]

    # 2. ⚠️ 关键：清空工作区（状态隔离）
    chapter_scratchpad = {}

    # 3. 清空草稿
    current_draft = ""

    return {
        "chapter_title": chapter_title,
        "chapter_scratchpad": {},  # 物理清空
        "current_draft": ""
    }
```

**为什么需要这步？**
- 防止上一章的数据污染本章
- 确保每章从零开始
- 实现"阅后即焚"的机制

---

### 6. Researcher（检索智能体）

**作用**：根据章节主题，生成多个检索查询并获取文档

**位置**：`rag_project/agent/nodes/researcher.py`

**工作流程**：
```
输入：chapter_title = "政策环境分析"
    ↓
第1步：生成检索查询
    调用LLM（researcher模式，temp=0.1）
    Prompt: "生成3-5个查询，覆盖不同角度..."
    输出：["2024年交通投资政策", "江西交通基础设施政策", "交通监管政策"]
    ↓
第2步：多路检索
    对每个查询调用retriever.search()
    结果1：[doc1, doc2, doc3, ...]  # 5篇
    结果2：[doc4, doc5, doc6, ...]  # 5篇
    结果3：[doc7, doc8, doc9, ...]  # 5篇
    ↓
第3步：合并去重
    合并：15篇文档
    去重：按文本内容hash去重
    排序：按score降序
    限制：保留top 20
    ↓
输出：chapter_scratchpad = {
    "queries": [...],
    "retrieved_docs": [doc1, doc2, ..., doc20]
}
```

**关键设计**：
- **多查询策略**：不同角度检索，避免单一查询的盲区
- **去重机制**：相同内容只保留一份
- **Top 20限制**：避免信息过载，给LLM太多内容会降低质量

---

### 7. Analyst（分析智能体）

**作用**：分析检索结果，提取关键事实并生成商业洞察

**位置**：`rag_project/agent/nodes/analyst.py`

**工作流程**：
```
输入：chapter_scratchpad["retrieved_docs"]（20篇文档）
    ↓
第1步：文档摘要
    限制：只取前10篇（避免LLM输入过长）
    格式化：
    文档1:
      来源: report.txt
      相关度: 0.95
      内容: 2024年交通投资增长15%...
    文档2: ...
    ...
    ↓
第2步：LLM分析
    调用LLM（analyst模式，temp=0.5）
    Prompt: "分析以上文档，提取关键事实并生成洞察..."
    要求：
    - key_facts: 5-10个，每个≤50字
    - insights: 3-5个，每个≤100字
    - 必须有引用来源 [来源: xxx]
    ↓
第3步：解析结果
    LLM输出：
    {
      "key_facts": ["2024年投资增长15% [来源: report.txt]", ...],
      "insights": ["投资增长反映行业活力", ...]
    }
    ↓
第4步：更新scratchpad
    保留：queries, retrieved_docs
    新增：key_facts, insights
    ↓
输出：chapter_scratchpad = {
    "queries": [...],
    "retrieved_docs": [...],
    "key_facts": [...],      # ← 新增
    "insights": [...]         # ← 新增
}
```

**为什么只看前10篇？**
- LLM的上下文窗口有限（4000 tokens）
- Top 10通常已包含最相关的内容
- 提高处理速度

**Key Facts vs Insights**：
- **Key Facts**：从文档直接提取的事实（数据、政策、事件）
- **Insights**：基于事实的分析和判断（趋势、原因、建议）

---

### 8. Writer（写作智能体）

**作用**：根据分析结果，撰写专业报告章节

**位置**：`rag_project/agent/nodes/writer.py`

**工作流程**：
```
输入：
  - chapter_title = "政策环境分析"
  - key_facts = ["2024年投资增长15%", ...]
  - insights = ["投资增长反映行业活力", ...]
  - context_summary = ""  # 前文摘要（首章为空）
    ↓
第1步：构建Prompt
    Prompt: """
    请为江西交通投资集团董事会撰写报告章节。

    章节标题: 政策环境分析

    前文摘要: （首章无前文）

    关键事实:
    - 2024年投资增长15%
    - 政策支持加强
    ...

    商业洞察:
    - 投资增长反映行业活力
    - 政策红利释放
    ...

    写作要求:
    1. 语调: 董事会级别的正式、专业、简洁
    2. 结构: 开篇(2-3句) → 主体(2-3小节) → 结尾(1-2句)
    3. 每段≤150字，使用项目符号
    4. 保留引用标注 [来源: xxx]
    5. 篇幅: 800-1200字
    """
    ↓
第2步：调用LLM
    调用LLM（writer模式，temp=0.7）
    ↓
第3步：生成草稿
    输出：
    # 政策环境分析

    ## 总体概况
    2024年以来，国家持续加大对交通基础设施的投资力度...

    ## 政策支持
    根据《交通强国建设纲要》 [来源: policy.txt]，政策支持力度...

    ## 投资趋势
    数据显示2024年投资增长15% [来源: report.txt]，反映出...

    ## 小结
    综上所述，政策环境持续向好，为集团发展提供了有力支撑...
    ↓
输出：current_draft = "上述全文"
```

**⚠️ 关键约束**：
- Writer**不能**直接读取`retrieved_docs`
- 必须**只使用**Analyst处理后的`key_facts`和`insights`
- 这确保了结构化分析链不被破坏

---

### 9. Human Review（人工审核）

**作用**：在每章生成后暂停，由人工审核质量

**位置**：`rag_project/agent/nodes/human_review.py`

**工作流程**：
```
第1步：显示草稿
    打印：
    ========================================================================
    📖 人工审核: 政策环境分析
    ========================================================================

    # 政策环境分析

    ## 总体概况
    2024年以来，国家持续加大对交通基础设施的投资力度...
    ...

    ========================================================================

第2步：获取反馈
    交互模式：
    请选择操作:
      1. approve - 通过，进入下一章
      2. revise:data - 数据不足，重新检索
      3. revise:logic - 逻辑问题，重新分析
      4. revise:writing - 文笔问题，重新润色

    请输入选择 (1-4):
    ↓
第3步：处理反馈
    如果选择 1 (approve):
        - 构建 full_chapter = f"# {chapter_title}\n\n{current_draft}"
        - 添加到 context_pool
        - 清空 chapter_scratchpad = {}
        - current_chapter_index += 1
        → 进入下一章

    如果选择 2 (revise:data):
        - 保持所有状态不变
        - current_chapter_index 不增加
        → 返回 Researcher 重新检索

    如果选择 3 (revise:logic):
        → 返回 Analyst 重新分析

    如果选择 4 (revise:writing):
        → 返回 Writer 重新撰写
```

**路由逻辑**（`should_continue`函数）：
```python
def should_continue(feedback, current_index, total_chapters):
    if feedback["decision"] == "approve":
        if current_index + 1 < total_chapters:
            return "continue"  # 下一章
        else:
            return "end"  # 报告完成
    elif feedback["decision"] == "revise":
        type = feedback["feedback_type"]
        if type == "data":
            return "researcher"
        elif type == "logic":
            return "analyst"
        elif type == "writing":
            return "writer"
```

**为什么需要人工介入？**
- LLM会"幻觉"，需要人工把关事实准确性
- 不同章节需要不同的重点，需要人工调整方向
- 确保报告符合董事会的要求和语调

---

### 10. Archiver（归档节点）

**作用**：合并所有章节，生成最终报告

**位置**：`rag_project/agent/nodes/archiver.py`

**工作流程**：
```
输入：context_pool = [
    "# 第一章 行业概述\n\n内容...",
    "# 第二章 政策环境\n\n内容...",
    "# 第三章 现状分析\n\n内容...",
    ...
]
    ↓
第1步：添加封面
    cover = """
    # 江西交通投资集团战略规划报告

    **生成时间**: 2026年03月28日

    **主题**: 生成2024年江西交通投资政策解读报告

    ---

    """
    ↓
第2步：合并章节
    body = "\n\n---\n\n".join(context_pool)
    ↓
第3步：添加页脚
    footer = """

    ---

    **报告说明**:
    本报告由AI智能体系统基于企业知识库自动生成，包含以下章节:

    1. 行业概述
    2. 政策环境
    3. 现状分析
    ...

    **共N章**
    """
    ↓
输出：final_report = cover + body + footer
```

---

### 11. LangGraph工作流引擎

**作用**：将所有节点编排成一个完整的工作流

**位置**：`rag_project/agent/graph.py`

**关键代码**：
```python
def create_report_graph():
    # 1. 创建状态图
    workflow = StateGraph(GraphState)

    # 2. 添加所有节点
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("prepare_chapter", prepare_chapter_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("archiver", archiver_node)

    # 3. 设置入口
    workflow.set_entry_point("coordinator")

    # 4. 添加边（连接节点）
    workflow.add_edge("coordinator", "prepare_chapter")
    workflow.add_edge("prepare_chapter", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "human_review")

    # 5. 添加条件边（路由）
    workflow.add_conditional_edges(
        "human_review",
        review_routing,  # 路由函数
        {
            "continue": "prepare_chapter",  # 下一章
            "researcher": "researcher",      # 重新检索
            "analyst": "analyst",            # 重新分析
            "writer": "writer",              # 重新写作
            "end": "archiver"                # 结束
        }
    )

    # 6. 编译（添加检查点和中断）
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"]  # ⚠️ 关键：人工审核前暂停
    )

    return app
```

**什么是`interrupt_before`？**
- 在执行到`human_review`节点之前，**暂停**工作流
- 等待人类输入反馈
- 收到反馈后，**恢复**执行

---

## 数据流

### 完整数据流示例

```
用户输入："生成2024年交通投资政策解读报告"
    ↓
[GraphState初始化]
{
    "user_input": "生成2024年交通投资政策解读报告",
    "global_plan": [],
    "current_chapter_index": 0,
    "context_pool": [],
    "chapter_scratchpad": {},
    "current_draft": "",
    "human_feedback": {}
}
    ↓
[Coordinator处理]
→ global_plan = ["行业概述", "政策环境", "现状分析", "战略建议"]
→ current_chapter_index = 0
    ↓
[Prepare Chapter处理（第一章）]
→ chapter_title = "行业概述"
→ chapter_scratchpad = {}  ⚠️ 清空
→ current_draft = ""
    ↓
[Researcher处理]
→ chapter_scratchpad = {
    "queries": ["2024交通投资", "江西交通现状"],
    "retrieved_docs": [
        {"text": "2024年交通投资增长15%", "score": 0.95, ...},
        {"text": "基础设施建设加快", "score": 0.88, ...},
        ...  # 20篇文档
    ]
}
    ↓
[Analyst处理]
→ chapter_scratchpad = {
    "queries": [...],
    "retrieved_docs": [...],
    "key_facts": [
        "2024年交通投资增长15% [来源: report.txt]",
        "基础设施建设加快 [来源: stats.txt]",
        ...
    ],
    "insights": [
        "投资增长反映行业活力",
        "基础设施缺口仍大",
        ...
    ]
}
    ↓
[Writer处理]
→ current_draft = "# 行业概述\n\n2024年以来，交通行业..."
    ↓
[Human Review - 用户选择approve]
→ context_pool = [
    "# 行业概述\n\n2024年以来，交通行业..."
]
→ chapter_scratchpad = {}  ⚠️ 再次清空
→ current_chapter_index = 1
→ current_draft = ""
    ↓
[Prepare Chapter处理（第二章）]
→ chapter_title = "政策环境"
→ chapter_scratchpad = {}  ⚠️ 再次清空
    ↓
... 重复上述流程 ...
    ↓
[所有章节完成后]
→ context_pool = [第一章, 第二章, 第三章, 第四章]
    ↓
[Archiver处理]
→ final_report = """
# 江西交通投资集团战略规划报告

**生成时间**: 2026年03月28日
**主题**: 生成2024年交通投资政策解读报告

---

# 行业概述

2024年以来，交通行业...

---

# 政策环境

...

---

**报告说明**:
本报告由AI智能体系统基于企业知识库自动生成...
"""
```

---

## 运行指南

### 前置条件

1. **安装依赖**
   ```bash
   pip install -r requirements-agent.txt
   ```

2. **设置API Key**
   ```bash
   # Windows
   set DEEPSEEK_API_KEY=your_api_key_here

   # Linux/Mac
   export DEEPSEEK_API_KEY=your_api_key_here
   ```

3. **启动Milvus**
   ```bash
   docker-compose up -d
   ```

4. **确认Milvus有数据**
   ```bash
   python -c "from rag_project.storage.milvus_manager import MilvusManager; mm = MilvusManager('config/milvus_config.yaml'); print(mm.get_collection_stats())"
   ```

### 运行方式

#### 方式1：交互模式（推荐）

适合实际使用，可以逐章审核和调整。

```bash
cd "E:\02 Final Year Project\RAG Project"
python scripts/run_agent_report.py "生成2024年江西交通投资政策解读报告"
```

**运行过程**：
```
============================================================
🚀 智能体战略报告生成系统
============================================================
📝 请求: 生成2024年江西交通投资政策解读报告
💾 输出: output/report.md
👤 交互模式
============================================================

📋 Report generation started...

✅ Coordinator: 生成大纲完成
   章节列表: ['行业概述', '政策环境分析', '现状评估', '问题与挑战', '战略建议']

📋 准备章节: 行业概述 (第1/5章)

   🔍 Researcher: 检索完成
   查询: ['2024交通投资概述', '江西交通行业现状', '交通行业发展']
   文档: 20篇

   📊 Analyst: 分析完成
   关键事实: 8条
   商业洞察: 4条

   ✍️  Writer: 草稿生成完成 (956字)

============================================================
📖 人工审核: 行业概述
============================================================

# 行业概述

## 总体概况
2024年以来，中国交通行业持续发展...
...

============================================================

请选择操作:
  1. approve - 通过，进入下一章
  2. revise:data - 数据不足，重新检索
  3. revise:logic - 逻辑问题，重新分析
  4. revise:writing - 文笔问题，重新润色

请输入选择 (1-4): 1  ← 你输入1

✅ Chapter approved!

📋 准备章节: 政策环境分析 (第2/5章)
   ... (重复上述流程)

============================================================
📦 Archiver: 报告归档完成

✅ 报告生成完成！

============================================================

✅ 报告已保存到: output/report.md
```

#### 方式2：自动模式（测试）

适合快速测试，不需要人工介入。

```bash
python scripts/run_agent_report.py "生成2024年江西交通投资政策解读报告" --auto
```

**运行过程**：
```
...
   ⏭️  自动模式: 默认通过
...
```

#### 方式3：自定义输出路径

```bash
python scripts/run_agent_report.py "生成年度总结报告" --output reports/2024_summary.md
```

### 人工审核最佳实践

#### 何时选择 `approve`？
- ✅ 内容准确，数据正确
- ✅ 逻辑清晰，层次分明
- ✅ 语言专业，符合董事会语调
- ✅ 有明确的引用来源

#### 何时选择 `revise:data`？
- ❌ 检索结果少于5篇
- ❌ 关键数据缺失（具体数字、日期）
- ❌ 引用来源不权威或缺失
- ❌ 内容与主题不相关

#### 何时选择 `revise:logic`？
- ❌ 洞察过于表面，缺乏深度
- ❌ 事实与结论之间逻辑断层
- ❌ 前后矛盾或重复
- ❌ 分析角度单一

#### 何时选择 `revise:writing`？
- ❌ 语言不正式，不符合董事会语调
- ❌ 段落过长，结构混乱
- ❌ 缺乏专业术语
- ❌ 格式不统一

### 常见问题

#### Q1: 报告生成速度慢？
**原因**: LLM API调用延迟 + Milvus检索时间

**解决**:
- 检查网络连接
- 增加`agent_config.yaml`中的`timeout`值
- 考虑使用更快的模型

#### Q2: 检索结果不相关？
**原因**: 查询词质量或知识库覆盖不足

**解决**:
- 使用`revise:data`并提供具体的缺失数据
- 检查Milvus索引是否正常
- 扩充知识库内容

#### Q3: 报告质量不稳定？
**原因**: LLM的随机性（temperature参数）

**解决**:
- 降低`agent_config.yaml`中的`temperature`值
- 使用`revise:writing`多次润色
- 在系统提示词中增加更具体的约束

#### Q4: 内存不足？
**原因**: Milvus + Embeddings模型占用大量内存

**解决**:
- 关闭其他应用
- 将Embeddings模型移到CPU：修改`milvus_config.yaml`的`device: cpu`
- 减少批处理大小：`batch_size: 16`

---

## 总结

### 核心设计原则

1. **单一职责**：每个Agent只做一件事
2. **状态隔离**：章节数据互不干扰
3. **人类主导**：关键决策由人工控制
4. **结构化流**：检索→分析→写作，层层递进

### 技术亮点

- ✅ LangGraph工作流编排
- ✅ 三层记忆架构
- ✅ HITL人工介入
- ✅ 条件路由机制
- ✅ 降级容错方案

### 适用场景

- ✅ 企业战略报告生成
- ✅ 政策解读报告
- ✅ 行业研究报告
- ✅ 年度/季度总结
- ✅ 专题研究报告

---

**文档版本**: v1.0
**更新时间**: 2026-03-28
**作者**: Claude (Sonnet 4.5)
