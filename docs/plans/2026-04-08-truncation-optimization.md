# 硬截断优化 & 滚动摘要 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 agent 节点中的硬截断替换为智能压缩，并实现跨章节滚动摘要机制，解决信息丢失和章节间零连贯性问题。

**Architecture:** 保持 scratchpad 隔离设计不变（原始文档仍阅后即焚），新增"首尾提取"替代纯截断，激活已有的 `context_summary` 字段实现滚动压缩摘要。

**Tech Stack:** Python, LangGraph, DeepSeek API (OpenAI SDK)

---

## 修改总览

| # | 文件 | 改动 | 优先级 |
|---|------|------|--------|
| 1 | `rag_project/agent/nodes/analyst.py` | 文档文本 `text[:500]` → 首尾提取 + 动态预算 | 高 |
| 2 | `rag_project/agent/nodes/writer.py` | 移除 `document_summary[:500]` 截断 | 高 |
| 3 | `rag_project/agent/nodes/strategist.py` | 3章诊断压缩替代 `chapter_3[:3000]` | 高 |
| 4 | `rag_project/agent/nodes/archiver.py` | 章节预览 `chapter[:100]` → LLM 摘要 | 中 |
| 5 | `rag_project/agent/nodes/prep_chapter.py` | 实现滚动 context_summary 压缩 | 高 |
| 6 | `rag_project/agent/nodes/writer.py` | Writer prompt 注入 context_summary | 高 |

---

## Task 1: Analyst 文档文本首尾提取

**问题：** `analyst.py:176` 将每个文档截断为前500字，丢失文档后半部分的关键数据和结论。

**策略：** 用"首尾提取"替代纯截断。对每个文档提取前 N 字和后 M 字，中间用省略标记连接。同时引入总 token 预算，动态分配每个文档的提取量。

**Files:**
- Modify: `rag_project/agent/nodes/analyst.py:117-185`

**Step 1: 新增 `_smart_extract` 函数**

在 `_generate_document_summary` 函数之前，添加智能提取函数：

```python
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
```

**Step 2: 新增 `_generate_document_summary_v2` 函数**

替代原函数，引入动态预算分配：

```python
def _generate_document_summary_v2(documents: List[Dict[str, Any]], total_budget: int = 18000) -> str:
    """
    生成文档摘要（v2: 首尾提取 + 动态预算分配）。

    改进:
    - 不再硬限制为10个文档，根据总预算动态决定处理数量
    - 每个文档使用首尾提取而非纯截断
    - 元数据（来源、页码）占用独立预算，不影响文本预算

    预算说明（平衡方案）:
    - 总预算 18000 字符 ≈ 6000-9000 tokens
    - DeepSeek-V3 上下文 64K，Analyst prompt 总占用约 8000-12000 tokens（约 19%）
    - 兼顾信息保留和 API 成本

    Args:
        documents: 文档列表
        total_budget: 总字符预算（默认18000，约6000-9000 tokens）

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

        # 提取元数据（复用原有逻辑）
        source = metadata.get("source", "")
        page = metadata.get("page_number", "N/A")
        title = metadata.get("title", "")
        doc_type = metadata.get("doc_type", "")

        # 确定显示名称（原有 fallback chain 逻辑不变）
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
```

**Step 3: 修改 `analyst_node` 调用**

将 `analyst_node` 中第88行：

```python
# 旧代码
document_summary = _generate_document_summary(retrieved_docs, limit=10)
```

替换为：

```python
# 新代码: 使用 v2 版本，自动处理预算分配
document_summary = _generate_document_summary_v2(retrieved_docs, total_budget=18000)
```

**Step 4: 保留旧函数作为回退**

不删除 `_generate_document_summary`，保留作为回退使用。

**Step 5: 验证**

运行 agent 并检查日志中：
- 文档摘要是否包含 `[中间内容省略]` 标记（说明长文档被正确处理）
- 总摘要字符数是否在 12000 以内
- 短文档（<800字）是否完整保留

```bash
cd "E:/02 Final Year Project/RAG Project"
python -c "
from rag_project.agent.nodes.analyst import _smart_extract

# 测试短文本（不应被截断）
short = '短文本内容' * 10
result = _smart_extract(short, budget=900)
assert result == short, '短文本不应被修改'

# 测试长文本（应首尾提取）
long_text = 'A' * 2000
result = _smart_extract(long_text, budget=900)
assert '[中间内容省略]' in result, '长文本应包含省略标记'
assert result.startswith('A'), '应保留开头'
assert result.endswith('A'), '应保留结尾'
print('All tests passed!')
"
```

**Step 6: Commit**

```bash
git add rag_project/agent/nodes/analyst.py
git commit -m "refactor: replace hard truncation with smart head-tail extraction in analyst"
```

---

## Task 2: Writer 移除 document_summary 截断

**问题：** `writer.py:342` 将 Analyst 生成的文档摘要（已压缩）再次截断为500字，导致 Writer 丢失后几个文档的引用信息。

**Files:**
- Modify: `rag_project/agent/nodes/writer.py:341-342`

**Step 1: 修改 `_generate_writing_prompt` 中的截断**

将第341-342行：

```python
# 旧代码
if document_summary:
    context_parts.append(f"参考文档摘要: {document_summary[:500]}...")
```

替换为：

```python
# 新代码: 传递完整摘要，不再截断
if document_summary:
    # Analyst 已对文档做了预算控制，直接传递完整摘要
    context_parts.append(f"参考文档摘要:\n{document_summary}")
```

**Step 2: 调整 LLM max_tokens 预算**

Writer 当前 `max_tokens=4096`，输入变大后输出可能被截断。需确保输入+输出在模型上下文内。

DeepSeek-V3 上下文窗口为 64K tokens，当前输入估算：
- prompt 约 5000-8000 字符 ≈ 2000-3000 tokens
- 新增完整摘要约 12000 字符 ≈ 4000 tokens
- 总输入约 6000-7000 tokens，远低于 64K 上限

**无需修改 max_tokens**，64K 窗口完全足够。

**Step 3: Commit**

```bash
git add rag_project/agent/nodes/writer.py
git commit -m "fix: remove document_summary truncation in writer prompt"
```

---

## Task 3: Strategist 诊断阶段全章压缩

**问题：** `strategist.py:75` 只读取第3章，且 `strategist.py:128` 将其截断为3000字。Strategist 完全看不到第1-2章的环境分析和区域战略，生成的蓝图缺少宏观背景。

**策略：** 读取全部3章，用 LLM 生成"诊断阶段综合摘要"（约2000字），然后从中提取 SWOT。

**Files:**
- Modify: `rag_project/agent/nodes/strategist.py:22-111`

**Step 1: 新增 `_compress_diagnosis` 函数**

在 `strategist_node` 函数之前添加：

```python
def _compress_diagnosis(context_pool: List[str], user_input: str, llm_manager) -> str:
    """
    将诊断阶段（前3章）压缩为结构化综合摘要。

    相比原方案(chapter_3[:3000])的改进:
    - 涵盖全部3章内容，而非只看第3章
    - 使用 LLM 智能压缩，保留关键数据和分析结论
    - 输出约2000字的结构化摘要

    Args:
        context_pool: 已完成的章节列表
        user_input: 用户原始请求
        llm_manager: LLM 实例

    Returns:
        诊断阶段综合摘要文本
    """
    # 拼合前3章内容（每章限5000字避免过长）
    chapters_text = ""
    for i, chapter in enumerate(context_pool[:3]):
        # 提取章节标题
        title = f"第{i+1}章"
        for line in chapter.split('\n'):
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        content = chapter[:5000] if len(chapter) > 5000 else chapter
        chapters_text += f"\n\n=== {title} ===\n{content}"

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
5. 2000字以内
6. 必须使用中文

诊断阶段内容:
{chapters_text}

请生成综合摘要:"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=2048)
        logger.info(f"Compressed diagnosis chapters into {len(response)} chars")
        return response
    except Exception as e:
        logger.error(f"Failed to compress diagnosis: {e}. Using raw Chapter 3.")
        # 回退: 使用第3章的前5000字（比原来的3000字多）
        chapter_3 = context_pool[2] if len(context_pool) >= 3 else ""
        return chapter_3[:5000]
```

**Step 2: 修改 `strategist_node` 使用压缩摘要**

替换 `strategist_node` 中的第74-84行（从 `chapter_3 = context_pool[2]...` 到 `_extract_swot_from_chapter` 调用）：

```python
# 旧代码:
# chapter_3 = context_pool[2] if len(context_pool) >= 3 else ""
# if not chapter_3:
#     ...
# swot_data = _extract_swot_from_chapter(chapter_3, llm_manager)

# 新代码: 压缩全部3章后再提取 SWOT
diagnosis_summary = _compress_diagnosis(context_pool, user_input, llm_manager)

if not diagnosis_summary:
    logger.error("Failed to generate diagnosis summary")
    return _get_fallback_blueprint(user_input)

swot_data = _extract_swot_from_chapter(diagnosis_summary, llm_manager)
```

**Step 3: 修改 `_extract_swot_from_chapter` 移除截断**

将第128行的 `chapter_text[:3000]` 替换为直接使用完整文本（因为输入已经是压缩后的摘要）：

```python
# 旧代码
章节文本:
{chapter_text[:3000]}

# 新代码（输入已压缩，无需再次截断）
章节文本:
{chapter_text}
```

**Step 4: Commit**

```bash
git add rag_project/agent/nodes/strategist.py
git commit -m "feat: compress all 3 diagnosis chapters before SWOT extraction"
```

---

## Task 4: Archiver 章节预览增强

**问题：** `archiver.py:259` 将每章预览截断为100字，导致执行摘要生成时 LLM 缺乏足够信息。

**Files:**
- Modify: `rag_project/agent/nodes/archiver.py:248-260`

**Step 1: 新增 `_generate_chapter_summary` 函数**

在 `_generate_executive_summary` 之前添加：

```python
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
```

**Step 2: 修改 `_generate_executive_summary` 中的章节概要生成**

替换第248-260行：

```python
# 旧代码:
# for i, chapter in enumerate(context_pool):
#     ...
#     preview = chapter[:100].replace('\n', ' ')[:100]
#     chapter_overview += f"- {title}: {preview}...\n"

# 新代码:
for i, chapter in enumerate(context_pool):
    lines = chapter.split('\n')
    title = f"第{i+1}章"
    for line in lines:
        if line.strip().startswith('#'):
            title = line.strip().lstrip('#').strip()
            break
    brief = _generate_chapter_brief(chapter, max_chars=300)
    chapter_overview += f"- {title}: {brief}\n"
```

**Step 3: Commit**

```bash
git add rag_project/agent/nodes/archiver.py
git commit -m "feat: replace chapter preview truncation with structured brief extraction"
```

---

## Task 5: PrepChapter 实现 context_summary 滚动压缩

**问题：** `state.py` 定义了 `context_summary` 但从未被任何节点写入，Writer 读取时始终为空字符串。

**策略：** 在 PrepChapter 节点中，当 scratchpad 中有上一章的分析结果时，调用 LLM 压缩后追加到 context_summary。

**Files:**
- Modify: `rag_project/agent/nodes/prep_chapter.py:22-134`

**Step 1: 新增 `_compress_chapter_knowledge` 函数**

在 `prep_chapter.py` 顶部 import 后添加：

```python
def _compress_chapter_knowledge(
    chapter_title: str,
    scratchpad: Dict,
    existing_summary: str,
    llm_manager
) -> str:
    """
    将上一章的知识压缩并合并到滚动摘要中。

    如果是第一章（无前序知识），返回空字符串。
    如果已有滚动摘要，将新知识压缩合并。

    Args:
        chapter_title: 上一章标题
        scratchpad: 上一章的 scratchpad（含 key_facts, insights）
        existing_summary: 现有的滚动摘要
        llm_manager: LLM 实例（可选，为 None 时使用简单拼接）

    Returns:
        更新后的滚动摘要（约 300-500 字）
    """
    key_facts = scratchpad.get("key_facts", [])
    insights = scratchpad.get("insights", [])

    if not key_facts and not insights:
        return existing_summary

    # 构建新知识文本
    new_knowledge = f"\n【{chapter_title}】\n"

    if isinstance(key_facts, dict):
        # 结构化 facts（如 SWOT/PEST 模型输出）
        for category, facts in key_facts.items():
            if isinstance(facts, list):
                new_knowledge += f"  {category}: {'; '.join(str(f) for f in facts[:5])}\n"
            else:
                new_knowledge += f"  {category}: {facts}\n"
    elif isinstance(key_facts, list):
        new_knowledge += "  关键事实: " + "; ".join(str(f) for f in key_facts[:8]) + "\n"

    if insights:
        new_knowledge += "  核心洞察: " + "; ".join(str(i) for i in insights[:5]) + "\n"

    # 如果没有 LLM，使用简单拼接（带长度控制）
    if llm_manager is None:
        combined = existing_summary + "\n" + new_knowledge if existing_summary else new_knowledge
        # 简单截断保底
        return combined[-2000:] if len(combined) > 2000 else combined

    # 使用 LLM 滚动压缩
    prompt = f"""请将以下新旧知识合并为一份紧凑的滚动摘要。

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

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=512)
        return response.strip()
    except Exception as e:
        # LLM 失败时回退到简单拼接
        combined = existing_summary + "\n" + new_knowledge if existing_summary else new_knowledge
        return combined[-2000:] if len(combined) > 2000 else combined
```

**Step 2: 修改 `prepare_chapter_node` 签名和逻辑**

函数需要接收 llm_manager 参数用于压缩。但 PrepChapter 当前不接收依赖注入。

修改函数签名：

```python
def prepare_chapter_node(state: Dict, llm_manager=None) -> Dict:
```

修改 graph.py 中的节点注册：

```python
# 旧代码
workflow.add_node("prepare_chapter", prepare_chapter_node)

# 新代码
workflow.add_node("prepare_chapter", lambda state: prepare_chapter_node(state, prep_llm))
```

在 `graph.py` 中初始化 PrepChapter 专用的 LLM：

```python
prep_llm = LLMManager("coordinator")  # 复用 coordinator 配置（低温度）
```

**Step 3: 在 PrepChapter 中调用压缩**

在 `prepare_chapter_node` 的 return 之前，添加滚动摘要更新逻辑：

```python
    # === 新增: 滚动摘要压缩 ===
    context_summary_update = {}
    scratchpad = state.get("chapter_scratchpad", {})
    current_summary = state.get("context_summary", "")

    # 如果 scratchpad 中有上一章的知识（非空且非刚清空），进行压缩
    if scratchpad and scratchpad.get("key_facts"):
        prev_title = state.get("chapter_title", "")
        compressed = _compress_chapter_knowledge(
            chapter_title=prev_title,
            scratchpad=scratchpad,
            existing_summary=current_summary,
            llm_manager=llm_manager
        )
        context_summary_update = {"context_summary": compressed}
        logger.info(f"Updated context_summary ({len(compressed)} chars) after compressing '{prev_title}'")

    # 合并返回
    return {
        "chapter_title": chapter_title,
        "chapter_question": chapter_question,
        "chapter_context": chapter_context,
        "chapter_scratchpad": chapter_scratchpad,
        "current_draft": current_draft,
        **context_summary_update  # 如果有更新则包含
    }
```

**注意:** 此逻辑必须在 scratchpad 清空**之前**执行。需要调整函数内的执行顺序：先读取旧 scratchpad 做压缩，再清空。

调整后的函数流程：

```python
def prepare_chapter_node(state: Dict, llm_manager=None) -> Dict:
    # 1. 先读取上一章的 scratchpad（在清空之前）
    prev_scratchpad = state.get("chapter_scratchpad", {})
    prev_chapter_title = state.get("chapter_title", "")
    current_summary = state.get("context_summary", "")

    # 2. 滚动压缩（使用上一章数据）
    context_summary_update = {}
    if prev_scratchpad and prev_scratchpad.get("key_facts"):
        compressed = _compress_chapter_knowledge(
            chapter_title=prev_chapter_title,
            scratchpad=prev_scratchpad,
            existing_summary=current_summary,
            llm_manager=llm_manager
        )
        context_summary_update = {"context_summary": compressed}

    # 3. 设置新章节状态（清空 scratchpad）
    global_plan = state.get("global_plan", [])
    current_index = state.get("current_chapter_index", 0)
    chapter_metadata = global_plan[current_index] if current_index < len(global_plan) else {}
    # ... (原有标题、问题、上下文生成逻辑不变) ...

    # 4. 清空 scratchpad（阅后即焚）
    chapter_scratchpad = {}

    # 5. 推演阶段注入蓝图（原有逻辑不变）
    # ...

    # 6. 返回
    return {
        "chapter_title": chapter_title,
        "chapter_question": chapter_question,
        "chapter_context": chapter_context,
        "chapter_scratchpad": chapter_scratchpad,
        "current_draft": "",
        **context_summary_update
    }
```

**Step 4: 修改 graph.py**

```python
# 在 create_report_graph 中添加
prep_llm = LLMManager("coordinator")  # 复用低温度配置

# 修改节点注册
workflow.add_node("prepare_chapter", lambda state: prepare_chapter_node(state, prep_llm))
```

**Step 5: Commit**

```bash
git add rag_project/agent/nodes/prep_chapter.py rag_project/agent/graph.py
git commit -m "feat: implement rolling context_summary compression in prep_chapter"
```

---

## Task 6: Writer 注入 context_summary

**问题：** Writer 的 prompt 中读取了 `context_summary`，但它始终为空字符串，等于没有跨章节上下文。

**Files:**
- Modify: `rag_project/agent/nodes/writer.py:339-340`

**Step 1: 增强 context_summary 的呈现**

将 writer.py 第339-340行：

```python
# 旧代码
if context_summary:
    context_parts.append(f"摘要: {context_summary}")
```

替换为：

```python
# 新代码: 更明确的提示
if context_summary:
    context_parts.append(
        f"前序章节已确立的关键信息（请在写作中保持一致，不要矛盾）:\n{context_summary}"
    )
```

**Step 2: 在写作 prompt 通用要求中增加跨章节一致性指令**

在 `_generate_writing_prompt` 的"通用写作要求"部分末尾添加：

```python
    # 如果有 context_summary，添加一致性约束
    if context_summary:
        consistency_instruction = """
7. **跨章节一致性**:
   - 如果前序章节已提及具体数字（如投资额、增长率），必须使用相同数字
   - 不要重复前序章节已有的内容，而是在其基础上深化
   - 如果发现前序结论与当前分析矛盾，以当前分析为准但需说明变化原因
"""
```

并将 `consistency_instruction` 拼接到 prompt 末尾。

**Step 3: Commit**

```bash
git add rag_project/agent/nodes/writer.py
git commit -m "feat: inject context_summary into writer prompt with consistency constraints"
```

---

## 执行顺序与依赖关系

```
Task 1 (Analyst 首尾提取)     ──┐
Task 2 (Writer 移除截断)       ──┤  可并行执行
Task 3 (Strategist 全章压缩)   ──┤
Task 4 (Archiver 预览增强)     ──┘
                │
                ▼
Task 5 (滚动摘要压缩)  ←── 依赖 scratchpad 流程理解
                │
                ▼
Task 6 (Writer 注入摘要) ←── 依赖 Task 5 产出 context_summary
```

## 预期效果

| 指标 | 优化前 | 优化后（平衡方案） |
|------|--------|--------|
| 每个文档传给 Analyst 的信息量 | 前500字 | 首尾900字（60%头+40%尾） |
| Writer 看到的文档摘要量 | 500字 | 完整摘要（~18000字） |
| Strategist 看到的诊断内容 | 第3章前3000字 | 3章综合压缩摘要（~2000字） |
| Archiver 章节预览 | 100字随机截取 | 标题+小节+关键句（~300字） |
| Writer 跨章节视野 | 0字 | 滚动摘要（~500字） |
| 总预算占 64K 上下文比 | ~5% | ~19% |
| 额外 LLM 调用/章 | 0 | +1次（滚动摘要压缩） |
| 额外延迟/章 | 0 | +3-5秒 |

## 不改动的截断点

以下截断为合理设计，**不做修改**：

- `researcher.py`: `queries[:5]`, `top_docs[:20]` — 检索量控制，合理
- `llm_manager.py`: 各 agent 的 `max_tokens` — 输出长度限制，合理
- 所有 `logger.info/debug` 中的截断 — 日志展示，不影响功能
- `analyst.py:165` `title[:80]` — 显示名称长度，合理