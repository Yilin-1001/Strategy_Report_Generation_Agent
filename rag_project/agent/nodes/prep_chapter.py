"""
Prepare Chapter Node - Initializes chapter state and enforces state isolation

This helper node is called at the beginning of each chapter iteration to:
1. Compress previous chapter knowledge into rolling context_summary
2. Set the current chapter title from the global plan
3. Clear the chapter scratchpad (enforcing state isolation between chapters)
4. Clear the current draft (ensuring fresh start)
5. Log progress with chapter number (X of Y)

This is critical for maintaining the "阅后即焚" (burn after reading) principle
where chapter_scratchpad serves as a temporary workspace that is reset for each chapter.
"""

import logging
from typing import Dict

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def _compress_chapter_knowledge(
    chapter_title: str,
    key_facts,
    insights: list,
    existing_summary: str,
    llm_manager
) -> str:
    """
    将上一章的知识压缩并合并到滚动摘要中。

    Args:
        chapter_title: 上一章标题
        key_facts: 上一章的关键事实（list 或 dict）
        insights: 上一章的洞察
        existing_summary: 现有的滚动摘要
        llm_manager: LLM 实例（可选，为 None 时使用简单拼接）

    Returns:
        更新后的滚动摘要（约 300-500 字）
    """
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
        return combined[-3000:] if len(combined) > 3000 else combined

    # 使用 LLM 滚动压缩
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

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=512)
        return response.strip()
    except Exception as e:
        # LLM 失败时回退到简单拼接
        logger.warning(f"LLM compression failed, using simple concat: {e}")
        combined = existing_summary + "\n" + new_knowledge if existing_summary else new_knowledge
        return combined[-3000:] if len(combined) > 3000 else combined


def _detect_knowledge_gaps(
    next_chapter_title: str,
    context_summary: str,
    llm_manager
) -> str:
    """
    检测前序章节是否存在下一章需要的知识缺口。

    使用 LLM 判断前序章节摘要是否已覆盖下一章所需的关键信息，
    如果存在缺口，返回补充检索方向的提示文本。

    Args:
        next_chapter_title: 下一章标题
        context_summary: 前序章节的压缩摘要
        llm_manager: LLM 实例

    Returns:
        缺口提示文本（空字符串表示无缺口）
    """
    if not context_summary or not next_chapter_title:
        return ""

    prompt = f"""基于前序章节的压缩摘要，判断是否有下一章需要但尚未覆盖的关键信息。

下一章: {next_chapter_title}

前序章节摘要:
{context_summary}

如果存在知识缺口，用1-2句话描述需要补充检索的方向。如果没有明显缺口，返回 NONE。"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=200)
        if "NONE" in response.upper():
            return ""
        return response.strip()
    except Exception as e:
        logger.warning(f"Knowledge gap detection failed: {e}")
        return ""


def prepare_chapter_node(state: Dict, llm_manager=None) -> Dict:
    """
    为当前章节准备状态，设置标题并清空工作区。

    新增滚动摘要压缩: 在清空 scratchpad 之前，将上一章的知识压缩到 context_summary 中，
    确保后续章节能看到前序章节的关键信息。

    推演阶段特殊处理 (initiatives phase):
    - 如果当前章节 phase == "initiatives" (第4-8章)
    - 且 strategic_blueprint 存在并已批准
    - 将自动注入战略蓝图上下文到 chapter_scratchpad
    - 确保后续生成的战略举措不偏离总目标

    Args:
        state: Current GraphState containing:
            - global_plan: List of chapter metadata
            - current_chapter_index: Index of current chapter (0-based)
            - strategic_blueprint: Strategic blueprint (optional)
            - current_phase: Current phase
            - user_input: Original user request
            - chapter_scratchpad: Previous chapter's scratchpad (to be compressed then cleared)
            - context_summary: Rolling summary from previous chapters
        llm_manager: LLM instance for chapter knowledge compression

    Returns:
        Dict with updates:
            - chapter_title: Title from global_plan[current_chapter_index]
            - chapter_question: Research question for this chapter
            - chapter_context: Context for this chapter
            - chapter_scratchpad: Empty dict OR with strategic_blueprint
            - current_draft: Empty string "" (fresh start)
            - context_summary: Updated rolling summary (if previous chapter had data)
    """
    # === Step 1: 读取上一章的知识（由 human_review 保存）===
    pending_knowledge = state.get("_pending_chapter_knowledge", {})
    current_summary = state.get("context_summary", "")

    # === Step 2: 滚动压缩（使用上一章数据）===
    context_summary_update = {}
    if pending_knowledge and pending_knowledge.get("key_facts"):
        compressed = _compress_chapter_knowledge(
            chapter_title=pending_knowledge.get("title", ""),
            key_facts=pending_knowledge["key_facts"],
            insights=pending_knowledge.get("insights", []),
            existing_summary=current_summary,
            llm_manager=llm_manager
        )
        context_summary_update = {"context_summary": compressed}
        logger.info(f"Updated context_summary ({len(compressed)} chars) "
                    f"after compressing '{pending_knowledge.get('title', '')}'")

    # === Step 3: 设置新章节状态 ===
    global_plan = state.get("global_plan", [])
    current_index = state.get("current_chapter_index", 0)
    total_chapters = len(global_plan)

    # Get current chapter metadata from global plan
    chapter_metadata = global_plan[current_index] if current_index < len(global_plan) else {}

    # Handle both Dict (new) and str (old) formats for backward compatibility
    if isinstance(chapter_metadata, dict):
        chapter_title = chapter_metadata.get("title", "")
        chapter_phase = chapter_metadata.get("phase", "")
    else:
        chapter_title = str(chapter_metadata)
        chapter_phase = ""

    # Log progress with chapter number (X of Y)
    chapter_num = current_index + 1
    logger.info(f"Preparing Chapter {chapter_num} of {total_chapters}: {chapter_title} (phase: {chapter_phase or 'unknown'})")

    # CRITICAL: Clear chapter_scratchpad to enforce state isolation
    chapter_scratchpad = {}

    # Inject strategic blueprint for initiatives phase
    strategic_blueprint = state.get("strategic_blueprint", {})
    current_phase = state.get("current_phase", "")
    is_initiatives_phase = chapter_phase == "initiatives" and current_phase == "initiatives"

    if is_initiatives_phase and strategic_blueprint and strategic_blueprint.get("approved", False):
        chapter_scratchpad["strategic_blueprint"] = strategic_blueprint
        logger.info(f"Injected strategic blueprint into chapter scratchpad (mission: {strategic_blueprint.get('mission', '')[:30]}...)")
    elif chapter_phase == "initiatives" and (not strategic_blueprint or not strategic_blueprint.get("approved", False)):
        logger.warning(f"Chapter {chapter_num} is in initiatives phase but strategic blueprint is not approved. "
                       f"Proceeding without blueprint constraints.")

    # Clear current_draft to ensure fresh start
    current_draft = ""

    # Generate chapter_question from chapter_title
    if "：" in chapter_title or ":" in chapter_title:
        separator = "：" if "：" in chapter_title else ":"
        parts = chapter_title.split(separator, 1)
        chapter_question = parts[1].strip() if len(parts) > 1 else chapter_title
    else:
        chapter_question = chapter_title

    # Generate chapter_context from user input if available
    user_input = state.get("user_input", "")
    if user_input:
        chapter_context = f"基于用户请求: {user_input}\n章节: {chapter_title}"
        if is_initiatives_phase:
            chapter_context += f"\n阶段: 推演阶段 (Initiatives Phase)"
    else:
        chapter_context = chapter_title

    # === Step 4: 知识缺口检测 (Agent化决策点) ===
    # 检测前序章节是否存在下一章需要的知识缺口
    if context_summary_update.get("context_summary") and llm_manager:
        gap_hint = _detect_knowledge_gaps(
            next_chapter_title=chapter_title,
            context_summary=context_summary_update["context_summary"],
            llm_manager=llm_manager
        )
        if gap_hint:
            chapter_context += f"\n\n[知识缺口提示]: {gap_hint}"
            logger.info(f"Detected knowledge gap for '{chapter_title}': {gap_hint[:80]}...")

    logger.debug(f"Initialized chapter '{chapter_title}' with clean workspace")
    logger.debug(f"Chapter question: {chapter_question}")

    return {
        "chapter_title": chapter_title,
        "chapter_question": chapter_question,
        "chapter_context": chapter_context,
        "chapter_scratchpad": chapter_scratchpad,
        "current_draft": current_draft,
        **context_summary_update
    }
