"""
Prepare Chapter Node - Initializes chapter state and enforces state isolation

This helper node is called at the beginning of each chapter iteration to:
1. Recompute global context_summary from all completed chapters (context_pool)
2. Set the current chapter title from the global plan
3. Clear the chapter scratchpad (enforcing state isolation between chapters)
4. Clear the current draft (ensuring fresh start)
5. Log progress with chapter number (X of Y)

Memory architecture:
- context_pool: accumulates approved chapter texts (operator.add, append-only)
- context_summary: recomputed from context_pool each time, budget scales with chapters
"""

import logging
from typing import Dict

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def _calc_summary_budget(completed_chapters: int) -> tuple:
    """
    按已完成章节数比例计算摘要预算（阶段感知分级策略）。

    诊断阶段 (ch1-3): 温和增长 400 → 700 → 1000
    推演阶段 (ch4-8): 加速增长，携带诊断摘要 + 蓝图 + 前序推演

    Args:
        completed_chapters: 已完成的章节数（context_pool长度）

    Returns:
        (target_chars, max_tokens) 元组
    """
    if completed_chapters <= 0:
        return 0, 0

    diagnosis_end = 3

    if completed_chapters <= diagnosis_end:
        target_chars = 300 + completed_chapters * 350
    else:
        base_from_diagnosis = 1350
        initiatives_count = completed_chapters - diagnosis_end
        target_chars = base_from_diagnosis + initiatives_count * 450

    target_chars = min(target_chars, 3500)
    max_tokens = int(target_chars * 0.65)
    return target_chars, max_tokens


def _full_recompute_summary(context_pool: list, llm_manager,
                            current_chapter_index: int) -> str:
    """
    从所有已完成章节原文（context_pool）重新计算全局摘要。

    每次都从原始章节文本出发，避免滚动压缩的级联信息损失。
    摘要预算按已完成章节数比例增长。

    Args:
        context_pool: 已审核通过的章节原文列表
        llm_manager: LLM 实例
        current_chapter_index: 当前章节索引（用于日志）

    Returns:
        重算后的全局摘要
    """
    completed = len(context_pool)
    if completed == 0:
        return ""

    target_chars, max_tokens = _calc_summary_budget(completed)
    logger.info(f"Recomputing summary from {completed} completed chapters "
                f"(budget: {target_chars} chars / {max_tokens} tokens)")

    # 拼接所有已完成章节原文
    all_chapters = "\n\n".join(
        f"【第{i+1}章】\n{chapter}"
        for i, chapter in enumerate(context_pool)
    )

    prompt = f"""基于以下{completed}个已完成章节的完整原文，生成一份全局摘要。

要求:
1. 按章节顺序组织，标注每个结论的来源章节（如"第二章指出..."）
2. 保留所有具体数字（金额、比例、增长率、投资额等）
3. 保留政策名称、法规文件名和战略定位表述
4. 保留关键论证结论和核心洞察
5. 删除冗余和重复，确保信息密度
6. 控制在{target_chars}字以内
7. 使用中文

已完成章节:
{all_chapters}

输出摘要:"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=max_tokens)
        summary = response.strip()
        logger.info(f"Recomputed summary: {len(summary)} chars "
                    f"from {completed} chapters (target: {target_chars})")
        return summary
    except Exception as e:
        logger.error(f"Full recompute failed: {e}")
        # 回退：截断拼接
        fallback = "\n".join(chapter[:200] for chapter in context_pool)
        return fallback[:target_chars]


def _detect_knowledge_gaps(
    next_chapter_title: str,
    context_summary: str,
    llm_manager
) -> str:
    """
    检测前序章节是否存在下一章需要的知识缺口。

    Args:
        next_chapter_title: 下一章标题
        context_summary: 前序章节的摘要
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

    摘要策略: 从 context_pool（所有已完成章节原文）全量重算 context_summary，
    避免滚动压缩的级联信息损失。预算按已完成章节数比例增长。

    推演阶段特殊处理 (initiatives phase):
    - 如果当前章节 phase == "initiatives" (第4-8章)
    - 且 strategic_blueprint 存在并已批准
    - 将自动注入战略蓝图上下文到 chapter_scratchpad

    Args:
        state: Current GraphState
        llm_manager: LLM instance for summary recomputation

    Returns:
        Dict with updates:
            - chapter_title: Title from global_plan
            - chapter_question: Research question
            - chapter_context: Context for this chapter
            - chapter_scratchpad: Empty dict OR with strategic_blueprint
            - current_draft: Empty string (fresh start)
            - context_summary: Recomputed from all completed chapters
    """
    # === Step 1: 从 context_pool 全量重算摘要 ===
    context_pool = state.get("context_pool", [])
    current_index = state.get("current_chapter_index", 0)
    context_summary_update = {}

    if context_pool and llm_manager:
        compressed = _full_recompute_summary(
            context_pool=context_pool,
            llm_manager=llm_manager,
            current_chapter_index=current_index
        )
        context_summary_update = {"context_summary": compressed}
        logger.info(f"Recomputed context_summary ({len(compressed)} chars) "
                    f"from {len(context_pool)} completed chapters")

    # === Step 2: 设置新章节状态 ===
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
