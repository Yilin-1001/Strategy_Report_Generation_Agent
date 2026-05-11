"""
Human Review node and routing function for the agent workflow.

This module handles:
- Processing human review decisions (approve, revise:data, revise:logic, revise:writing)
- Routing to appropriate next node based on decision
- Managing state updates (context_pool, scratchpad, chapter_index)
"""

import logging
from datetime import datetime
from typing import Dict, Any

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def should_continue(state: Dict[str, Any]) -> str:
    """
    增强的路由函数，支持两阶段战略推演架构的决策路由。

    支持的战略蓝图路由:
    - approve_blueprint → "prepare_chapter" (进入推演阶段)
    - revise_blueprint → "strategist" (重新生成蓝图)
    - 第三章完成 + approve → "strategist" (首次生成蓝图)
    - 蓝图已批准 + 继续 → "prepare_chapter" (进入第四章)

    Args:
        state: Current workflow state containing:
            - review_decision: The human's decision
            - current_chapter_index: Current chapter index
            - global_plan: List of chapter metadata (Dict)
            - strategic_blueprint: Strategic blueprint dict (optional)
            - context_pool: List of completed chapter contents

    Returns:
        Route name: "continue", "end", "researcher", "analyst", "writer", "strategist", "prepare_chapter"

    Enhanced routing logic:
        修订路由:
        - revise:data → "researcher"
        - revise:logic → "analyst"
        - revise:writing → "writer"

        战略蓝图路由 (新增):
        - approve_blueprint → "prepare_chapter" (进入推演阶段)
        - revise_blueprint → "strategist" (重新生成蓝图)

        章节流转:
        - approve + 诊断三章全部完成(context_pool>=3) → "strategist" (生成蓝图)
        - approve + 蓝图已批准 + 有更多章节 → "prepare_chapter"
        - approve + 最后一章 → "end"
        - finished → "end"
    """
    decision = state.get("review_decision")
    current_index = state.get("current_chapter_index", 0)
    global_plan = state.get("global_plan", [])
    blueprint = state.get("strategic_blueprint", {})
    context_pool = state.get("context_pool", [])

    # Handle revision routes (unchanged)
    if decision == "revise:data":
        return "researcher"
    elif decision == "revise:logic":
        return "analyst"
    elif decision == "revise:writing":
        return "writer"
    elif decision == "finished":
        return "end"

    # NEW: Handle strategic blueprint review routes
    if decision == "revise_blueprint":
        return "strategist"  # Regenerate blueprint

    if decision == "approve_blueprint":
        return "prepare_chapter"  # Enter initiatives phase

    # Normal chapter approval or phase transition
    if decision == "approve" or decision is None:
        # Only route to strategist when ALL 3 diagnosis chapters are complete.
        # Check context_pool length (not just index) because after Chapter 2
        # approval the index increments to 2, but Chapter 3 hasn't been
        # generated yet.  Using context_pool avoids a false-positive trigger.
        if current_index == 2 and len(context_pool) >= 3:
            # Check if blueprint exists and is approved
            if not blueprint or not blueprint.get("approved"):
                return "strategist"  # Generate or regenerate blueprint
            else:
                # Blueprint approved, index already incremented by approve_blueprint
                # Check: current_index points to the next chapter to process
                if current_index < len(global_plan):
                    return "prepare_chapter"
                else:
                    return "end"

        # Normal flow: current_index has been incremented to the next chapter
        # Check if that next chapter exists in global_plan
        if current_index < len(global_plan):
            return "prepare_chapter"
        else:
            return "end"

    # Default to end if no valid decision
    return "end"


def human_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    增强的人工审核节点，处理章节审核和战略蓝图审核。

    对于章节批准 (approve):
        - 将 current_draft 添加到 context_pool
        - 清空 scratchpad 为下一章节做准备
        - 增加章节索引

    对于战略蓝图批准 (approve_blueprint):
        - 设置 current_phase = "initiatives"
        - 标记 strategic_blueprint['approved'] = True
        - 清空 scratchpad 为第四章做准备

    对于蓝图修订 (revise_blueprint):
        - 保持状态不变
        - 路由回 strategist 节点

    对于章节修订 (revise:*):
        - 保持状态不变
        - 不增加章节索引

    特殊处理: 第三章 (index=2) 完成后
        - 暂停并等待战略蓝图批准
        - 不自动进入第四章

    Args:
        state: Current workflow state containing:
            - review_decision: The human's decision
            - current_chapter_index: Current chapter index
            - global_plan: List of chapter metadata (Dict with title, phase, etc.)
            - current_draft: The current chapter draft content
            - context_pool: List of completed chapter contents
            - chapter_scratchpad: Dict of scratchpad notes
            - strategic_blueprint: Strategic blueprint dict (optional)
            - current_phase: Current phase (diagnosis/initiatives)

    Returns:
        Updated state dictionary
    """
    decision = state.get("review_decision")
    current_chapter_index = state.get("current_chapter_index", 0)

    # Handle blueprint approval (NEW)
    if decision == "approve_blueprint":
        updated_state = {}
        updated_state["current_phase"] = "initiatives"

        # Update blueprint approval status (mutate in-place is OK for nested dicts)
        strategic_blueprint = state.get("strategic_blueprint", {})
        if strategic_blueprint:
            strategic_blueprint["approved"] = True
            updated_state["strategic_blueprint"] = strategic_blueprint

        # Clear scratchpad for next chapter
        updated_state["chapter_scratchpad"] = {}

        # Increment index to move past Chapter 3 into initiatives phase
        updated_state["current_chapter_index"] = current_chapter_index + 1

        logger.info("Strategic blueprint approved - entering INITIATIVES phase")
        return updated_state

    # Handle blueprint revision request (NEW)
    if decision == "revise_blueprint":
        # Route back to strategist without modifying state.
        # Do NOT return full state - context_pool uses operator.add and would duplicate.
        logger.info("Strategic blueprint revision requested - routing to strategist")
        return {}

    # Handle normal chapter approval
    if decision == "approve":
        global_plan = state.get("global_plan", [])
        current_draft = state.get("current_draft", "")
        context_pool = state.get("context_pool", [])

        # Check if this chapter was already approved (prevent duplicate additions)
        last_approved_index = state.get("_last_approved_index", -1)

        if current_chapter_index == last_approved_index:
            # This chapter was already approved, just increment index
            logger.info(f"Chapter {current_chapter_index} already approved, skipping context_pool addition")
            return {
                "chapter_scratchpad": {},
                "current_chapter_index": current_chapter_index + 1,
                "auto_revision_count": 0  # Reset on approve
            }

        # Get current chapter title from metadata (global_plan now contains Dict)
        if current_chapter_index < len(global_plan):
            chapter_metadata = global_plan[current_chapter_index]
            # Handle both Dict (new) and str (old) formats
            if isinstance(chapter_metadata, dict):
                chapter_title = chapter_metadata.get("title", f"Chapter {current_chapter_index + 1}")
            else:
                chapter_title = str(chapter_metadata)
        else:
            chapter_title = f"Chapter {current_chapter_index + 1}"

        if current_draft:
            # Build full chapter with title and content
            # ALWAYS enforce the correct title from global_plan, replacing any
            # LLM-generated title to prevent wrong chapter numbers (e.g. "第二章" for Chapter 3)
            draft_lines = current_draft.strip().split('\n')
            if draft_lines and draft_lines[0].strip().startswith('#'):
                # Strip the LLM-generated title and use the correct one from global_plan
                rest_of_draft = '\n'.join(draft_lines[1:]).strip()
                full_chapter = f"# {chapter_title}\n\n{rest_of_draft}"
            else:
                # Add title to draft
                full_chapter = f"# {chapter_title}\n\n{current_draft}"

            # Create updated state
            # IMPORTANT: context_pool uses Annotated[List[str], operator.add]
            # so we must return ONLY the new item, not the full accumulated list
            updated_state = {}
            updated_state["context_pool"] = [full_chapter]
            updated_state["chapter_scratchpad"] = {}

            # Save scratchpad knowledge for rolling context_summary compression
            # (prepare_chapter will read this before clearing)
            scratchpad = state.get("chapter_scratchpad", {})
            if scratchpad.get("key_facts"):
                updated_state["_pending_chapter_knowledge"] = {
                    "title": state.get("chapter_title", ""),
                    "key_facts": scratchpad.get("key_facts", []),
                    "insights": scratchpad.get("insights", [])
                }
            else:
                updated_state["_pending_chapter_knowledge"] = {}

            # Special handling: Don't increment after Chapter 2 (index=2) - wait for blueprint approval
            if current_chapter_index == 2:
                # Keep index at 2, wait for blueprint approval
                logger.info(f"Added chapter '{chapter_title}' (index {current_chapter_index}) to context_pool. "
                           f"Diagnosis phase complete - waiting for strategic blueprint approval.")
                updated_state["_last_approved_index"] = current_chapter_index
                updated_state["auto_revision_count"] = 0  # Reset on approve
                return updated_state
            else:
                # Normal flow: increment index
                updated_state["current_chapter_index"] = current_chapter_index + 1
                updated_state["_last_approved_index"] = current_chapter_index
                updated_state["auto_revision_count"] = 0  # Reset on approve

            logger.info(f"Added chapter '{chapter_title}' (index {current_chapter_index}) to context_pool")
            return updated_state
        else:
            # No draft, just increment index (or not if chapter 2)
            updated_state = {}
            updated_state["chapter_scratchpad"] = {}

            if current_chapter_index == 2:
                # Don't increment, wait for blueprint
                updated_state["_last_approved_index"] = current_chapter_index
                updated_state["auto_revision_count"] = 0  # Reset on approve
            else:
                updated_state["current_chapter_index"] = current_chapter_index + 1
                updated_state["_last_approved_index"] = current_chapter_index
                updated_state["auto_revision_count"] = 0  # Reset on approve

            return updated_state

    # Handle revise decisions: inject revision feedback into scratchpad for downstream nodes
    # The review_decision is already set via update_state() before resuming.
    # Note: cannot return context_pool (uses operator.add) or it would duplicate.
    current_count = state.get("auto_revision_count", 0)

    # Build revision feedback from human_feedback and LLM review
    chapter_scratchpad = state.get("chapter_scratchpad", {})
    human_feedback = state.get("human_feedback", {})
    # llm_review_result is at top-level state (set by reviewer node), not nested in human_feedback
    llm_review = state.get("llm_review_result") or {}

    # Start with LLM improvement_hints (may have been edited by user via _edit_review_fields)
    improvement_hints = dict(llm_review.get("improvement_hints", {}))

    # Merge user's manual comments into the appropriate role's hint
    # so downstream nodes pick them up through existing hint-reading logic
    user_comments = human_feedback.get("comments", "").strip()
    if user_comments:
        feedback_type = human_feedback.get("feedback_type", "")
        role_map = {
            "revise:data": "researcher",
            "revise:logic": "analyst",
            "revise:writing": "writer",
        }
        target_role = role_map.get(decision)
        if target_role:
            existing = improvement_hints.get(target_role, "")
            separator = " | " if existing else ""
            improvement_hints[target_role] = f"{existing}{separator}[用户指令] {user_comments}"
            logger.info(f"Merged user comments into {target_role} hint")

    # Build revision history: save the current draft as a historical version
    # This enables the frontend "修改历史" tab to show previous drafts
    revision_history = chapter_scratchpad.get("revision_history", [])
    current_draft = state.get("current_draft", "")

    # Create a new history entry for this revision
    history_entry = {
        "version": len(revision_history) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": _get_action_label(decision),
        "score": llm_review.get("score", 0),
        "issues": llm_review.get("issues", [])[:3],  # Keep top 3 issues for summary
        "draft_preview": current_draft[:600] if current_draft else "",
        "draft_full": current_draft,  # Store full draft for expandable view
        "decision": decision,
        "comments": user_comments,
    }
    revision_history.append(history_entry)

    chapter_scratchpad["revision_feedback"] = {
        "decision": decision,
        "comments": human_feedback.get("comments", ""),
        "issues": llm_review.get("issues", []),
        "score": llm_review.get("score", 0),
        "dimension_scores": llm_review.get("dimension_scores", {}),
        "improvement_hints": improvement_hints,
        "previous_draft_summary": state.get("current_draft", "")[:800]
    }
    chapter_scratchpad["revision_history"] = revision_history

    logger.info(f"Revision requested (decision={decision}), auto_revision_count: {current_count} -> {current_count + 1}")
    logger.info(f"Injected revision feedback: {chapter_scratchpad['revision_feedback']['issues'] or chapter_scratchpad['revision_feedback']['comments'][:100]}")
    logger.info(f"Revision history now has {len(revision_history)} entries")

    return {
        "auto_revision_count": current_count + 1,
        "chapter_scratchpad": chapter_scratchpad
    }


def _get_action_label(decision: str) -> str:
    """Convert decision code to human-readable action label."""
    action_map = {
        "revise:data": "补充数据",
        "revise:logic": "重新分析",
        "revise:writing": "重写内容",
        "approve": "批准通过",
        "finished": "终止报告",
    }
    return action_map.get(decision, decision)
