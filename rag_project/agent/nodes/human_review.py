"""
Human Review node and routing function for the agent workflow.

This module handles:
- Processing human review decisions (approve, revise:data, revise:logic, revise:writing)
- Routing to appropriate next node based on decision
- Managing state updates (context_pool, scratchpad, chapter_index)
"""

from typing import Dict, Any


def should_continue(state: Dict[str, Any]) -> str:
    """
    Determine the next routing destination based on review decision.

    Args:
        state: Current workflow state containing:
            - review_decision: The human's decision (approve, revise:data, revise:logic, revise:writing, finished)
            - chapter_index: Current chapter index
            - chapter_titles: List of all chapter titles

    Returns:
        Route name: "continue", "end", "researcher", "analyst", or "writer"

    Routing logic:
        - approve + more chapters ��� "continue"
        - approve + last chapter → "end"
        - revise:data → "researcher"
        - revise:logic → "analyst"
        - revise:writing → "writer"
        - finished → "end"
    """
    decision = state.get("review_decision")

    # Handle revision routes
    if decision == "revise:data":
        return "researcher"
    elif decision == "revise:logic":
        return "analyst"
    elif decision == "revise:writing":
        return "writer"
    elif decision == "finished":
        return "end"

    # Handle approve routes
    if decision == "approve":
        chapter_index = state.get("chapter_index", 0)
        chapter_titles = state.get("chapter_titles", [])

        # Check if there are more chapters to process
        if chapter_index + 1 < len(chapter_titles):
            return "continue"
        else:
            return "end"

    # Default to end if no valid decision
    return "end"


def human_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process human review decision and update state accordingly.

    For approve decisions:
        - Add current_draft to context_pool as a full chapter (with title)
        - Clear scratchpad for next chapter
        - Increment chapter_index

    For revise decisions:
        - Keep all state as-is
        - Don't increment chapter_index

    Args:
        state: Current workflow state containing:
            - review_decision: The human's decision
            - chapter_index: Current chapter index
            - chapter_titles: List of all chapter titles
            - current_draft: The current chapter draft content
            - context_pool: List of completed chapter contents
            - scratchpad: List of scratchpad notes

    Returns:
        Updated state dictionary
    """
    decision = state.get("review_decision")

    # Handle approve: update state for next chapter
    if decision == "approve":
        chapter_index = state.get("chapter_index", 0)
        chapter_titles = state.get("chapter_titles", [])
        current_draft = state.get("current_draft", "")
        context_pool = state.get("context_pool", [])

        # Get current chapter title
        if chapter_index < len(chapter_titles):
            chapter_title = chapter_titles[chapter_index]
        else:
            chapter_title = f"Chapter {chapter_index + 1}"

        # Build full chapter with title and content
        full_chapter = f"# {chapter_title}\n\n{current_draft}"

        # Create updated state
        updated_state = state.copy()
        updated_state["context_pool"] = context_pool + [full_chapter]
        updated_state["scratchpad"] = []
        updated_state["chapter_index"] = chapter_index + 1

        return updated_state

    # Handle revise decisions: keep state as-is
    # Don't modify anything, let the revising node work with current state
    return state
