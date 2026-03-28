"""
Prepare Chapter Node - Initializes chapter state and enforces state isolation

This helper node is called at the beginning of each chapter iteration to:
1. Set the current chapter title from the global plan
2. Clear the chapter scratchpad (enforcing state isolation between chapters)
3. Clear the current draft (ensuring fresh start)
4. Log progress with chapter number (X of Y)

This is critical for maintaining the "阅后即焚" (burn after reading) principle
where chapter_scratchpad serves as a temporary workspace that is reset for each chapter.
"""

import logging
from typing import Dict

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_chapter_node(state: Dict) -> Dict:
    """
    Prepare state for the current chapter by setting title and clearing workspace.

    This node enforces state isolation by ensuring that each chapter starts with
    a clean scratchpad and empty draft, preventing data leakage between chapters.

    Args:
        state: Current GraphState containing:
            - global_plan: List of chapter titles
            - current_chapter_index: Index of current chapter (0-based)

    Returns:
        Dict with updates:
            - chapter_title: Title from global_plan[current_chapter_index]
            - chapter_scratchpad: Empty dict {} (CRITICAL: state isolation)
            - current_draft: Empty string "" (fresh start)

    Example:
        >>> state = {
        ...     "global_plan": ["Chapter 1", "Chapter 2", "Chapter 3"],
        ...     "current_chapter_index": 0,
        ...     "chapter_scratchpad": {"old": "data"},
        ...     "current_draft": "old content"
        ... }
        >>> result = prepare_chapter_node(state)
        >>> result["chapter_title"]
        'Chapter 1'
        >>> result["chapter_scratchpad"]
        {}
        >>> result["current_draft"]
        ''
    """
    global_plan = state.get("global_plan", [])
    current_index = state.get("current_chapter_index", 0)
    total_chapters = len(global_plan)

    # Get current chapter title from global plan
    chapter_title = global_plan[current_index] if current_index < len(global_plan) else ""

    # Log progress with chapter number (X of Y)
    chapter_num = current_index + 1
    logger.info(f"Preparing Chapter {chapter_num} of {total_chapters}: {chapter_title}")

    # CRITICAL: Clear chapter_scratchpad to enforce state isolation
    # This ensures each chapter starts with a clean workspace
    chapter_scratchpad = {}

    # Clear current_draft to ensure fresh start
    current_draft = ""

    logger.debug(f"Initialized chapter '{chapter_title}' with clean workspace")

    return {
        "chapter_title": chapter_title,
        "chapter_scratchpad": chapter_scratchpad,
        "current_draft": current_draft
    }
