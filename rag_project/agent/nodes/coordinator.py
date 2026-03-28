"""
Coordinator Node - Generates structured chapter outline from user request

This is the first agent node in the workflow that analyzes the user's request
and generates a comprehensive chapter outline covering:
- Industry background
- Policy environment
- Status analysis
- Problems and challenges
- Strategic suggestions
"""

import json
import logging
from typing import Dict, List

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def coordinator_node(state: Dict, llm_manager) -> Dict:
    """
    Generate a structured chapter outline from the user's request.

    This node analyzes the user's input and creates a comprehensive plan
    with 5-8 chapters covering all essential aspects of the topic.

    Args:
        state: Current GraphState containing:
            - user_input: User's original request
        llm_manager: LLMManager instance configured for coordinator agent

    Returns:
        Dict with updates:
            - global_plan: List of chapter titles (5-8 chapters)
            - current_chapter_index: Set to 0

    Example:
        >>> state = {"user_input": "Generate a report on China's transportation industry"}
        >>> result = coordinator_node(state, llm_manager)
        >>> print(result["global_plan"])
        ['Industry Background', 'Policy Environment', 'Current Status Analysis',
         'Major Problems', 'Strategic Suggestions', 'Future Outlook']
    """
    user_input = state.get("user_input", "")

    logger.info(f"Coordinator node generating plan for: {user_input[:100]}...")

    # Generate prompt for outline creation
    prompt = _generate_outline_prompt(user_input)

    try:
        # Invoke LLM to generate outline
        response = llm_manager.invoke(prompt, temperature=0.3)

        # Parse JSON response
        global_plan = _parse_outline_response(response)

        # Validate the plan
        if not _validate_plan(global_plan):
            logger.warning("Generated plan failed validation, using fallback")
            global_plan = _get_default_plan(user_input)

        logger.info(f"Generated {len(global_plan)} chapters: {[c[:50] for c in global_plan]}")

    except Exception as e:
        logger.error(f"Error generating plan: {e}. Using fallback plan.")
        global_plan = _get_default_plan(user_input)

    return {
        "global_plan": global_plan,
        "current_chapter_index": 0
    }


def _generate_outline_prompt(user_input: str) -> str:
    """
    Generate a prompt for the LLM to create a structured outline.

    Args:
        user_input: User's original request

    Returns:
        Formatted prompt string
    """
    return f"""Based on the following user request, generate a comprehensive research report outline with 5-8 chapters.

User Request: {user_input}

Requirements:
1. Generate 5-8 chapter titles that comprehensively cover the topic
2. Each chapter should be clear and concise (ideally under 20 words)
3. The outline should cover these aspects:
   - Industry/Topic Background (overview and history)
   - Policy Environment (regulations, standards, government initiatives)
   - Current Status Analysis (market size, trends, key players)
   - Problems and Challenges (pain points, bottlenecks, issues)
   - Strategic Suggestions (solutions, recommendations, best practices)
   - Additional relevant topics (future outlook, case studies, etc.)

4. Return ONLY a valid JSON array of strings, like this:
   ["Chapter 1: Title", "Chapter 2: Title", ...]

5. Do not include any explanation or text outside the JSON array
6. Use professional, academic language appropriate for research reports

Generate the outline now:"""


def _parse_outline_response(response: str) -> List[str]:
    """
    Parse the LLM response to extract the chapter list.

    Args:
        response: Raw LLM response string

    Returns:
        List of chapter titles

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    # Try to extract JSON from response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    # Parse JSON
    try:
        outline = json.loads(response)

        # Ensure it's a list
        if isinstance(outline, list):
            # Ensure all items are strings
            return [str(item) for item in outline]
        else:
            raise ValueError("Response is not a list")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response content: {response[:500]}")
        raise ValueError(f"Invalid JSON response: {e}")


def _validate_plan(plan: List[str]) -> bool:
    """
    Validate that the generated plan meets requirements.

    Args:
        plan: List of chapter titles

    Returns:
        True if valid, False otherwise
    """
    # Check that plan is a list
    if not isinstance(plan, list):
        return False

    # Check minimum length (at least 3 chapters)
    if len(plan) < 3:
        logger.warning(f"Plan has only {len(plan)} chapters, minimum is 3")
        return False

    # Check maximum length (at most 10 chapters to be reasonable)
    if len(plan) > 10:
        logger.warning(f"Plan has {len(plan)} chapters, maximum is 10")
        return False

    # Check that all chapters are non-empty strings
    for i, chapter in enumerate(plan):
        if not isinstance(chapter, str) or len(chapter.strip()) == 0:
            logger.warning(f"Chapter {i} is not a valid string")
            return False

    return True


def _get_default_plan(user_input: str) -> List[str]:
    """
    Generate a fallback default plan when LLM fails.

    Args:
        user_input: User's original request (for context)

    Returns:
        Default chapter list
    """
    logger.info("Using default fallback plan")

    # Extract key topic from user input for personalization
    topic = "研究主题"  # Default
    if len(user_input) > 0:
        # Try to extract a meaningful topic (first few words)
        words = user_input.split()[:5]
        topic = " ".join(words)

    return [
        f"第一章：{topic}行业背景与发展历程",
        f"第二章：{topic}政策环境与监管框架",
        f"第三章：{topic}发展现状与市场规模分析",
        f"第四章：{topic}存在的主要问题与挑战",
        f"第五章：{topic}发展瓶颈与制约因素",
        f"第六章：{topic}发展战略与对策建议",
        f"第七章：{topic}未来展望与趋势预测"
    ]
