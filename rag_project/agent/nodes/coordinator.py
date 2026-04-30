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
    生成基于两阶段战略推演架构的固定8章大纲。

    该节点不再使用LLM动态生成，而是返回包含阶段(phase)和分析模型(analysis_model)
    元数据的固定8章结构，适用于省属国企政策导向型战略规划报告。

    两阶段架构:
    - Diagnosis阶段（第1-3章）: 宏观环境、区域战略、内部诊断
    - Initiatives阶段（第4-8章）: 战略思路、主业举措、创新驱动、产业协同、治理效能

    Args:
        state: Current GraphState containing:
            - user_input: User's original request
        llm_manager: LLMManager instance (not used, kept for interface compatibility)

    Returns:
        Dict with updates:
            - global_plan: List[Dict] with keys: title, phase, analysis_model, index
            - current_chapter_index: Set to 0
            - current_phase: Set to "diagnosis"

    Example:
        >>> state = {"user_input": "生成江西交投集团战略规划报告"}
        >>> result = coordinator_node(state, llm_manager)
        >>> print(result["global_plan"][0])
        {'title': '第一章：宏观政策环境与时代要求',
         'phase': 'diagnosis',
         'analysis_model': 'PEST模型 (侧重P-政策与E-经济维度)',
         'index': 0}
    """
    user_input = state.get("user_input", "")

    logger.info(f"Coordinator node generating fixed 8-chapter strategic plan for: {user_input[:100]}...")

    # 固定的8章结构，包含phase和analysis_model元数据
    global_plan = [
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
            "analysis_model": "波特五力模型与SWOT分析 (强制要求在分析结尾输出结构化的SWOT矩阵)",
            "index": 2
        },
        {
            "title": "第四章：总体战略思路与政策响应目标",
            "phase": "initiatives",
            "analysis_model": "平衡计分卡(BSC)模型 (从财务、客户/民生、内部运营、学习与成长四个维度设定目标)",
            "index": 3
        },
        {
            "title": "第五章：主责主业：高质量建设与保通保畅举措",
            "phase": "initiatives",
            "analysis_model": "BCG波士顿矩阵 (将主业作为'现金牛'业务，侧重精益化与稳健回报)",
            "index": 4
        },
        {
            "title": "第六章：创新驱动：绿色低碳与智慧交投建设",
            "phase": "initiatives",
            "analysis_model": "安索夫矩阵 (将创新业务作为新产品/新市场拓展，侧重第二增长曲线)",
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
            "analysis_model": "麦肯锡7S模型 (从结构、制度、风格、员工、技能等维度构建组织保障)",
            "index": 7
        }
    ]

    logger.info(f"Generated fixed 8-chapter strategic plan with {len([c for c in global_plan if c['phase'] == 'diagnosis'])} diagnosis chapters and {len([c for c in global_plan if c['phase'] == 'initiatives'])} initiatives chapters")

    return {
        "global_plan": global_plan,
        "current_chapter_index": 0,
        "current_phase": "diagnosis"
    }


def _generate_outline_prompt(user_input: str) -> str:
    """
    Generate a prompt for the LLM to create a structured outline.

    Args:
        user_input: User's original request

    Returns:
        Formatted prompt string
    """
    return f"""根据以下用户请求，生成一个全面的研究报告大纲，包含5-8个章节。

用户请求: {user_input}

要求:
1. 生成5-8个中文章节标题，全面覆盖主题
2. 每个章节标题应清晰简洁（最好不超过20字）
3. 大纲应涵盖以下方面：
   - 行业/主题背景（概述和历史）
   - 政策环境（法规、标准、政府举措）
   - 发展现状分析（市场规模、趋势、关键参与者）
   - 问题与挑战（痛点、瓶颈、问题）
   - 战略建议（解决方案、建议、最佳实践）
   - 其他相关主题（未来展望、案例研究等）

4. 仅返回有效的JSON字符串数组，格式如下：
   ["第一章：标题", "第二章：标题", ...]

5. 不要在JSON数组外包含任何解释或文本
6. 使用适合研究报告的专业、学术语言

现在生成大纲:"""


def _parse_outline_response(response: str) -> List[Dict]:  # Kept for compatibility, now returns List[Dict]
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


def _validate_plan(plan: List[Dict]) -> bool:  # Kept for compatibility, now validates List[Dict]
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


def _get_default_plan(user_input: str) -> List[Dict]:  # Kept for compatibility, now returns List[Dict]
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
