"""
Reviewer node for LLM-based draft evaluation.

This module handles:
- Evaluating chapter drafts using an independent reviewer model
- Generating multi-dimensional scores (topic relevance, analysis depth, etc.)
- Identifying issues and providing improvement hints for downstream nodes
- Setting llm_review_result in state for frontend display

Uses SiliconFlow Qwen model as independent reviewer to avoid self-evaluation hallucination.
"""

import json
import re
import logging
import os
from typing import Dict, Any

from openai import OpenAI

from rag_project.utils.logger import setup_logger
from rag_project.utils.config_loader import load_config
from rag_project.agent.llm_manager import LLMManager

logger = setup_logger(__name__)

# Initialize reviewer client (shared across all invocations)
_reviewer_client = None
_reviewer_model = None
_reviewer_max_retries = 0
_fallback_llm = None


def _init_reviewer():
    """Initialize the independent reviewer client."""
    global _reviewer_client, _reviewer_model, _reviewer_max_retries, _fallback_llm

    if _reviewer_client is not None:
        return  # Already initialized

    try:
        reviewer_config = load_config("config/agent_config.yaml").get("reviewer", {})
        reviewer_api_key = os.environ.get(reviewer_config.get("api_key_env", "SILICONFLOW_API_KEY"))
        reviewer_timeout = reviewer_config.get("timeout", 180)
        _reviewer_max_retries = reviewer_config.get("max_retries", 2)

        if reviewer_api_key:
            _reviewer_client = OpenAI(
                api_key=reviewer_api_key,
                base_url=reviewer_config.get("base_url", "https://api.siliconflow.cn/v1"),
                timeout=reviewer_timeout
            )
            _reviewer_model = reviewer_config.get("model", "Qwen/Qwen2.5-72B-Instruct")
            logger.info(f"Initialized independent reviewer: {_reviewer_model}")
        else:
            logger.warning("SILICONFLOW_API_KEY not set, using fallback coordinator LLM")
            _fallback_llm = LLMManager("coordinator")
    except Exception as e:
        logger.warning(f"Failed to init independent reviewer: {e}. Using fallback.")
        _fallback_llm = LLMManager("coordinator")


def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM评审节点，评估草稿质量并生成改进建议。

    评估6个维度：
    - 主题契合度 (topic_relevance): 0-15分
    - 分析深度 (analysis_depth): 0-20分
    - 写作专业度 (writing_quality): 0-15分
    - 引用充分性 (citation_sufficiency): 0-15分
    - 内容真实性 (groundedness): 0-20分
    - 上下文连贯性 (context_coherence): 0-15分

    Args:
        state: Current workflow state containing:
            - current_draft: The draft text to evaluate
            - chapter_title: Current chapter title
            - global_plan: List of chapter metadata
            - current_chapter_index: Index of current chapter
            - context_summary: Summary of previous chapters
            - strategic_blueprint: Strategic blueprint (for initiatives phase)

    Returns:
        Updated state with llm_review_result set
    """
    _init_reviewer()

    draft = state.get("current_draft", "")
    if not draft:
        logger.warning("No draft to review, returning empty result")
        return {"llm_review_result": None}

    # Get chapter metadata
    global_plan = state.get("global_plan", [])
    current_index = state.get("current_chapter_index", 0)
    chapter_meta = global_plan[current_index] if current_index < len(global_plan) else {}
    chapter_title = state.get("chapter_title", "")

    if isinstance(chapter_meta, dict):
        phase = chapter_meta.get("phase", "diagnosis")
        analysis_model = chapter_meta.get("analysis_model", "")
    else:
        phase = "diagnosis"
        analysis_model = ""

    # Stats
    chinese_chars = sum(1 for c in draft if '一' <= c <= '鿿')
    citations = len(re.findall(r'\[来源:', draft))

    # Get context for coherence check
    context_summary = state.get("context_summary", "")
    strategic_blueprint = state.get("strategic_blueprint", {})

    # Build prompt
    prompt = _build_review_prompt(
        draft=draft,
        chapter_title=chapter_title,
        phase=phase,
        analysis_model=analysis_model,
        citations=citations,
        chinese_chars=chinese_chars,
        context_summary=context_summary,
        strategic_blueprint=strategic_blueprint
    )

    try:
        response_text = _call_reviewer_llm(prompt)
        result = _parse_review_response(response_text, citations, chinese_chars)
        logger.info(f"LLM Review: score={result['score']}, suggestion={result['suggestion']}, issues={len(result['issues'])}")
        return {"llm_review_result": result}
    except Exception as e:
        logger.warning(f"LLM review failed: {e}. Returning default result.")
        return {
            "llm_review_result": {
                "score": 70,
                "dimension_scores": {},
                "issues": [],
                "suggestion": "approve",
                "improvement_hints": {}
            }
        }


def _call_reviewer_llm(prompt: str) -> str:
    """Call the reviewer LLM with retry logic."""
    global _reviewer_client, _reviewer_model, _reviewer_max_retries, _fallback_llm

    if _reviewer_client is not None:
        last_error = None
        for attempt in range(_reviewer_max_retries + 1):
            try:
                response = _reviewer_client.chat.completions.create(
                    model=_reviewer_model,
                    messages=[
                        {"role": "system", "content": "你是国企战略规划报告评审专家，严格按照JSON格式输出评审结果。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    extra_body={'enable_thinking': False}
                )
                logger.info(f"Review by {_reviewer_model} (attempt {attempt + 1})")
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < _reviewer_max_retries:
                    import time
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Review attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise last_error

    # Fallback to coordinator LLM
    logger.info("Using fallback coordinator LLM for review")
    return _fallback_llm.invoke(prompt, temperature=0.3, max_tokens=1024)


def _build_review_prompt(
    draft: str,
    chapter_title: str,
    phase: str,
    analysis_model: str,
    citations: int,
    chinese_chars: int,
    context_summary: str,
    strategic_blueprint: Dict
) -> str:
    """Build phase-aware review prompt."""

    base_criteria = f"""
## 评分维度与标准（满分100分，总分=各维度得分之和，≥70分为合格）

### 1. 主题契合度（15分）
章节内容是否紧扣标题和研究问题，是否存在偏离。
- 13-15分（优秀）：每段内容都直接服务于研究问题，无任何偏离
- 10-12分（良好）：基本扣题，偶有轻微延伸但不影响主线
- 7-9分（一般）：约30%内容偏离主题，存在与研究问题关联不强的段落
- 0-6分（差）：严重跑题，大部分内容与标题和问题无关

### 2. 分析深度（20分）
是否充分运用{analysis_model}分析框架，分析维度是否完整，是否有深入洞察。
- 17-20分（优秀）：分析模型所有维度完整覆盖，各维度有深入分析和独到洞察，逻辑链完整
- 13-16分（良好）：主要维度覆盖，分析有一定深度但个别维度较浅
- 8-12分（一般）：仅覆盖部分维度，分析停留在表面陈述，缺乏深入推理
- 0-7分（差）：未体现分析模型框架，内容仅为事实罗列，无分析逻辑

### 3. 写作专业度（15分）
是否符合国企公文语态和报告规范。
- 13-15分（优秀）：全文使用规范公文语态，结构清晰（总-分-总），语言凝练权威
- 10-12分（良好）：大部分符合公文风格，偶有口语化表述，结构基本完整
- 7-9分（一般）：语态不够统一，部分段落过于随意或学术化，结构有缺失
- 0-6分（差）：大量口语化表述，无国企公文特征，结构混乱

### 4. 引用充分性（15分）
引用数量与质量。当前引用{citations}个。
- 13-15分（优秀）：引用≥5个，全部为真实文件名，引用位置恰当且支撑对应论点
- 10-12分（良好）：引用4个，或≥5个但个别引用位置不够精准
- 7-9分（一般）：引用2-3个，支撑不够充分，或存在通用占位引用
- 0-6分（差）：引用0-1个，或引用全部为占位符，无法溯源

### 5. 内容真实性（20分）
数据、政策名称、事实陈述是否有据可查，幻觉程度评估。
- 17-20分（优秀）：所有数据和政策名称均可溯源至引用文档，无幻觉内容
- 13-16分（良好）：大部分数据可溯源，偶有无法验证的表述但不影响结论
- 8-12分（一般）：存在1-2处明显无法验证的数据或政策名称，可能为幻觉
- 0-7分（差）：多处数据或事实明显虚构，政策名称与实际不符，严重幻觉

### 6. 上下文连贯性（15分）
与前序章节的数据和结论是否一致，是否存在矛盾或重复。
"""

    if not context_summary:
        base_criteria += """
**注意**：本章节为首章，无前序章节内容，因此：
- 上下文连贯性维度自动计为满分（15分）
- 总分基数为85分（其他5个维度），最终换算为100分制
- 评分时只需评估前5个维度，第6维度固定15分
"""
    else:
        base_criteria += """
- 13-15分（优秀）：与前序章节数据完全一致，在不重复的前提下自然深化，逻辑连贯
- 10-12分（良好）：基本一致，偶有轻微不一致但已有说明
- 7-9分（一般）：存在1处明显数据矛盾或大段重复前序内容
- 0-6分（差）：多处数据矛盾，大量重复前序章节，或与前序结论冲突且无解释
"""

    # Phase-specific evaluation
    phase_specific = ""
    if phase == "diagnosis":
        phase_specific = f"""
## 诊断阶段特别评估

本章节属于诊断阶段，重点评估：
1. **信息覆盖度**: 检索信息是否准确覆盖章节主题
2. **分析模型完整性**: {analysis_model}的各维度是否都有分析内容
3. **数据准确性**: 政策名称、数据指标是否准确可查

前序章节摘要（用于评估连贯性）：
{context_summary[:500] if context_summary else "（无，为首章）"}
"""
    elif phase == "initiatives":
        mission = strategic_blueprint.get("mission", "") if strategic_blueprint else ""
        pillars = strategic_blueprint.get("strategic_pillars", []) if strategic_blueprint else []
        blueprint_info = f"""
核心使命: {mission}
战略支柱: {'; '.join(pillars[:3]) if pillars else '未设定'}
""" if strategic_blueprint else "（蓝图未生成或未批准）"

        phase_specific = f"""
## 推演阶段特别评估

本章节属于推演阶段，重点评估：
1. **蓝图对齐度**: 是否显式引用核心使命或战略支柱
2. **举措可执行性**: 提出的举措是否具体、可落地、有时限
3. **逻辑一致性**: 是否与前序诊断结论呼应

战略蓝图摘要：
{blueprint_info}

前序章节摘要：
{context_summary[:500] if context_summary else "（无）"}
"""

    length_check = f"""
## 篇幅检查
- 目标字数: 1000-1800字
- 当前字数: {chinese_chars}字
- {"✓ 符合要求" if 1000 <= chinese_chars <= 1800 else "✗ 需调整篇幅"}
"""

    return f"""你是国企战略规划报告评审专家。请对以下章节进行专业评审。

## 章节信息
标题: {chapter_title}
阶段: {phase}
分析模型: {analysis_model}
引用数量: {citations}个
中文字数: {chinese_chars}字

{base_criteria}

{phase_specific}

{length_check}

## 草稿内容（前2000字）
{draft[:2000]}

---

## 输出要求

请严格按以下JSON格式输出评审结果：

```json
{{"total_score": <总分1-100>,
    "dimension_scores": {{
        "topic_relevance": <0-15>,
        "analysis_depth": <0-20>,
        "writing_quality": <0-15>,
        "citation_sufficiency": <0-15>,
        "groundedness": <0-20>,
        "context_coherence": <0-15>
    }},
    "issues": ["问题1", "问题2", ...],
    "suggestion": "approve 或 revise:data 或 revise:logic 或 revise:writing",
    "improvement_hints": {{
        "researcher": "针对数据检索的具体改进建议",
        "analyst": "针对分析的具体改进建议",
        "writer": "针对写作的具体改进建议"
    }}
}}```

**评分标准**:
- 总分≥70分 → approve
- 总分50-69分 → revise:xxx（根据主要问题类型选择）
- 总分<50分 → revise:writing

只输出JSON，不要包含其他内容。
"""


def _parse_review_response(response_text: str, citations: int, chinese_chars: int) -> Dict:
    """Parse review response with multi-level fallback."""

    # Clean response
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Strategy 1: Direct parse
    result = _try_parse_json(text)
    if result:
        return result

    # Strategy 2: Extract JSON block
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        result = _try_parse_json(brace_match.group(0))
        if result:
            return result

    # Strategy 3: Fix common errors
    fixed = brace_match.group(0) if brace_match else text
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    fixed = re.sub(r'//.*?\n', '\n', fixed)
    result = _try_parse_json(fixed)
    if result:
        return result

    # Strategy 4: Regex extract
    result = _try_regex_extract(text)
    if result:
        return result

    # Strategy 5: Old format fallback
    logger.warning("All JSON parsing strategies failed, using fallback")
    return {
        "score": 70,
        "dimension_scores": {},
        "issues": [],
        "suggestion": "approve",
        "improvement_hints": {}
    }


def _try_parse_json(text: str) -> Dict:
    """Try to parse JSON and convert to standard format."""
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if "total_score" not in result and "dimension_scores" not in result:
        return None

    total_score = result.get("total_score", 70)
    score = max(0, min(100, total_score))

    dimension_scores = result.get("dimension_scores", {})

    issues = result.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    suggestion = result.get("suggestion", "approve").lower()
    if suggestion not in ("approve", "revise:data", "revise:logic", "revise:writing"):
        suggestion = "approve" if total_score >= 70 else "revise:writing"

    improvement_hints = result.get("improvement_hints", {})
    if not isinstance(improvement_hints, dict):
        improvement_hints = {}

    return {
        "score": score,
        "dimension_scores": dimension_scores,
        "issues": issues,
        "suggestion": suggestion,
        "improvement_hints": improvement_hints
    }


def _try_regex_extract(text: str) -> Dict:
    """Extract fields using regex from non-standard JSON."""

    ts_match = re.search(r'"total_score"\s*:\s*(\d+)', text)
    if not ts_match:
        return None

    total_score = int(ts_match.group(1))
    score = max(0, min(100, total_score))

    dim_keys = ["topic_relevance", "analysis_depth", "writing_quality",
                "citation_sufficiency", "groundedness", "context_coherence"]
    dimension_scores = {}
    for key in dim_keys:
        m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
        if m:
            dimension_scores[key] = int(m.group(1))

    issues = []
    issues_match = re.search(r'"issues"\s*:\s*\[([^\]]*)\]', text)
    if issues_match:
        issues = [s.strip().strip('"').strip("'") for s in issues_match.group(1).split(',') if s.strip().strip('"')]

    sug_match = re.search(r'"suggestion"\s*:\s*"([^"]+)"', text)
    suggestion = sug_match.group(1).lower() if sug_match else ("approve" if score >= 70 else "revise:writing")

    improvement_hints = {}
    for role in ["researcher", "analyst", "writer"]:
        hint_match = re.search(rf'"{role}"\s*:\s*"([^"]+)"', text)
        if hint_match:
            improvement_hints[role] = hint_match.group(1)

    logger.info(f"Regex extraction: score={score}, dims={len(dimension_scores)}, hints={len(improvement_hints)}")

    return {
        "score": score,
        "dimension_scores": dimension_scores,
        "issues": issues,
        "suggestion": suggestion,
        "improvement_hints": improvement_hints
    }