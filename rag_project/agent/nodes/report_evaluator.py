"""
Report Evaluator node for full-report quality assessment.

Uses the same 5-dimension scoring rubric as the ablation experiment evaluator
to evaluate the complete final report after all chapters are assembled.

Dimensions (each 0-20 points, total 0-100):
1. 方法论运用与分析框架严谨度
2. 战略一致性与外部环境契合度
3. 逻辑连贯性与战略闭环思维
4. 创新性与前瞻洞察力
5. 隐性约束洞察与组织治理深度
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

# 全文评审客户端（复用 reviewer 的配置）
_evaluator_client = None
_evaluator_model = None
_fallback_llm = None

# 全文5维度评分标准（与消融实验 evaluator.py 一致）
REPORT_SCORING_SYSTEM_PROMPT = """你是一位拥有20年经验的国际顶尖战略咨询合伙人，专门评估战略规划报告。

你必须严格按JSON格式输出评分，不要输出任何其他文字、分析或解释。不要使用markdown代码块。直接输出纯JSON。

## 评分维度（每项0-20分，允许0.5分精度）

### 维度一：方法论运用与分析框架严谨度 (0-20)
评估各章节是否正确、深入地运用了指定的战略分析模型（PEST/SWOT/BCG/五力/BSC/安索夫/7S/ESG等）。
- 18-20: 模型框架运用精准、维度完整、各要素间逻辑衔接紧密
- 14-17: 模型框架基本正确，但维度覆盖不完整或分析深度不够
- 10-13: 模型框架使用有偏差，维度缺失或混淆
- 0-9: 未使用指定模型或使用完全错误

### 维度二：战略一致性与外部环境契合度 (0-20)
评估报告是否紧密结合政策导向（交通强国、交通强省、国企改革），对外部环境认知是否准确深入。
- 18-20: 政策引用精准、外部环境认知深刻、战略定位高度契合
- 14-17: 政策引用较准确但深度不足，或战略定位有偏差
- 10-13: 政策引用泛泛，缺乏针对性分析
- 0-9: 脱离政策背景，战略定位模糊

### 维度三：逻辑连贯性与战略闭环思维 (0-20)
评估诊断阶段→战略推演→实施举措之间是否形成完整闭环，章节间逻辑是否连贯。
- 18-20: 诊断-战略-举措完美闭环，问题-对策一一对应，章节间高度连贯
- 14-17: 基本形成闭环，但部分对策缺乏诊断依据，或章节间有脱节
- 10-13: 闭环不完整，诊断与举措脱节
- 0-9: 各章节独立无关联，无闭环思维

### 维度四：创新性与前瞻洞察力 (0-20)
- 18-20: 提出多维度原创性战略洞察，前瞻判断有力，创新举措兼具突破性与可行性
- 14-17: 有一定的原创洞察和前瞻分析，但创新深度不足或部分流于常规
- 10-13: 以文档信息复述为主，缺乏独立思考和前瞻性判断
- 0-9: 完全依赖文档搬运，无任何原创洞察或前瞻分析

### 维度五：隐性约束洞察与组织治理深度 (0-20)
评估报告是否识别了组织内部的隐性约束（利益相关方博弈、变革阻力、文化惯性），治理策略是否深入。
- 18-20: 深刻洞察组织摩擦力，提出具体可行的变革管理策略
- 14-17: 识别了部分组织约束，但应对策略偏宏观
- 10-13: 对组织约束的认知停留在表面
- 0-9: 忽视组织内部约束，治理建议空泛"""


def _init_evaluator():
    """Initialize the report evaluator client (reuse reviewer config)."""
    global _evaluator_client, _evaluator_model, _fallback_llm

    if _evaluator_client is not None:
        return

    try:
        reviewer_config = load_config("config/agent_config.yaml").get("reviewer", {})
        api_key = os.environ.get(reviewer_config.get("api_key_env", "SILICONFLOW_API_KEY"))
        timeout = reviewer_config.get("timeout", 180)

        if api_key:
            _evaluator_client = OpenAI(
                api_key=api_key,
                base_url=reviewer_config.get("base_url", "https://api.siliconflow.cn/v1"),
                timeout=timeout
            )
            _evaluator_model = reviewer_config.get("model", "Pro/moonshotai/Kimi-K2.5")
            logger.info(f"Initialized report evaluator: {_evaluator_model}")
        else:
            logger.warning("API key not set, using fallback LLM")
            _fallback_llm = LLMManager("coordinator")
    except Exception as e:
        logger.warning(f"Failed to init evaluator: {e}. Using fallback.")
        _fallback_llm = LLMManager("coordinator")


def _extract_chapter_content(report_text: str) -> str:
    """Extract only chapter content from report, stripping structural overhead.

    Removes: cover page, TOC, executive summary, blueprint appendix, consistency notes.
    Keeps only the actual chapter body text for evaluation.
    """
    lines = report_text.split('\n')
    chapter_lines = []
    in_chapter = False
    in_appendix = False

    for line in lines:
        stripped = line.strip()

        # Skip until we hit the first chapter heading
        if not in_chapter:
            if re.match(r'^#\s*第[一二三四五六七八九十\d]+章', stripped):
                in_chapter = True
                chapter_lines.append(line)
            continue

        # Stop at appendix section
        if stripped.startswith('#') and ('附录' in stripped or 'Appendix' in stripped
                                          or '战略蓝图' in stripped):
            in_appendix = True
            continue

        # Stop at consistency review notes
        if '一致性审查' in stripped or 'consistency' in stripped.lower():
            break

        if in_appendix:
            continue

        # Skip section separators
        if stripped == '---':
            continue

        chapter_lines.append(line)

    content = '\n'.join(chapter_lines).strip()

    # If extraction yielded too little, fall back to original
    if len(content) < len(report_text) * 0.3:
        return report_text

    return content


def _build_eval_prompt(report_text: str) -> str:
    """Build the evaluation prompt for the full report."""
    # Extract chapter content only
    chapter_content = _extract_chapter_content(report_text)

    # Truncate to fit context window
    if len(chapter_content) > 25000:
        eval_text = chapter_content[:18000] + "\n\n...[中间章节内容省略]...\n" + chapter_content[-7000:]
    else:
        eval_text = chapter_content

    return f"""请评估以下战略规划报告，直接输出JSON，不要输出其他任何文字。

报告内容:
{eval_text}

---

输出格式（严格遵守）:
{{"d1_score":0,"d1_analysis":"详细评语","d2_score":0,"d2_analysis":"详细评语","d3_score":0,"d3_analysis":"详细评语","d4_score":0,"d4_analysis":"详细评语","d5_score":0,"d5_analysis":"详细评语","total_score":0,"suggestions":"详细改进建议"}}

说明:
- d1=方法论, d2=战略一致, d3=逻辑闭环, d4=创新前瞻, d5=组织治理
- 每项0-20分，total=五项之和
- 每个维度的analysis必须写100-200字的详细评语，包含：(1)各章节在该维度的具体表现 (2)做得好的地方 (3)存在的不足 (4)具体的改进方向
- suggestions给出200字左右的总体改进建议，按优先级排序"""


def _call_evaluator_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the evaluator LLM."""
    global _evaluator_client, _evaluator_model, _fallback_llm

    if _evaluator_client is not None:
        try:
            response = _evaluator_client.chat.completions.create(
                model=_evaluator_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=8192,
                extra_body={"enable_thinking": False}
            )
            logger.info(f"Report evaluation by {_evaluator_model}")
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Evaluator LLM call failed: {e}. Using fallback.")

    logger.info("Using fallback LLM for report evaluation")
    return _fallback_llm.invoke(user_prompt, temperature=0.3, max_tokens=4096)


def _parse_eval_response(response_text: str) -> Dict:
    """Parse evaluation response into structured dict."""
    text = response_text.strip()

    # Remove <think/> blocks
    text = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', text, flags=re.DOTALL)

    # Clean markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct JSON parse
    try:
        raw = json.loads(text)
        return _normalize_eval_result(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            return _normalize_eval_result(raw)
        except json.JSONDecodeError:
            pass

    # Regex fallback
    return _regex_extract_eval(text)


def _normalize_eval_result(raw: Dict) -> Dict:
    """Normalize parsed JSON into standard format."""
    dim_mapping = {
        "d1": ("methodology", "方法论运用与分析框架严谨度"),
        "d2": ("strategic_alignment", "战略一致性与外部环境契合度"),
        "d3": ("logical_coherence", "逻辑连贯性与战略闭环思维"),
        "d4": ("innovation_insight", "创新性与前瞻洞察力"),
        "d5": ("organizational_governance", "隐性约束洞察与组织治理深度"),
    }

    dimension_scores = {}
    for key, (eng_name, cn_name) in dim_mapping.items():
        score = raw.get(f"{key}_score", 0)
        analysis = raw.get(f"{key}_analysis", "")

        try:
            score = max(0, min(20, float(score)))
        except (ValueError, TypeError):
            score = 0

        dimension_scores[eng_name] = {
            "score": score,
            "label": cn_name,
            "analysis": analysis,
        }

    total_score = raw.get("total_score", 0)
    if not total_score:
        total_score = sum(v["score"] for v in dimension_scores.values())

    try:
        total_score = max(0, min(100, float(total_score)))
    except (ValueError, TypeError):
        total_score = sum(v["score"] for v in dimension_scores.values())

    suggestions = raw.get("suggestions", "")
    if not isinstance(suggestions, str):
        suggestions = str(suggestions) if suggestions else ""

    return {
        "total_score": int(total_score),
        "dimension_scores": dimension_scores,
        "suggestions": suggestions,
    }


def _regex_extract_eval(text: str) -> Dict:
    """Fallback regex extraction."""
    dim_mapping = {
        "d1": ("methodology", "方法论运用与分析框架严谨度"),
        "d2": ("strategic_alignment", "战略一致性与外部环境契合度"),
        "d3": ("logical_coherence", "逻辑连贯性与战略闭环思维"),
        "d4": ("innovation_insight", "创新性与前瞻洞察力"),
        "d5": ("organizational_governance", "隐性约束洞察与组织治理深度"),
    }

    dimension_scores = {}
    for key, (eng_name, cn_name) in dim_mapping.items():
        score_match = re.search(rf'"{key}_score"\s*:\s*([\d.]+)', text)
        analysis_match = re.search(rf'"{key}_analysis"\s*:\s*"([^"]*)"', text)
        score = float(score_match.group(1)) if score_match else 0
        analysis = analysis_match.group(1) if analysis_match else ""
        dimension_scores[eng_name] = {
            "score": max(0, min(20, score)),
            "label": cn_name,
            "analysis": analysis,
        }

    total_match = re.search(r'"total_score"\s*:\s*([\d.]+)', text)
    total_score = float(total_match.group(1)) if total_match else sum(v["score"] for v in dimension_scores.values())

    suggestions_match = re.search(r'"suggestions"\s*:\s*"([^"]*)"', text)
    suggestions = suggestions_match.group(1) if suggestions_match else ""

    logger.info(f"Regex extraction: total={total_score}")

    return {
        "total_score": int(max(0, min(100, total_score))),
        "dimension_scores": dimension_scores,
        "suggestions": suggestions,
    }


def report_evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    全文评审节点，使用消融实验的5维度评分标准评估完整报告。

    在 archiver 生成最终报告后执行，输出存入 state["report_evaluation"]。

    5个维度（每项0-20分，满分100）：
    - 维度一：方法论运用与分析框架严谨度
    - 维度二：战略一致性与外部环境契合度
    - 维度三：逻辑连贯性与战略闭环思维
    - 维度四：创新性与前瞻洞察力
    - 维度五：隐性约束洞察与组织治理深度
    """
    _init_evaluator()

    final_report = state.get("final_report", "")
    if not final_report:
        logger.warning("No final_report to evaluate, returning empty result")
        return {"report_evaluation": None}

    user_prompt = _build_eval_prompt(final_report)

    try:
        response_text = _call_evaluator_llm(REPORT_SCORING_SYSTEM_PROMPT, user_prompt)
        result = _parse_eval_response(response_text)
        logger.info(f"Report evaluation: total_score={result['total_score']}")
        return {"report_evaluation": result}
    except Exception as e:
        logger.warning(f"Report evaluation failed: {e}. Returning default result.")
        return {
            "report_evaluation": {
                "total_score": 0,
                "dimension_scores": {},
                "suggestions": f"评审失败: {str(e)}"
            }
        }