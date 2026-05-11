"""
Reviewer node for LLM-based chapter-level draft evaluation.

Chapter-level 5-dimension scoring rubric (each 0-20 points, total 0-100):
1. 模型运用与框架完整性
2. 数据支撑与证据质量
3. 内部逻辑与结构清晰度
4. 内容深度与专业水准
5. 写作质量与规范表达
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

_reviewer_client = None
_reviewer_model = None
_reviewer_max_retries = 0
_fallback_llm = None

# 章节专属评估指引：每章的核心期望
_CHAPTER_EXPECTATIONS = {
    0: "本章使用PEST模型，期望政策(P)/经济(E)/社会(S)/技术(T)四维度完整覆盖，政策维度需引用具体法规文件，经济维度需有数据支撑。",
    1: "本章侧重省级政策承接与区域占位分析，期望将'交通强国'战略与省级'交通强省'目标逐级衔接，体现区域差异化定位。",
    2: "本章使用波特五力与SWOT模型，期望在分析结尾输出结构化的SWOT矩阵表格，五力分析需覆盖供应商/买方/新进入者/替代品/行业内竞争。",
    3: "本章使用平衡计分卡(BSC)模型，期望从财务、客户/民生、内部运营、学习与成长四个维度设定可量化目标与KPI。",
    4: "本章使用BCG波士顿矩阵，期望将主业明确为'现金牛'业务定位，提出精益化运营与稳健回报的具体举措。",
    5: "本章使用安索夫矩阵，期望绘制创新路线图（市场渗透/产品开发/市场开发/多元化），明确第二增长曲线的路径与时间表。",
    6: "本章使用产业链协同与ESG社会责任模型，期望在产业协同中体现'交通+'融合模式，ESG维度需覆盖环境/社会/治理三方面。",
    7: "本章使用麦肯锡7S模型，期望从结构(Structure)/制度(Systems)/风格(Style)/员工(Staff)/技能(Skills)/共同价值观(Shared Values)/战略(Strategy)七个维度构建组织保障体系。",
}

# 章节级评分维度定义（诊断/推演阶段共用核心结构，描述文字自适应）
_DIAGNOSIS_DIMENSIONS = """## 评分维度（每项0-20分，允许0.5分精度）— 诊断章节评估标准

### 维度一：模型运用与框架完整性 (0-20)
评估本章是否正确、完整地运用了指定分析模型（{analysis_model}）。
- 18-20: 模型框架运用精准、维度覆盖完整、各要素间逻辑衔接紧密
- 14-17: 模型框架基本正确，但维度覆盖不完整或分析深度不够
- 10-13: 模型框架使用有偏差，维度缺失或混淆，缺乏实质分析
- 0-9: 未使用指定模型或使用完全错误，仅堆砌信息

### 维度二：数据支撑与证据质量 (0-20)
评估本章的分析是否有充分的数据、引用和事实支撑，论据是否可信。
- 18-20: 论据充分、数据来源明确、引用精准，分析有扎实的证据基础
- 14-17: 有一定的数据支撑，但部分论据缺乏来源或证据力度不够
- 10-13: 论据单薄，多为概括性描述，缺乏具体数据和事实
- 0-9: 无数据支撑，纯主观臆断或简单复述

### 维度三：内部逻辑与结构清晰度 (0-20)
评估本章内部的论证结构是否清晰，段落间是否有逻辑递进，小节安排是否合理。
- 18-20: 论证结构严谨、段落间逻辑递进自然、层次分明、结构清晰
- 14-17: 基本结构合理，但部分段落间逻辑跳跃或层次不够清晰
- 10-13: 结构松散，段落间缺乏逻辑联系，组织混乱
- 0-9: 无明显结构，各段落独立无关联

### 维度四：内容深度与专业洞察 (0-20)
评估本章是否从数据中提炼出有深度的分析洞察，是否超越了信息复述。
- 18-20: 从数据中提炼出深刻、独到的分析洞察，展现了高度专业水准
- 14-17: 有一定的分析深度和非表面化洞察，但深度或新颖度不足
- 10-13: 以信息复述为主，缺乏独立的深入分析
- 0-9: 完全依赖资料搬运，无任何原创性分析或洞察

### 维度五：写作质量与规范表达 (0-20)
评估本章的语言是否专业通顺、格式是否规范、是否符合国企公文风格。
- 18-20: 语言精炼专业、术语使用准确、格式规范、完全符合公文风格
- 14-17: 语言基本通顺，但部分表述不够精炼或格式有不一致
- 10-13: 语言偏口语化或表述模糊，格式不规范
- 0-9: 语言混乱、错别字多、格式严重不规范"""

_INITIATIVES_DIMENSIONS = """## 评分维度（每项0-20分，允许0.5分精度）— 推演章节评估标准

### 维度一：模型运用与战略设计严谨度 (0-20)
评估本章是否运用指定模型（{analysis_model}）指导战略目标设定和举措设计。
- 18-20: 模型引导的战略设计精准、目标量化可行、举措与模型逻辑高度一致
- 14-17: 模型应用基本正确，但目标设定偏宏观或举措与模型衔接不够紧密
- 10-13: 模型仅停留在表面引用，未深入指导战略设计
- 0-9: 未使用指定模型或战略设计与模型脱节

### 维度二：数据支撑与证据质量 (0-20)
评估本章的战略举措是否有充分的数据、案例和事实支撑。
- 18-20: 战略举措有充分的数据和案例支撑，量化目标有据可依，论据可信
- 14-17: 有一定的数据支撑，但部分举措缺乏依据或量化不够
- 10-13: 论据单薄，多为概括性描述，缺乏具体数据和案例
- 0-9: 无数据支撑，纯主观臆断或简单复述

### 维度三：内部逻辑与结构清晰度 (0-20)
评估本章内部的论证结构是否清晰，战略目标→举措→路径是否层次分明。
- 18-20: 战略目标→核心举措→实施路径层次分明、逻辑递进自然
- 14-17: 基本结构合理，但部分举措与目标逻辑跳跃或层次不够清晰
- 10-13: 结构松散，战略目标与举措缺乏对应关系
- 0-9: 无明显结构，各段落独立无关联

### 维度四：内容深度与战略创新性 (0-20)
评估本章是否提出了有深度的战略洞察和创新性举措。
- 18-20: 提出深刻的战略洞察和创新性举措，分析角度独特且可行
- 14-17: 有一定的战略深度和创新思路，但创新度或可行性不足
- 10-13: 以常规做法为主，缺乏独立的战略思考
- 0-9: 完全复述已有方案，无任何创新或深度分析

### 维度五：写作质量与规范表达 (0-20)
评估本章的语言是否专业通顺、格式是否规范、是否符合国企公文风格。
- 18-20: 语言精炼专业、术语使用准确、格式规范、完全符合公文风格
- 14-17: 语言基本通顺，但部分表述不够精炼或格式有不一致
- 10-13: 语言偏口语化或表述模糊，格式不规范
- 0-9: 语言混乱、错别字多、格式严重不规范"""


def _build_system_prompt(phase: str, analysis_model: str) -> str:
    """根据章节阶段和分析模型动态生成评审系统提示词。"""
    base_prompt = (
        "你是一位拥有20年经验的国际顶尖战略咨询合伙人，专门评估战略规划报告。\n\n"
        "你必须严格按JSON格式输出评分，不要输出任何其他文字、分析或解释。"
        "不要使用markdown代码块。直接输出纯JSON。\n\n"
    )

    if phase == "diagnosis":
        dim_text = _DIAGNOSIS_DIMENSIONS.format(analysis_model=analysis_model or "战略分析模型")
    else:
        dim_text = _INITIATIVES_DIMENSIONS.format(analysis_model=analysis_model or "战略分析模型")

    return base_prompt + dim_text


def _init_reviewer():
    """Initialize the independent reviewer client."""
    global _reviewer_client, _reviewer_model, _reviewer_max_retries, _fallback_llm

    if _reviewer_client is not None:
        return

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
            _reviewer_model = reviewer_config.get("model", "Pro/moonshotai/Kimi-K2.5")
            logger.info(f"Initialized independent reviewer: {_reviewer_model}")
        else:
            logger.warning("SILICONFLOW_API_KEY not set, using fallback coordinator LLM")
            _fallback_llm = LLMManager("coordinator")
    except Exception as e:
        logger.warning(f"Failed to init independent reviewer: {e}. Using fallback.")
        _fallback_llm = LLMManager("coordinator")


def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM章节评审节点，使用章节级5维度评分标准评估单章草稿质量。

    5个维度（每项0-20分，满分100）：
    - 维度一：模型运用与框架完整性
    - 维度二：数据支撑与证据质量
    - 维度三：内部逻辑与结构清晰度
    - 维度四：内容深度与专业水准
    - 维度五：写作质量与规范表达
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

    # Get context for coherence check
    context_summary = state.get("context_summary", "")
    strategic_blueprint = state.get("strategic_blueprint", {})

    # 构建阶段自适应的系统提示词
    system_prompt = _build_system_prompt(phase, analysis_model)

    user_prompt = _build_review_prompt(
        draft=draft,
        chapter_title=chapter_title,
        phase=phase,
        analysis_model=analysis_model,
        context_summary=context_summary,
        strategic_blueprint=strategic_blueprint,
        chapter_index=current_index
    )

    try:
        response_text = _call_reviewer_llm(system_prompt, user_prompt)
        result = _parse_review_response(response_text, phase)
        logger.info(f"LLM Review [{phase}]: score={result['score']}, issues={len(result['issues'])}")
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


def _call_reviewer_llm(system_prompt: str, user_prompt: str) -> str:
    """调用评审LLM，支持重试逻辑和动态系统提示词。"""
    global _reviewer_client, _reviewer_model, _reviewer_max_retries, _fallback_llm

    if _reviewer_client is not None:
        last_error = None
        for attempt in range(_reviewer_max_retries + 1):
            try:
                response = _reviewer_client.chat.completions.create(
                    model=_reviewer_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=8192,
                    extra_body={"enable_thinking": False}
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

    logger.info("Using fallback coordinator LLM for review")
    return _fallback_llm.invoke(user_prompt, temperature=0.3, max_tokens=8192)


def _build_review_prompt(
    draft: str,
    chapter_title: str,
    phase: str,
    analysis_model: str,
    context_summary: str,
    strategic_blueprint: Dict,
    chapter_index: int
) -> str:
    """构建评审提示词，包含章节专属期望块。"""

    # 截断草稿以适应上下文窗口
    if len(draft) > 2000:
        eval_text = draft[:1500] + "\n\n...[内容省略]...\n" + draft[-500:]
    else:
        eval_text = draft

    # 章节专属期望块
    chapter_expectation = _CHAPTER_EXPECTATIONS.get(chapter_index, "")

    # 构建上下文信息
    context_info = ""
    if context_summary:
        context_info = f"\n前序章节摘要：\n{context_summary[:500]}"

    blueprint_info = ""
    if strategic_blueprint and phase == "initiatives":
        mission = strategic_blueprint.get("mission", "")
        pillars = strategic_blueprint.get("strategic_pillars", [])
        if mission or pillars:
            blueprint_info = f"""
战略蓝图（本章战略举措需与之契合）：
- 核心使命: {mission}
- 战略支柱: {'; '.join(pillars[:3]) if pillars else '未设定'}
"""

    phase_label = "诊断阶段" if phase == "diagnosis" else "推演阶段"

    return f"""请评估以下战略规划报告章节，直接输出JSON，不要输出其他任何文字。

## 章节信息
标题: {chapter_title}
阶段: {phase_label}
分析模型: {analysis_model}
章节专属期望: {chapter_expectation}

{context_info}
{blueprint_info}

## 章节内容
{eval_text}

---

输出格式（严格遵守）:
{{"d1_score":0,"d1_analysis":"详细评语","d2_score":0,"d2_analysis":"详细评语","d3_score":0,"d3_analysis":"详细评语","d4_score":0,"d4_analysis":"详细评语","d5_score":0,"d5_analysis":"详细评语","total_score":0,"issues":["具体问题描述1","具体问题描述2"],"suggestions":"详细改进建议"}}

说明:
- d1=模型运用, d2=数据支撑, d3=内部逻辑, d4=内容深度, d5=写作质量
- 每项0-20分，total=五项之和
- 每个维度的analysis必须写100-200字的详细评语，包含：(1)该维度的具体表现 (2)做得好的地方 (3)存在的不足 (4)具体的改进方向
- issues必须列出3-5个具体问题，每个问题描述要包含问题所在位置和具体内容（如"第三节中关于XX的分析缺乏数据支撑"）
- suggestions给出200字左右的总体改进建议，包含优先级排序"""


def _parse_review_response(response_text: str, phase: str = "diagnosis") -> Dict:
    """解析评审响应为标准格式，使用阶段感知阈值。"""

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

    # Parse JSON
    raw = _try_parse_json(text)
    if not raw:
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            raw = _try_parse_json(brace_match.group(0))

    if not raw:
        # Fix common JSON errors and retry
        fixed = text
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        fixed = re.sub(r'//.*?\n', '\n', fixed)
        raw = _try_parse_json(fixed)

    if not raw:
        logger.warning("All JSON parsing strategies failed, using regex extraction")
        return _regex_extract_review(text, phase)

    return _normalize_review(raw, phase)


def _try_parse_json(text: str) -> Dict:
    """Try to parse JSON, return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_review(raw: Dict, phase: str = "diagnosis") -> Dict:
    """将解析后的JSON标准化为评审输出格式，使用阶段感知阈值。"""
    dim_mapping = {
        "d1": ("model_application", "模型运用与框架完整性"),
        "d2": ("data_support", "数据支撑与证据质量"),
        "d3": ("internal_logic", "内部逻辑与结构清晰度"),
        "d4": ("content_depth", "内容深度与专业水准"),
        "d5": ("writing_quality", "写作质量与规范表达"),
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

    issues = raw.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    suggestions = raw.get("suggestions", "")
    if not isinstance(suggestions, str):
        suggestions = str(suggestions) if suggestions else ""

    # 阶段感知阈值：诊断阶段70分通过，推演阶段72分通过
    approve_threshold = 72 if phase == "initiatives" else 70
    suggestion = "approve" if total_score >= approve_threshold else "revise:writing"
    if 50 <= total_score < approve_threshold:
        weakest = min(dimension_scores.items(), key=lambda x: x[1]["score"])
        if weakest[0] in ("model_application", "internal_logic"):
            suggestion = "revise:logic"
        elif weakest[0] in ("data_support", "content_depth"):
            suggestion = "revise:data"
        else:
            suggestion = "revise:writing"

    improvement_hints = {}
    if suggestions:
        improvement_hints["writer"] = suggestions
    for eng_name, data in dimension_scores.items():
        if data["score"] < 14 and data["analysis"]:
            if eng_name in ("model_application", "internal_logic"):
                improvement_hints.setdefault("analyst", data["analysis"])
            elif eng_name in ("data_support", "content_depth"):
                improvement_hints.setdefault("researcher", data["analysis"])

    return {
        "score": int(total_score),
        "dimension_scores": dimension_scores,
        "issues": issues,
        "suggestion": suggestion,
        "improvement_hints": improvement_hints,
    }


def _regex_extract_review(text: str, phase: str = "diagnosis") -> Dict:
    """正则提取兜底方案，使用阶段感知阈值。"""

    dim_keys = ["d1", "d2", "d3", "d4", "d5"]
    dim_mapping = {
        "d1": ("model_application", "模型运用与框架完整性"),
        "d2": ("data_support", "数据支撑与证据质量"),
        "d3": ("internal_logic", "内部逻辑与结构清晰度"),
        "d4": ("content_depth", "内容深度与专业水准"),
        "d5": ("writing_quality", "写作质量与规范表达"),
    }

    dimension_scores = {}
    for key in dim_keys:
        eng_name, cn_name = dim_mapping[key]
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

    issues = []
    issues_match = re.search(r'"issues"\s*:\s*\[([^\]]*)\]', text)
    if issues_match:
        issues = [s.strip().strip('"').strip("'") for s in issues_match.group(1).split(',') if s.strip().strip('"')]

    suggestions_match = re.search(r'"suggestions"\s*:\s*"([^"]*)"', text)
    suggestions = suggestions_match.group(1) if suggestions_match else ""

    # 阶段感知阈值
    approve_threshold = 72 if phase == "initiatives" else 70
    suggestion = "approve" if total_score >= approve_threshold else "revise:writing"

    improvement_hints = {}
    if suggestions:
        improvement_hints["writer"] = suggestions

    logger.info(f"Regex extraction: score={total_score}, dims={len(dimension_scores)}")

    return {
        "score": int(max(0, min(100, total_score))),
        "dimension_scores": dimension_scores,
        "issues": issues,
        "suggestion": suggestion,
        "improvement_hints": improvement_hints,
    }
