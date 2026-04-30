"""
Strategist Node - Generates strategic blueprint from diagnosis phase

This node is triggered after the diagnosis phase (Chapters 1-3) is complete.
It extracts SWOT analysis from Chapter 3, applies TOWS matrix analysis,
and generates a comprehensive strategic blueprint including:
- Mission statement
- SWOT analysis
- TOWS strategies (SO, WO, ST, WT)
- Strategic pillars (3-5 pillars)
- KPIs across BSC dimensions
"""

import json
import logging
from typing import Dict, Any, List

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def _compress_diagnosis(context_pool: List[str], user_input: str, llm_manager) -> str:
    """
    将诊断阶段（前3章）压缩为结构化综合摘要。

    相比原方案(chapter_3[:3000])的改进:
    - 涵盖全部3章内容，而非只看第3章
    - 使用 LLM 智能压缩，保留关键数据和分析结论
    - 输出约2000字的结构化摘要

    Args:
        context_pool: 已完成的章节列表
        user_input: 用户原始请求
        llm_manager: LLM 实例

    Returns:
        诊断阶段综合摘要文本
    """
    # 拼合前3章内容（每章限8000字避免过长）
    chapters_text = ""
    for i, chapter in enumerate(context_pool[:3]):
        # 提取章节标题
        title = f"第{i+1}章"
        for line in chapter.split('\n'):
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        content = chapter[:8000] if len(chapter) > 8000 else chapter
        chapters_text += f"\n\n=== {title} ===\n{content}"

    prompt = f"""请将以下诊断阶段的三章内容压缩为一份结构化综合摘要。

主题: {user_input}

要求:
1. 保留所有关键数据（投资额、增长率、政策指标等具体数字）
2. 保留所有政策名称、战略定位、核心结论
3. 重点保留SWOT分析相关内容（优势、劣势、机会、威胁）
4. 按以下结构输出:
   - 宏观环境要点（政策、经济）
   - 区域战略定位
   - 行业竞争态势
   - 内部优势与劣势
   - 外部机遇与威胁
   - 关键数据汇总（列出所有具体数字）
5. 3000字以内
6. 必须使用中文

诊断阶段内容:
{chapters_text}

请生成综合摘要:"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.3, max_tokens=4096)
        logger.info(f"Compressed diagnosis chapters into {len(response)} chars")
        return response
    except Exception as e:
        logger.error(f"Failed to compress diagnosis: {e}. Using raw Chapter 3.")
        # 回退: 使用第3章的前5000字（比原来的3000字多）
        chapter_3 = context_pool[2] if len(context_pool) >= 3 else ""
        return chapter_3[:5000]


def strategist_node(state: Dict[str, Any], llm_manager) -> Dict[str, Any]:
    """
    基于诊断阶段（第1-3章）生成战略蓝图。

    触发时机: 当diagnosis阶段的章节（前三章）完成并存入context_pool后触发。

    核心逻辑:
    1. 从context_pool中提取第三章（包含SWOT分析）
    2. 使用LLM提取结构化的SWOT矩阵
    3. 应用TOWS矩阵分析生成战略选项
    4. 推导生成核心使命、战略支柱和量化KPI
    5. 返回结构化的strategic_blueprint草案（未批准状态）

    Args:
        state: Current workflow state containing:
            - context_pool: List of completed chapters (Chapters 1-3)
            - user_input: Original user request for context
        llm_manager: LLMManager instance for blueprint generation

    Returns:
        Dict with updates:
            - strategic_blueprint: Dict containing:
                - mission: Core mission statement (20-30 chars)
                - swot_analysis: Structured SWOT matrix
                - tows_strategies: TOWS strategy combinations (SO, WO, ST, WT)
                - strategic_pillars: List of 3-5 strategic pillars
                - kpis: KPIs organized by BSC dimensions
                - approved: False (pending human review)

    Example:
        >>> state = {
        ...     "context_pool": ["第一章内容...", "第二章内容...", "第三章包含SWOT..."],
        ...     "user_input": "生成江西交投集团战略规划"
        ... }
        >>> result = strategist_node(state, llm_manager)
        >>> blueprint = result["strategic_blueprint"]
        >>> "mission" in blueprint
        True
        >>> blueprint["approved"]
        False
    """
    context_pool = state.get("context_pool", [])
    user_input = state.get("user_input", "")

    logger.info(f"Strategist node generating strategic blueprint from {len(context_pool)} completed chapters")

    # Validate that we have at least 3 chapters (diagnosis phase)
    if len(context_pool) < 3:
        logger.warning(f"Expected at least 3 chapters for blueprint generation, got {len(context_pool)}")
        # Continue anyway with available chapters

    # 新代码: 压缩全部3章后再提取 SWOT
    diagnosis_summary = _compress_diagnosis(context_pool, user_input, llm_manager)

    if not diagnosis_summary:
        logger.error("Failed to generate diagnosis summary")
        return _get_fallback_blueprint(user_input)

    # Extract SWOT from compressed diagnosis summary
    logger.info("Extracting SWOT analysis from compressed diagnosis summary...")

    try:
        swot_data = _extract_swot_from_chapter(diagnosis_summary, llm_manager)
        logger.info(f"Extracted SWOT: S={len(swot_data.get('Strengths', []))}, "
                    f"W={len(swot_data.get('Weaknesses', []))}, "
                    f"O={len(swot_data.get('Opportunities', []))}, "
                    f"T={len(swot_data.get('Threats', []))}")

        # Step 2: Generate TOWS strategies and complete blueprint
        logger.info("Generating TOWS strategies and strategic blueprint...")
        strategic_blueprint = _generate_strategic_blueprint(
            swot_data=swot_data,
            user_input=user_input,
            llm_manager=llm_manager
        )

        # Set approval status to False (pending human review)
        strategic_blueprint["approved"] = False

        logger.info(f"Generated strategic blueprint with {len(strategic_blueprint.get('strategic_pillars', []))} pillars "
                    f"and {len(strategic_blueprint.get('kpis', {}))} KPI dimensions")

        return {
            "strategic_blueprint": strategic_blueprint,
            "current_draft": ""  # Clear draft to signal blueprint review
        }

    except Exception as e:
        logger.error(f"Error generating strategic blueprint: {e}. Using fallback.")
        return _get_fallback_blueprint(user_input)


def _extract_swot_from_chapter(chapter_text: str, llm_manager) -> Dict[str, List[str]]:
    """
    从第三章文本中提取结构化的SWOT分析。

    Args:
        chapter_text: Chapter 3 text content (should contain SWOT analysis)
        llm_manager: LLMManager instance

    Returns:
        Dict with keys: Strengths, Weaknesses, Opportunities, Threats
    """
    prompt = f"""你是一位战略分析专家。请从以下章节文本中提取结构化的SWOT分析。

章节文本:
{chapter_text}

任务要求:
1. 仔细阅读文本，识别并提取SWOT四个维度的内容
2. 每个维度至少提取3-5项
3. 确保提取的内容是原文中明确提到的优势、劣势、机会、威胁
4. 返回JSON格式

返回JSON结构:
{{
    "Strengths": ["优势1", "优势2", "优势3", ...],
    "Weaknesses": ["劣势1", "劣势2", "劣势3", ...],
    "Opportunities": ["机会1", "机会2", "机会3", ...],
    "Threats": ["威胁1", "威胁2", "威胁3", ...]
}}

输出要求:
- 必须使用中文
- 每项应该简洁明确（1-2句话）
- 只返回JSON对象，不要包含其他文本

现在请提取SWOT分析:"""

    response = llm_manager.invoke(prompt, temperature=0.3)

    # Parse JSON response
    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        swot_data = json.loads(response)

        # Validate structure
        required_keys = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
        for key in required_keys:
            if key not in swot_data:
                logger.warning(f"Missing {key} in SWOT data, adding empty list")
                swot_data[key] = []
            if not isinstance(swot_data[key], list):
                swot_data[key] = [str(swot_data[key])]

        return swot_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SWOT JSON: {e}")
        # Return empty SWOT structure
        return {
            "Strengths": [],
            "Weaknesses": [],
            "Opportunities": [],
            "Threats": []
        }


def _generate_strategic_blueprint(swot_data: Dict, user_input: str, llm_manager) -> Dict[str, Any]:
    """
    基于SWOT分析生成完整的战略蓝图。

    使用TOWS矩阵分析方法，将SWOT转化为战略选项，并生成:
    - 核心使命
    - TOWS策略 (SO, WO, ST, WT)
    - 战略支柱 (3-5个)
    - KPIs (按平衡计分卡四个维度)

    Args:
        swot_data: Structured SWOT analysis
        user_input: Original user request
        llm_manager: LLMManager instance

    Returns:
        Complete strategic blueprint dict
    """
    prompt = _generate_blueprint_prompt(swot_data, user_input)

    response = llm_manager.invoke(prompt, temperature=0.5)

    # Parse JSON response
    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        blueprint = json.loads(response)

        # Validate required fields
        required_fields = ["mission", "swot_analysis", "tows_strategies", "strategic_pillars", "kpis"]
        for field in required_fields:
            if field not in blueprint:
                logger.warning(f"Missing {field} in blueprint, adding default")
                if field == "mission":
                    blueprint[field] = "服务国家战略，推动高质量发展"
                elif field == "swot_analysis":
                    blueprint[field] = swot_data
                elif field == "tows_strategies":
                    blueprint[field] = {"SO": [], "WO": [], "ST": [], "WT": []}
                elif field == "strategic_pillars":
                    blueprint[field] = ["战略支柱1：业务发展", "战略支柱2：创新驱动", "战略支柱3：管理提升"]
                elif field == "kpis":
                    blueprint[field] = {
                        "财务维度": {},
                        "客户/民生维度": {},
                        "运营维度": {},
                        "学习成长维度": {}
                    }

        return blueprint

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse blueprint JSON: {e}")
        return _get_fallback_blueprint_structure(swot_data)


def _generate_blueprint_prompt(swot_data: Dict, user_input: str) -> str:
    """
    生成战略蓝图生成的提示词。

    Args:
        swot_data: Structured SWOT analysis
        user_input: Original user request

    Returns:
        Formatted prompt string
    """
    return f"""你是一位资深的国企战略规划专家。基于前期的诊断分析，现在需要制定省属国企的战略蓝图。

## 已识别的SWOT分析

**优势 (Strengths)**:
{chr(10).join(f"- {s}" for s in swot_data.get('Strengths', []) if s)}

**劣势 (Weaknesses)**:
{chr(10).join(f"- {w}" for w in swot_data.get('Weaknesses', []) if w)}

**机会 (Opportunities)**:
{chr(10).join(f"- {o}" for o in swot_data.get('Opportunities', []) if o)}

**威胁 (Threats)**:
{chr(10).join(f"- {t}" for t in swot_data.get('Threats', []) if t)}

## 任务要求

请使用TOWS矩阵分析法，生成完整的战略蓝图。返回JSON格式：

{{
    "mission": "一句话凝练核心使命（20-30字）",
    "swot_analysis": {{
        "strengths": ["优势1", "优势2", ...],
        "weaknesses": ["劣势1", "劣势2", ...],
        "opportunities": ["机会1", "机会2", ...],
        "threats": ["威胁1", "威胁2", ...]
    }},
    "tows_strategies": {{
        "SO": ["利用优势抓住机会的策略1（具体可执行）", "利用优势抓住机会的策略2"],
        "WO": ["弥补劣势抓住机会的策略1（具体可执行）"],
        "ST": ["利用优势应对威胁的策略1（具体可执行）"],
        "WT": ["减少劣势规避威胁的策略1（具体可执行）"]
    }},
    "strategic_pillars": [
        "战略支柱1：名称与描述（如：主业升级与资本优化：夯实交通投资建设主阵地，通过REITs等工具创新资本运作）",
        "战略支柱2：名称与描述（如：创新驱动与智慧赋能：培育智慧绿色交通新动能）",
        "战略支柱3：名称与描述（如：产业协同与价值延伸：构建交旅融合新生态）",
        "战略支柱4：名称与描述（如：治理提升与风险防控：完善现代企业制度）"
    ],
    "kpis": {{
        "财务维度": {{"净资产收益率(ROE)": "年均不低于5%", "基础设施REITs发行规模": "累计不低于100亿元", "资产负债率": "控制在65%以内"}},
        "客户/民生维度": {{"公众出行服务满意度": "达到92分以上", "路网通行效率指数": "年均提升3%", "重大交通项目带动效应评估": "达到优良等级"}},
        "运营维度": {{"资产数字化管理覆盖率": "达到100%", "新建项目绿色建造技术应用率": "达到100%", "项目全生命周期成本管控达标率": "达到95%"}},
        "学习成长维度": {{"数字化专业人才占比": "提升至15%", "年度新增授权专利及工法数量": "不少于20项", "核心骨干员工年均培训时长": "不低于80小时"}}
    }}
}}

## 输出要求

1. **所有内容必须使用中文**
2. **mission（核心使命）**:
   - 必须体现国企"服务国家战略、承担社会责任、推动高质量发展"的定位
   - 20-30字，高度凝练
   - 示例："服务交通强省战略，打造一流国有资本投资运营平台，推动现代化产业体系建设"
3. **TOWS策略**:
   - 必须具体可执行（避免空洞表述）
   - 每类策略至少2-3项
   - 体现内外部匹配的智慧
4. **战略支柱**:
   - 建议4个
   - 覆盖业务升级、创新驱动、产业协同、治理提升等维度
   - 每个支柱应有明确的名称和具体描述
5. **KPIs（关键要求）**:
   - 必须符合SMART原则（具体、可衡量、可达成、相关、有时限）
   - 按平衡计分卡四个维度组织
   - 每个维度至少3个指标
   - 指标必须包含具体的目标值和量化数字
   - 财务维度必须包含ROE、REITs发行规模、资产负债率等核心指标
   - 目标值应具有挑战性但可实现，体现国企高质量发展的要求
6. **只返回JSON对象，不要包含任何其他文本**

## 参考背景

用户请求: {user_input}

现在生成战略蓝图:"""


def _get_fallback_blueprint(user_input: str) -> Dict[str, Any]:
    """
    生成回退战略蓝图（当LLM失败时）。

    Args:
        user_input: Original user request

    Returns:
        Fallback blueprint structure
    """
    logger.warning("Using fallback strategic blueprint")

    return {
        "mission": "服务交通强省战略，打造一流国有资本投资运营平台，推动现代化产业体系建设",
        "swot_analysis": {
            "strengths": [
                "省属国企平台优势",
                "丰富的交通基础设施投资经验",
                "稳定的资金来源和信用评级"
            ],
            "weaknesses": [
                "创新能力有待提升",
                "市场化程度不足",
                "人才结构需优化"
            ],
            "opportunities": [
                "交通强省战略机遇",
                "新型基础设施建设投资",
                "绿色低碳转型需求"
            ],
            "threats": [
                "经济下行压力",
                "行业竞争加剧",
                "政策调整风险"
            ]
        },
        "tows_strategies": {
            "SO": [
                "利用省属平台优势，抢抓交通强省战略机遇",
                "发挥投资经验优势，参与新基建投资"
            ],
            "WO": [
                "通过战略合作弥补创新短板",
                "推进市场化改革���升竞争力"
            ],
            "ST": [
                "强化风险防控应对经济下行",
                "差异化策略应对行业竞争"
            ],
            "WT": [
                "深化改革提升组织韧性",
                "优化结构增强抗风险能力"
            ]
        },
        "strategic_pillars": [
            "战略支柱1：主业提质 - 夯实交通投资建设主阵地",
            "战略支柱2：创新驱动 - 培育智慧绿色交通新动能",
            "战略支柱3：产业协同 - 构建交旅融合新生态",
            "战略支柱4：治理提升 - 完善现代企业制度"
        ],
        "kpis": {
            "财务维度": {
                "净资产收益率(ROE)": "年均不低于5%",
                "基础设施REITs发行规模": "累计不低于100亿元",
                "资产负债率": "控制在65%以内"
            },
            "客户/民生维度": {
                "公众出行服务满意度": "达到92分以上",
                "路网通行效率指数": "年均提升3%",
                "重大交通项目带动效应评估": "达到优良等级"
            },
            "运营维度": {
                "资产数字化管理覆盖率": "达到100%",
                "新建项目绿色建造技术应用率": "达到100%",
                "项目全生命周期成本管控达标率": "达到95%"
            },
            "学习成长维度": {
                "数字化专业人才占比": "提升至15%",
                "年度新增授权专利及工法数量": "不少于20项",
                "核心骨干员工年均培训时长": "不低于80小时"
            }
        },
        "approved": False
    }


def _get_fallback_blueprint_structure(swot_data: Dict = None) -> Dict[str, Any]:
    """
    生成回退蓝图结构（用于JSON解析失败时）。

    Args:
        swot_data: Optional SWOT data to include

    Returns:
        Minimal blueprint structure
    """
    logger.warning("Using fallback blueprint structure")

    return {
        "mission": "服务国家战略，推动高质量发展",
        "swot_analysis": swot_data or {
            "Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []
        },
        "tows_strategies": {
            "SO": [], "WO": [], "ST": [], "WT": []
        },
        "strategic_pillars": [
            "战略支柱1：业务发展",
            "战略支柱2：创新驱动",
            "战略支柱3：管理提升"
        ],
        "kpis": {
            "财务维度": {
                "净资产收益率(ROE)": "年均不低于5%",
                "基础设施REITs发行规模": "累计不低于100亿元",
                "资产负债率": "控制在65%以内"
            },
            "客户/民生维度": {
                "公众出行服务满意度": "达到92分以上",
                "路网通行效率指数": "年均提升3%"
            },
            "运营维度": {
                "资产数字化管理覆盖率": "达到100%",
                "新建项目绿色建造技术应用率": "达到100%"
            },
            "学习成长维度": {
                "数字化专业人才占比": "提升至15%",
                "年度新增授权专利及工法数量": "不少于20项"
            }
        }
    }
