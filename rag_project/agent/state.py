from typing import TypedDict, List, Dict, Annotated, Optional
import operator

class GraphState(TypedDict):
    """LangGraph工作流的全局状态定义

    三层记忆架构:
    1. 长期记忆: Milvus RAG系统 (仅Researcher可访问)
    2. 短期工作区: chapter_scratchpad (章节专属沙盒，阅后即焚)
    3. 神圣上下文池: context_pool (仅存入审核通过的定稿)
    """

    # --- 输入层 ---
    user_input: str  # 用户的原始请求

    # --- 全局规划层 ---
    global_plan: List[Dict]  # Coordinator生成的完整章节大纲（包含title, phase, analysis_model元数据）
    current_chapter_index: int  # 当前执行的章节索引

    # --- 战略蓝图层（两阶段架构） ---
    strategic_blueprint: Optional[Dict]  # 战略蓝图（包含mission, swot_analysis, tows_strategies, pillars, kpis, approved状态）
    current_phase: str  # 当前所处战略阶段: "diagnosis" (诊断阶段) 或 "initiatives" (推演阶段)

    # --- 上下文层 (长期/跨章节记忆) ---
    # 使用operator.add确保纯累加，不覆盖
    context_pool: Annotated[List[str], operator.add]  # 已审核通过的章节原文
    context_summary: str  # 压缩后的全局上下文摘要

    # --- 当前章节层 (短期/工作区记忆) ---
    chapter_title: str  # 当前正在撰写的章节名
    chapter_question: str  # 当前章节的研究问题（从chapter_title转换而来）
    chapter_context: str  # 当前章节的上下文信息
    chapter_scratchpad: Dict  # 本章的结构化草稿本
    current_draft: str  # Writer生成的当前草稿文本
    _pending_chapter_knowledge: Dict  # human_review保存的上一章知识（供prep_chapter压缩用）

    # --- 控制层 ---
    human_feedback: Dict  # 人类结构化反馈指令
    review_decision: str  # 审核决策: "approve", "revise:data", "revise:logic", "revise:writing", "finished"
    auto_revision_count: int  # 当前章节的自动修订次数（防止死循环，max=1）
    llm_review_result: Optional[Dict]  # LLM 评审结果（score, issues, suggestion）

    # --- 输出层 ---
    final_report: str  # Archiver生成的最终完整报告
