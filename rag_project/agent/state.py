from typing import TypedDict, List, Dict, Annotated
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
    global_plan: List[str]  # Coordinator生成的完整章节大纲
    current_chapter_index: int  # 当前执行的章节索引

    # --- 上下文层 (长期/跨章节记忆) ---
    # 使用operator.add确保纯累加，不覆盖
    context_pool: Annotated[List[str], operator.add]  # 已审核通过的章节原文
    context_summary: str  # 压缩后的全局上下文摘要

    # --- 当前章节层 (短期/工作区记忆) ---
    chapter_title: str  # 当前正在撰写的章节名
    chapter_scratchpad: Dict  # 本章的结构化草稿本
    current_draft: str  # Writer生成的当前草稿文本

    # --- 控制层 ---
    human_feedback: Dict  # 人类结构化反馈指令
