import pytest
from rag_project.agent.state import GraphState

def test_graph_state_initialization():
    """测试GraphState初始化"""
    state = GraphState(
        user_input="生成2024年交通投资战略报告",
        global_plan=[],
        current_chapter_index=0,
        context_pool=[],
        context_summary="",
        chapter_title="",
        chapter_scratchpad={},
        current_draft="",
        human_feedback={}
    )
    assert state["user_input"] == "生成2024年交通投资战略报告"
    assert state["current_chapter_index"] == 0
    assert isinstance(state["chapter_scratchpad"], dict)

def test_context_pool_accumulation():
    """测试context_pool累加行为"""
    state1 = GraphState(
        user_input="test",
        global_plan=[],
        current_chapter_index=0,
        context_pool=["第一章内容"],
        context_summary="",
        chapter_title="",
        chapter_scratchpad={},
        current_draft="",
        human_feedback={}
    )

    # 模拟累加
    state2 = GraphState(**{**state1, "context_pool": state1["context_pool"] + ["第二章内容"]})
    assert len(state2["context_pool"]) == 2
    assert "第一章内容" in state2["context_pool"]
    assert "第二章内容" in state2["context_pool"]
