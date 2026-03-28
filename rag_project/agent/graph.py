"""
LangGraph workflow builder for report generation.

Creates a StateGraph that orchestrates the multi-agent report generation process.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rag_project.agent.state import GraphState
from rag_project.agent.nodes import (
    coordinator_node,
    prepare_chapter_node,
    researcher_node,
    analyst_node,
    writer_node,
    human_review_node,
    archiver_node,
    should_continue
)


def create_report_graph():
    """
    Create and compile the LangGraph workflow for report generation.

    The graph follows this flow:
    1. coordinator - Entry point, initializes the report
    2. prepare_chapter - Prepares chapter structure
    3. researcher - Retrieves relevant context
    4. analyst - Analyzes content and generates insights
    5. writer - Drafts the chapter content
    6. human_review - Awaits human feedback (interrupts execution)
    7. Either loops back to prepare_chapter or proceeds to archiver
    8. archiver - Finalizes and saves the report

    Returns:
        Compiled StateGraph with MemorySaver checkpointer
    """
    # Create the StateGraph with our state schema
    workflow = StateGraph(GraphState)

    # Add all nodes
    # Each node is wrapped in lambda to inject dependencies
    workflow.add_node("coordinator", lambda state: coordinator_node(state))
    workflow.add_node("prepare_chapter", prepare_chapter_node)
    workflow.add_node("researcher", lambda state: researcher_node(state))
    workflow.add_node("analyst", lambda state: analyst_node(state))
    workflow.add_node("writer", lambda state: writer_node(state))
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("archiver", archiver_node)

    # Set the entry point
    workflow.set_entry_point("coordinator")

    # Add linear edges for the main flow
    workflow.add_edge("coordinator", "prepare_chapter")
    workflow.add_edge("prepare_chapter", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "human_review")

    # Add conditional edges from human_review
    # Based on human feedback, either continue to next chapter or finish
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "continue": "prepare_chapter",  # Next chapter
            "end": "archiver"  # Finalize report
        }
    )

    # Add edge from archiver to END
    workflow.add_edge("archiver", END)

    # Compile the graph with MemorySaver checkpointer
    # This enables persistence and resumption of workflow execution
    checkpointer = MemorySaver()

    # Compile with interrupt before human_review
    # This pauses execution to allow human input
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )

    return app
