"""
LangGraph workflow builder for report generation with two-stage strategic deduction architecture.

Creates a StateGraph that orchestrates the multi-agent report generation process:
- Diagnosis Phase (Chapters 1-3): Environment analysis, regional strategy, internal diagnosis
- Strategic Blueprint Generation: TOWS matrix analysis, mission formulation, KPI setting
- Initiatives Phase (Chapters 4-8): Strategic initiatives with blueprint constraints
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rag_project.agent.state import GraphState
from rag_project.agent.llm_manager import LLMManager
from rag_project.agent.retriever import RAGRetriever
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
# NEW: Import strategist node for two-stage architecture
from rag_project.agent.nodes.strategist import strategist_node


def create_report_graph():
    """
    Create and compile the LangGraph workflow for report generation with two-stage strategic architecture.

    The graph follows this enhanced flow:
    1. coordinator - Entry point, generates fixed 8-chapter strategic plan
    2. prepare_chapter - Prepares chapter state (injects blueprint for initiatives phase)
    3. researcher - Retrieves relevant context from knowledge base
    4. analyst - Analyzes content using specified strategic models (PEST, SWOT, BCG, etc.)
    5. writer - Drafts chapter with strategic model constraints
    6. human_review - Awaits human feedback (interrupts execution)

    Branching after Chapter 3 (index=2):
    7a. strategist - Generates strategic blueprint from SWOT (TOWS matrix, mission, KPIs)
    8a. human_review - Strategic blueprint approval checkpoint
    7b-8b. [If approved] Continue to Chapter 4 (initiatives phase)

    Final stage:
    9. archiver - Generates executive summary and assembles final report with blueprint appendix

    Returns:
        Compiled StateGraph with MemorySaver checkpointer
    """
    # Create the StateGraph with our state schema
    workflow = StateGraph(GraphState)

    # Initialize LLM managers for each agent type
    coordinator_llm = LLMManager("coordinator")
    researcher_llm = LLMManager("researcher")
    analyst_llm = LLMManager("analyst")
    writer_llm = LLMManager("writer")
    strategist_llm = LLMManager("strategist")  # NEW: For strategic blueprint generation
    archiver_llm = LLMManager("archiver")  # NEW: For executive summary generation
    prep_llm = LLMManager("coordinator")  # For rolling context_summary compression

    # Initialize RAG retriever for researcher (reads retrieval strategy from agent config)
    retriever = RAGRetriever(agent_config_path="config/agent_config.yaml")

    # Add all nodes
    # Each node is wrapped in lambda to inject dependencies
    workflow.add_node("coordinator", lambda state: coordinator_node(state, coordinator_llm))
    workflow.add_node("prepare_chapter", lambda state: prepare_chapter_node(state, prep_llm))
    workflow.add_node("researcher", lambda state: researcher_node(state, retriever, researcher_llm))
    workflow.add_node("analyst", lambda state: analyst_node(state, analyst_llm))
    workflow.add_node("writer", lambda state: writer_node(state, writer_llm))
    workflow.add_node("strategist", lambda state: strategist_node(state, strategist_llm))  # NEW
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("archiver", lambda state: archiver_node(state, archiver_llm))  # UPDATED

    # Set the entry point
    workflow.set_entry_point("coordinator")

    # Add linear edges for the main flow
    workflow.add_edge("coordinator", "prepare_chapter")
    workflow.add_edge("prepare_chapter", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "human_review")

    # NEW: Add edge from strategist back to human_review for blueprint approval
    workflow.add_edge("strategist", "human_review")

    # Add conditional edges from human_review
    # Enhanced routing to support two-stage architecture:
    # - "prepare_chapter": Next chapter (including entering initiatives phase after blueprint approval)
    # - "strategist": Generate strategic blueprint (after Chapter 3)
    # - "researcher"/"analyst"/"writer": Revision routes
    # - "end": Finalize report
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "prepare_chapter": "prepare_chapter",  # Next chapter (includes initiatives phase)
            "strategist": "strategist",  # Generate/regenerate blueprint
            "researcher": "researcher",  # Data revision
            "analyst": "analyst",  # Logic revision
            "writer": "writer",  # Writing revision
            "end": "archiver"  # Finalize report
        }
    )

    # Add edge from archiver to END
    workflow.add_edge("archiver", END)

    # Compile the graph with MemorySaver checkpointer
    # This enables persistence and resumption of workflow execution
    checkpointer = MemorySaver()

    # Compile with interrupt before human_review
    # This pauses execution to allow human input (both for chapters and blueprint)
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )

    return app
