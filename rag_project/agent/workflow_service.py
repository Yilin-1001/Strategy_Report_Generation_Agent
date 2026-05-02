"""Bridge between LangGraph workflow and Gradio frontend."""
import threading
from typing import Dict, Any, Optional, Generator

from rag_project.agent.graph import create_report_graph

# Node display names for progress UI
NODE_LABELS = {
    "coordinator": "Coordinator (总体规划)",
    "prepare_chapter": "准备章节上下文",
    "researcher": "Researcher (数据检索)",
    "analyst": "Analyst (战略分析)",
    "writer": "Writer (内容撰写)",
    "reviewer": "Reviewer (质量评审)",
    "strategist": "Strategist (蓝图生成)",
    "archiver": "Archiver (报告汇总)",
    "human_review": "等待人工审核",
}

# Ordered list of pipeline nodes for progress indicator
PIPELINE_NODES = ["coordinator", "prepare_chapter", "researcher", "analyst", "writer", "reviewer"]


class WorkflowService:
    """Manages LangGraph workflow lifecycle for the Gradio frontend."""

    def __init__(self):
        self.graph = create_report_graph()
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def start_report(self, user_input: str, thread_id: str) -> Generator[Dict[str, Any], None, None]:
        """Start a new report generation, yielding intermediate states for each node.

        Yields dicts with:
            "type": "progress" — intermediate node completion update
            "node": str — the node that just completed
            "chapter_index": int — current chapter index
            "phase": str — current phase
            "node_output": dict — the actual output from this node (for rendering partial results)

        Final yield has "type": "done" plus full UI state.
        """
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"user_input": user_input}

        for event in self.graph.stream(initial_state, config=config):
            # Each event is {node_name: node_output}
            for node_name, node_output in event.items():
                if node_name == "human_review":
                    continue  # Skip, we'll handle after loop
                # Extract useful info from node output
                chapter_idx = node_output.get("current_chapter_index", 0) if isinstance(node_output, dict) else 0
                phase = node_output.get("current_phase", "diagnosis") if isinstance(node_output, dict) else "diagnosis"
                yield {
                    "type": "progress",
                    "node": node_name,
                    "chapter_index": chapter_idx,
                    "phase": phase,
                    "node_output": node_output if isinstance(node_output, dict) else {},
                }

        snapshot = self.graph.get_state(config)
        self._sessions[thread_id] = {
            "config": config,
            "snapshot": snapshot,
            "status": "waiting_review",
        }
        yield {
            "type": "done",
            **self._extract_ui_state(snapshot, thread_id),
        }

    def get_current_state(self, thread_id: str) -> Dict[str, Any]:
        """Get the current workflow state snapshot for UI rendering."""
        if thread_id not in self._sessions:
            return {"status": "not_found"}
        config = self._sessions[thread_id]["config"]
        snapshot = self.graph.get_state(config)
        self._sessions[thread_id]["snapshot"] = snapshot
        return self._extract_ui_state(snapshot, thread_id)

    def submit_review(self, thread_id: str, decision: str, feedback: str = "") -> Generator[Dict[str, Any], None, None]:
        """Submit human review decision and resume workflow, yielding progress."""
        session = self._sessions[thread_id]
        config = session["config"]

        feedback_data = {"decision": decision, "comment": feedback}

        self.graph.update_state(
            config,
            {"review_decision": decision, "human_feedback": feedback_data},
        )

        for event in self.graph.stream(None, config=config):
            for node_name, node_output in event.items():
                if node_name == "human_review":
                    continue
                chapter_idx = node_output.get("current_chapter_index", 0) if isinstance(node_output, dict) else 0
                phase = node_output.get("current_phase", "diagnosis") if isinstance(node_output, dict) else "diagnosis"
                yield {
                    "type": "progress",
                    "node": node_name,
                    "chapter_index": chapter_idx,
                    "phase": phase,
                    "node_output": node_output if isinstance(node_output, dict) else {},
                }

        snapshot = self.graph.get_state(config)
        session["snapshot"] = snapshot

        if snapshot.values.get("final_report"):
            session["status"] = "completed"
        else:
            session["status"] = "waiting_review"

        yield {
            "type": "done",
            **self._extract_ui_state(snapshot, thread_id),
        }

    def get_report(self, thread_id: str) -> Optional[str]:
        """Get the final report if generation is complete."""
        if thread_id not in self._sessions:
            return None
        session = self._sessions[thread_id]
        if session["status"] != "completed":
            return None
        config = session["config"]
        snapshot = self.graph.get_state(config)
        return snapshot.values.get("final_report")

    def list_reports(self) -> list:
        """List all saved reports."""
        reports = []
        for tid, session in self._sessions.items():
            if session["status"] == "completed":
                config = session["config"]
                snapshot = self.graph.get_state(config)
                values = snapshot.values
                reports.append({
                    "thread_id": tid,
                    "title": values.get("user_input", "未命名报告"),
                    "status": "completed",
                    "chapters": len(values.get("context_pool", [])),
                    "word_count": len(values.get("final_report", "")),
                })
        return reports

    def _extract_ui_state(self, snapshot, thread_id: str) -> Dict[str, Any]:
        """Convert LangGraph state snapshot into UI-friendly structure."""
        state = snapshot.values
        return {
            "thread_id": thread_id,
            "status": self._sessions[thread_id]["status"],
            "current_chapter_index": state.get("current_chapter_index", 0),
            "global_plan": state.get("global_plan", []),
            "current_phase": state.get("current_phase", "diagnosis"),
            "chapter_title": state.get("chapter_title", ""),
            "chapter_question": state.get("chapter_question", ""),
            "current_draft": state.get("current_draft", ""),
            "chapter_scratchpad": state.get("chapter_scratchpad", {}),
            "context_pool": state.get("context_pool", []),
            "strategic_blueprint": state.get("strategic_blueprint"),
            "is_blueprint_phase": self._is_blueprint_phase(snapshot),
            "final_report": state.get("final_report"),
            "review_decision": state.get("review_decision", ""),
            "human_feedback": state.get("human_feedback", {}),
            "llm_review_result": state.get("llm_review_result"),
        }

    def _is_blueprint_phase(self, snapshot) -> bool:
        """Check if workflow is paused for strategic blueprint review."""
        state = snapshot.values
        blueprint = state.get("strategic_blueprint")
        if not blueprint:
            return False
        return not blueprint.get("approved", False) and state.get("current_chapter_index", 0) >= 2
