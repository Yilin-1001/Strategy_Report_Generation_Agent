"""
Group 3: Full System (All modules enabled)

Uses rag_project's complete agent system with:
- Multi-agent workflow (coordinator, researcher, analyst, writer, strategist, archiver)
- Memory management via context_pool
- Strategic blueprint generation
- Auto-approve mode (no human review)

IMPORTANT: The graph uses interrupt_before=["human_review"], so the auto-approve
loop must replicate the state changes that human_review_node would make:
- Incrementing current_chapter_index
- Adding draft to context_pool
- Setting strategic_blueprint.approved = True
- Transitioning phase from diagnosis to initiatives
"""

import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup - ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ablation_experiment.config import TEST_QUERY, REPORTS_DIR
from rag_project.agent.graph import create_report_graph
from rag_project.agent.state import GraphState

logger = logging.getLogger("ablation.group3")


def _save_report(report_text: str, run_id: int) -> Path:
    """Persist the generated report to disk."""
    out_dir = REPORTS_DIR / "group3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{run_id}.md"
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Report saved to %s", out_path)
    return out_path


def run_group3(run_id: int) -> str:
    """Run Group 3: Full system with multi-agent + memory + blueprint.

    Uses rag_project's complete agent system in auto-approve mode.
    The graph executes: coordinator -> prepare_chapter -> researcher ->
    analyst -> writer -> [human_review] -> strategist -> archiver.

    Because we run in auto-approve mode, every interrupt at human_review
    is answered with a full state update that replicates what
    human_review_node would do, so the pipeline runs end-to-end without
    human interaction.

    Parameters
    ----------
    run_id : int
        Unique identifier for this experimental run (used for thread_id
        and output file naming).

    Returns
    -------
    str
        The final report text produced by the full system.
    """
    logger.info("=== Group 3 (Full System) - Run %d starting ===", run_id)

    # Reset global token registry for this run
    from rag_project.agent.llm_manager import reset_global_token_stats
    reset_global_token_stats()

    # ------------------------------------------------------------------
    # Build the LangGraph application
    # ------------------------------------------------------------------
    app = create_report_graph()
    config = {"configurable": {"thread_id": f"ablation_group3_run{run_id}"}}

    # ------------------------------------------------------------------
    # Kick off the first invocation.
    # The graph will run through coordinator, prepare_chapter, researcher,
    # analyst, writer and then stop *just before* human_review (because
    # interrupt_before=["human_review"]).
    # ------------------------------------------------------------------
    logger.info("Invoking graph with query: %s", TEST_QUERY)
    result = app.invoke({"user_input": TEST_QUERY}, config)
    logger.info("Initial invocation complete. current_chapter_index=%s",
                result.get("current_chapter_index"))

    # ------------------------------------------------------------------
    # Auto-approve loop — replicates human_review_node faithfully
    # ------------------------------------------------------------------
    max_iterations = 60
    for iteration in range(max_iterations):
        # Check for completion
        if result.get("final_report"):
            logger.info("Final report received at iteration %d", iteration)
            break

        # Read current state to make decisions
        current_index = result.get("current_chapter_index", 0)
        context_pool = result.get("context_pool", [])
        blueprint = result.get("strategic_blueprint") or {}
        current_draft = result.get("current_draft", "")
        global_plan = result.get("global_plan", [])
        scratchpad = result.get("chapter_scratchpad", {})
        last_approved_index = result.get("_last_approved_index", -1)

        # ----------------------------------------------------------
        # CASE A: After Chapter 3 + blueprint generated but not yet approved
        # ----------------------------------------------------------
        if (
            current_index == 2
            and len(context_pool) >= 3
            and blueprint
            and not blueprint.get("approved")
        ):
            # Replicate the approve_blueprint branch of human_review_node
            logger.info("Iteration %d: approving blueprint (chapter_index=%d)",
                        iteration, current_index)
            blueprint["approved"] = True
            state_update = {
                "review_decision": "approve_blueprint",
                "current_phase": "initiatives",
                "strategic_blueprint": blueprint,
                "chapter_scratchpad": {},
                "current_chapter_index": current_index + 1,
                # Transfer pending knowledge for context compression
                "_pending_chapter_knowledge": {},
            }

        # ----------------------------------------------------------
        # CASE B: Normal chapter approval (replicate "approve" branch)
        # ----------------------------------------------------------
        elif current_index != last_approved_index or last_approved_index == -1:
            logger.info("Iteration %d: approving chapter (chapter_index=%d)",
                        iteration, current_index)

            # Build the full chapter text with correct title
            if current_index < len(global_plan):
                chapter_meta = global_plan[current_index]
                if isinstance(chapter_meta, dict):
                    chapter_title = chapter_meta.get("title", f"Chapter {current_index + 1}")
                else:
                    chapter_title = str(chapter_meta)
            else:
                chapter_title = f"Chapter {current_index + 1}"

            if current_draft:
                draft_lines = current_draft.strip().split('\n')
                if draft_lines and draft_lines[0].strip().startswith('#'):
                    rest_of_draft = '\n'.join(draft_lines[1:]).strip()
                    full_chapter = f"# {chapter_title}\n\n{rest_of_draft}"
                else:
                    full_chapter = f"# {chapter_title}\n\n{current_draft}"
            else:
                full_chapter = f"# {chapter_title}"

            # Build _pending_chapter_knowledge from scratchpad (replicate lines 228-236)
            pending_knowledge = {}
            if scratchpad and scratchpad.get("key_facts"):
                pending_knowledge = {
                    "title": result.get("chapter_title", chapter_title),
                    "key_facts": scratchpad.get("key_facts", []),
                    "insights": scratchpad.get("insights", []),
                }

            state_update = {
                "review_decision": "approve",
                "context_pool": [full_chapter],  # operator.add
                "chapter_scratchpad": {},
                "_pending_chapter_knowledge": pending_knowledge,
                "_last_approved_index": current_index,
                "auto_revision_count": 0,
            }

            # Special: Chapter 3 (index 2) — don't increment, wait for blueprint
            if current_index == 2:
                logger.info("Chapter 3 approved — keeping index at 2, waiting for blueprint")
                # Do NOT increment current_chapter_index
            else:
                state_update["current_chapter_index"] = current_index + 1

        # ----------------------------------------------------------
        # CASE C: Already approved this chapter (edge case safety)
        # ----------------------------------------------------------
        else:
            logger.info("Iteration %d: chapter already approved, incrementing index",
                        iteration)
            state_update = {
                "review_decision": "approve",
                "chapter_scratchpad": {},
                "current_chapter_index": current_index + 1,
                "auto_revision_count": 0,
            }

        # Update the graph state and resume execution
        app.update_state(
            config,
            state_update,
            as_node="human_review",
        )
        result = app.invoke(None, config)

        logger.debug(
            "Iteration %d complete - current_chapter_index=%s, "
            "context_pool size=%d, has_final_report=%s",
            iteration,
            result.get("current_chapter_index"),
            len(result.get("context_pool", [])),
            bool(result.get("final_report")),
        )
    else:
        logger.warning(
            "Reached max iterations (%d) without producing a final report. "
            "context_pool size=%d, current_chapter_index=%s",
            max_iterations,
            len(result.get("context_pool", [])),
            result.get("current_chapter_index"),
        )

    # ------------------------------------------------------------------
    # Extract and save the report
    # ------------------------------------------------------------------
    report_text = result.get("final_report", "")

    if not report_text:
        logger.error("No final_report produced by the full system")
        # Fallback: assemble from context_pool
        if result.get("context_pool"):
            logger.info("Assembling fallback report from context_pool (%d chapters)",
                        len(result["context_pool"]))
            report_text = f"# 江西交通投资集团战略规划报告\n\n"
            report_text += f"**主题**: {TEST_QUERY}\n\n---\n\n"
            for chapter in result["context_pool"]:
                report_text += f"{chapter}\n\n---\n\n"

    report_path = _save_report(report_text, run_id)

    # Collect token stats from all LLM agents in the graph
    # We need to re-import the graph to access the LLM instances
    # Instead, we'll aggregate from the graph's node functions
    total_prompt = 0
    total_completion = 0
    total_total = 0
    total_calls = 0
    agent_stats = {}

    try:
        from rag_project.agent.nodes import (
            coordinator_node, prepare_chapter_node, researcher_node,
            analyst_node, writer_node, archiver_node
        )
        from rag_project.agent.nodes.strategist import strategist_node

        # Access the LLM managers from the graph module
        from rag_project.agent import graph as graph_module

        # The LLM managers are created in create_report_graph() as local vars
        # We can't access them directly, so we use a different approach:
        # Read token stats from a global registry if available
        import rag_project.agent.llm_manager as llm_mgr
        # Check for a global token registry
        if hasattr(llm_mgr, '_global_token_registry'):
            for name, stats in llm_mgr._global_token_registry.items():
                agent_stats[name] = stats
                total_prompt += stats["prompt_tokens"]
                total_completion += stats["completion_tokens"]
                total_total += stats["total_tokens"]
                total_calls += stats["call_count"]
    except Exception as e:
        logger.debug(f"Could not collect per-agent token stats: {e}")

    token_stats = {
        "call_count": total_calls,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_total,
        "per_agent": agent_stats,
    }

    logger.info("=== Group 3 (Full System) - Run %d complete ===", run_id)
    logger.info("Report length: %d characters", len(report_text))
    if total_total > 0:
        print(f"  Token usage: {total_total:,} total "
              f"(prompt: {total_prompt:,}, completion: {total_completion:,}, "
              f"calls: {total_calls})")
        for name, s in agent_stats.items():
            print(f"    {name}: {s['total_tokens']:,} tokens ({s['call_count']} calls)")

    return report_text, token_stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    report = run_group3(run_id=0)
    print(f"\nReport preview (first 500 chars):\n{report[:500]}")
