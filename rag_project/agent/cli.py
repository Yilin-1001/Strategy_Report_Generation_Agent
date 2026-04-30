"""
CLI interface for the report generation agent system.

Provides interactive and automated modes for running the multi-agent workflow.
Enhanced with LLM + Human combined review (LLM reviews first, human decides final).
Uses an independent reviewer model to avoid self-evaluation hallucination.
"""

import json
import os
from typing import Optional, Dict, Any
from openai import OpenAI

from rag_project.agent.graph import create_report_graph
from rag_project.agent.state import GraphState
from rag_project.agent.llm_manager import LLMManager
from rag_project.utils.logger import setup_logger
from rag_project.utils.config_loader import load_config

logger = setup_logger(__name__)


def _confirm_or_go_back(summary: str) -> bool:
    """Show confirmation prompt with undo. Returns True to confirm, False to go back."""
    print(f"  >> {summary}")
    while True:
        ans = input("  Confirm? (y=yes / b=back): ").strip().lower()
        if ans in ("y", "yes", ""):
            return True
        elif ans in ("b", "back", "n", "no"):
            print("  [UNDO] Going back to selection...\n")
            return False
        else:
            print("  Enter y to confirm, b to go back.\n")


class ReportGeneratorCLI:
    """
    Command-line interface for the report generation agent system.

    Supports two modes:
    - Interactive: Prompts user for feedback at each review point
    - Auto: Automatically approves all feedback without user input

    Attributes:
        output_path: Path to save the final report
        auto_mode: If True, automatically approve all feedback
        app: Compiled LangGraph application
    """

    def __init__(self):
        """
        Initialize the CLI.

        Initializes the report graph, default configuration,
        and an independent reviewer client using a different model.
        """
        self.app = create_report_graph()
        self.config = {"configurable": {"thread_id": "report_session_001"}}

        # Initialize independent reviewer using SiliconFlow Qwen model
        # This avoids self-evaluation hallucination by using a different model
        try:
            reviewer_config = load_config("config/agent_config.yaml").get("reviewer", {})
            reviewer_api_key = os.environ.get(reviewer_config.get("api_key_env", "SILICONFLOW_API_KEY"))
            reviewer_timeout = reviewer_config.get("timeout", 180)
            self.reviewer_max_retries = reviewer_config.get("max_retries", 2)
            if reviewer_api_key:
                self.reviewer_client = OpenAI(
                    api_key=reviewer_api_key,
                    base_url=reviewer_config.get("base_url", "https://api.siliconflow.cn/v1"),
                    timeout=reviewer_timeout
                )
                self.reviewer_model = reviewer_config.get("model", "Qwen/Qwen2.5-72B-Instruct")
                logger.info(f"Initialized independent reviewer: {self.reviewer_model} (timeout={reviewer_timeout}s, max_retries={self.reviewer_max_retries})")
            else:
                logger.warning("SILICONFLOW_API_KEY not set, falling back to coordinator LLM for review")
                self.reviewer_client = None
                self.reviewer_model = None
                self.reviewer_max_retries = 0
        except Exception as e:
            logger.warning(f"Failed to init independent reviewer: {e}. Falling back to coordinator LLM.")
            self.reviewer_client = None
            self.reviewer_model = None
            self.reviewer_max_retries = 0

        # Fallback: use coordinator LLM if reviewer is unavailable
        if self.reviewer_client is None:
            self.reviewer_llm = LLMManager("coordinator")

        logger.info("Initialized CLI with report_session_001")

    def generate_report(self, user_input: str, auto_mode: bool = False) -> str:
        """
        Run the report generation workflow.

        Args:
            user_input: The user's request/question
            auto_mode: If True, automatically approve all feedback

        Returns:
            str: The final generated report

        Raises:
            Exception: If workflow execution fails
        """
        print(f"\n{'='*60}")
        print(f"[REPORT] Report Generation Started")
        print(f"{'='*60}")
        print(f"Request: {user_input}")
        print(f"Mode: {'Auto' if auto_mode else 'Interactive'}")
        print(f"{'='*60}\n")

        try:
            # Start the workflow - runs coordinator through writer,
            # then pauses before human_review (due to interrupt_before)
            print("[START] Starting workflow...\n")
            self._stream_events({"user_input": user_input})

            # Main interrupt-handling loop:
            # Each iteration: collect feedback → update state → resume graph
            while True:
                current_state = self.app.get_state(self.config)

                # No more nodes to execute → workflow complete
                if not current_state.next:
                    break

                # Final report already generated
                if current_state.values.get("final_report"):
                    break

                # Collect user feedback at the interrupt point
                feedback = self._get_user_feedback(current_state.values, auto_mode)

                if not feedback:
                    print("[SKIP] Skipping feedback...")
                    break

                print("[FEEDBACK] Feedback received, continuing...")

                # Map feedback decision to review_decision expected by human_review_node
                decision = feedback["decision"]
                review_decision = "approve" if decision == "skip" else decision

                # Inject the review decision into graph state so human_review_node
                # can read it when the graph resumes
                self.app.update_state(
                    self.config,
                    {
                        "review_decision": review_decision,
                        "human_feedback": feedback,
                    }
                )

                # Resume the graph from the interrupt point.
                # human_review_node executes, then routing continues to the next
                # chapter or to the archiver for final report assembly.
                self._stream_events(None)

            # Get the final state
            final_state = self.app.get_state(self.config)
            final_report = final_state.values.get("final_report", "")

            if not final_report:
                raise ValueError("No final report generated")

            # 执行摘要质检（Qwen 独立评审）
            final_report = self._review_executive_summary(final_report, final_state)

            print(f"\n{'='*60}")
            print(f"[DONE] Report Generation Complete!")
            print(f"{'='*60}\n")

            return final_report

        except Exception as e:
            logger.error(f"Error during report generation: {e}")
            print(f"\n[ERROR] Error: {e}")
            raise

    def _stream_events(self, input_data) -> None:
        """
        Stream graph events and display progress indicators.

        Runs the graph until it completes or pauses at an interrupt.
        Displays progress for each node that completes.

        Args:
            input_data: Input dict for initial run, or None to resume from interrupt
        """
        for event in self.app.stream(
            input_data,
            self.config,
            stream_mode="values"
        ):
            if "current_chapter_index" in event:
                chapter_idx = event.get("current_chapter_index", 0)
                print(f"[CHAPTER] Processing chapter {chapter_idx + 1}...")

            if "chapter_title" in event and event["chapter_title"]:
                print(f"[SEARCH] Researching: {event['chapter_title']}")

            if "current_draft" in event and event["current_draft"]:
                chapter = event.get("chapter_title", "Unknown")
                print(f"[OK] Completed: {chapter}")

    def _get_user_feedback(self, state: Dict[str, Any], auto_mode: bool = False) -> Dict:
        """
        Get user feedback on the current draft with LLM pre-review.

        在交互模式下，先展示 LLM 评审结果，再让用户选择。
        用户可采纳 LLM 建议或自行决定。

        In auto mode, returns LLM-based decision (max 1 auto-revision per chapter).
        In interactive mode, prompts user for input with LLM suggestion reference.

        Automatically detects when a strategic blueprint is pending review
        (instead of a chapter) and presents appropriate options.

        Args:
            state: Current workflow state
            auto_mode: If True, use LLM decision

        Returns:
            Dict with 'decision', 'feedback_type', and 'comments' keys
        """
        # Detect strategic blueprint review mode
        blueprint = state.get("strategic_blueprint")
        current_index = state.get("current_chapter_index", 0)
        if (blueprint
                and isinstance(blueprint, dict)
                and blueprint.get("mission")  # Blueprint has been generated (not empty)
                and not blueprint.get("approved", False)
                and current_index >= 2):
            return self._get_blueprint_feedback(blueprint, auto_mode)

        chapter = state.get("chapter_title", "Unknown")
        draft = state.get("current_draft", "")

        # === LLM Pre-Review (Agent化决策点) ===
        llm_review = self._llm_evaluate_draft(draft, chapter, state)

        # Display draft
        print(f"\n{'─'*60}")
        print(f"[READ] Chapter: {chapter}")
        print(f"{'─'*60}")
        print(f"{draft}\n")
        print(f"{'─'*60}\n")

        # Display LLM Review result with dimension scores table
        issues_str = "; ".join(llm_review["issues"][:3]) if llm_review["issues"] else "None"
        dim_scores = llm_review.get("dimension_scores", {})
        hints = llm_review.get("improvement_hints", {})

        print(f"{'━'*60}")
        print(f"  Qwen 质检报告")
        print(f"{'━'*60}")
        print(f"  总分: {llm_review['score']}/100  |  建议: {llm_review['suggestion']}")
        print(f"{'─'*60}")

        if dim_scores:
            dim_names = {
                "topic_relevance": ("主题契合度", 15),
                "analysis_depth": ("分析深度", 20),
                "writing_quality": ("写作专业度", 15),
                "citation_sufficiency": ("引用充分性", 15),
                "groundedness": ("内容真实性", 20),
                "context_coherence": ("上下文连贯", 15),
            }
            print(f"  {'维度':<12} {'得分':>4} / {'满分':>3}   {'状态':<6}")
            print(f"  {'─'*42}")
            for key, (name, max_score) in dim_names.items():
                val = dim_scores.get(key, "")
                if val != "":
                    ratio = val / max_score if max_score > 0 else 0
                    status = "✓ 合格" if ratio >= 0.6 else "✗ 扣分"
                    print(f"  {name:<10} {val:>4} / {max_score:>3}   {status}")
            print(f"  {'─'*42}")
        else:
            print(f"  (无维度评分数据)")

        print(f"{'─'*60}")
        if llm_review["issues"]:
            print(f"  问题: {issues_str}")
        if hints.get("researcher"):
            print(f"  检索建议: {hints['researcher'][:80]}")
        if hints.get("analyst"):
            print(f"  分析建议: {hints['analyst'][:80]}")
        if hints.get("writer"):
            print(f"  写作建议: {hints['writer'][:80]}")
        print(f"{'━'*60}\n")

        # === Auto Mode: Score-based decision, skip edit prompt ===
        if auto_mode:
            auto_revision_count = state.get("auto_revision_count", 0)

            if llm_review["score"] >= 60 or auto_revision_count >= 1:
                # Approve: score >= 60 (auto mode threshold) OR already auto-revised once
                print(f"[AUTO] LLM decision: Approve (score={llm_review['score']}/100)\n")
                return {
                    "decision": "approve",
                    "feedback_type": "auto_llm",
                    "comments": f"LLM score: {llm_review['score']}/100. Issues: {issues_str}"
                }
            else:
                # Auto-revise: route to appropriate revision node
                print(f"[AUTO] LLM decision: {llm_review['suggestion']} (score={llm_review['score']}/100, auto-revision #{auto_revision_count + 1})\n")
                return {
                    "decision": llm_review["suggestion"],
                    "feedback_type": "auto_llm",
                    "comments": f"LLM auto-revision: {issues_str}",
                    "llm_review": llm_review
                }

        # === Human Edit Review (interactive mode only) ===
        try:
            edit_choice = input("  Edit review before proceeding? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            edit_choice = "n"

        if edit_choice in ("y", "yes"):
            llm_review = self._edit_review_fields(llm_review)
            # Refresh display variables after editing
            issues_str = "; ".join(llm_review["issues"][:3]) if llm_review["issues"] else "None"
            hints = llm_review.get("improvement_hints", {})
            print()

        # === Interactive Mode: User decides with LLM suggestion ===
        llm_suggest_choice = "3"  # Option 3 for LLM suggestion
        llm_suggest_text = f"Accept LLM suggestion: {llm_review['suggestion']}"

        while True:  # outer loop: allows full undo back to option selection
            print("Please review this chapter:")
            print("  1. Approve - Continue to next chapter")
            print("  2. Revise - Manually specify revision type")
            print(f"  3. {llm_suggest_text}")
            print("  4. Skip - Skip this chapter\n")
            try:
                choice = input("Your choice (1/2/3/4): ").strip()

                if choice == "1":
                    comments = input("Optional comments (press Enter to skip): ").strip()
                    if _confirm_or_go_back(
                        f"Approve: {chapter}" +
                        (f" | {comments}" if comments else "")
                    ):
                        print()
                        return {
                            "decision": "approve",
                            "feedback_type": "user",
                            "comments": comments or "Approved"
                        }

                elif choice == "2":
                    comments = input("Revision instructions: ").strip()
                    if not comments:
                        print("[X] Revision instructions required. Please try again.\n")
                        continue

                    # Ask revision type
                    print("\n  Revision type:")
                    print("    1. Writing  - Improve writing style and expression")
                    print("    2. Logic    - Re-analyze with different perspective")
                    print("    3. Data     - Re-research and find better data\n")
                    while True:
                        type_choice = input("  Type (1/2/3, default=1): ").strip()
                        if type_choice == "" or type_choice == "1":
                            feedback_type = "writing"
                            break
                        elif type_choice == "2":
                            feedback_type = "logic"
                            break
                        elif type_choice == "3":
                            feedback_type = "data"
                            break
                        else:
                            print("  [X] Invalid. Enter 1, 2, or 3.\n")
                            continue

                    if _confirm_or_go_back(
                        f"Revise ({feedback_type}): {chapter} | {comments[:50]}"
                    ):
                        print()
                        return {
                            "decision": f"revise:{feedback_type}",
                            "feedback_type": "user",
                            "comments": comments,
                            "llm_review": llm_review
                        }

                elif choice == "3":
                    # Accept LLM suggestion
                    if _confirm_or_go_back(f"Accept LLM suggestion: {llm_review['suggestion']}"):
                        print()
                        return {
                            "decision": llm_review["suggestion"],
                            "feedback_type": "user_llm",
                            "comments": f"Accepted LLM suggestion (score={llm_review['score']}/100)",
                            "llm_review": llm_review
                        }

                elif choice == "4":
                    if _confirm_or_go_back(f"Skip: {chapter}"):
                        print()
                        return {
                            "decision": "skip",
                            "feedback_type": "user",
                            "comments": "Skipped by user"
                        }

                else:
                    print("[X] Invalid choice. Please enter 1, 2, 3, or 4.\n")

            except (EOFError, KeyboardInterrupt):
                print("\n\n[WARN] Interrupt detected. Approving by default...\n")
                return {
                    "decision": "approve",
                    "feedback_type": "interrupt",
                    "comments": "Auto-approved due to interrupt"
                }

    def _edit_review_fields(self, llm_review: Dict) -> Dict:
        """
        Allow user to selectively edit LLM review fields.

        Displays editable fields, accepts multi-select input,
        prompts for new values, and confirms changes.
        Supports 'b' to go back and re-edit.

        Args:
            llm_review: Original LLM review dict

        Returns:
            Modified llm_review dict (may be unchanged if user skips)
        """
        # Work on a copy so we can discard on 'back'
        import copy
        working = copy.deepcopy(llm_review)
        hints = working.get("improvement_hints", {})

        # Define editable fields: (label, getter, setter)
        editable_fields = [
            ("suggestion", lambda d: d.get("suggestion", ""),
             lambda d, v: d.__setitem__("suggestion", v)),
            ("issues", lambda d: "; ".join(d.get("issues", [])),
             lambda d, v: d.__setitem__("issues", [s.strip() for s in v.split(";") if s.strip()])),
            ("researcher", lambda d: d.get("improvement_hints", {}).get("researcher", ""),
             lambda d, v: d.setdefault("improvement_hints", {}).__setitem__("researcher", v)),
            ("analyst", lambda d: d.get("improvement_hints", {}).get("analyst", ""),
             lambda d, v: d.setdefault("improvement_hints", {}).__setitem__("analyst", v)),
            ("writer", lambda d: d.get("improvement_hints", {}).get("writer", ""),
             lambda d, v: d.setdefault("improvement_hints", {}).__setitem__("writer", v)),
        ]

        while True:  # Outer loop: re-edit on 'back'
            # Display editable fields
            print(f"\n  {'─'*46}")
            print(f"  Edit Review Fields:")
            print(f"  {'─'*46}")
            for i, (label, getter, _) in enumerate(editable_fields, 1):
                current_val = getter(working)
                # Truncate long values for display
                display_val = current_val if len(str(current_val)) <= 35 else str(current_val)[:32] + "..."
                print(f"    {i}. {label:<12}: {display_val}")
            print(f"  {'─'*46}")

            # Get field selection
            try:
                selection = input("  Fields to edit (e.g. 1,3,5 / b=back / Enter=done): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return llm_review  # Return original on interrupt

            if selection.lower() in ("b", "back"):
                continue  # Restart outer loop to re-display
            if not selection:
                return working  # No edits, return (possibly modified from previous loop)

            # Parse field numbers
            try:
                indices = []
                for part in selection.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part)
                        if 1 <= idx <= len(editable_fields):
                            indices.append(idx)
                        else:
                            print(f"  [X] Invalid field number: {idx}. Valid range: 1-{len(editable_fields)}")
                    elif part:
                        print(f"  [X] Invalid input: {part}")
                if not indices:
                    print("  No valid fields selected.\n")
                    continue
            except Exception:
                print("  [X] Invalid input. Use comma-separated numbers.\n")
                continue

            # Edit each selected field
            for idx in indices:
                label, getter, setter = editable_fields[idx - 1]
                current_val = getter(working)

                print(f"\n  → Editing field {idx}: {label}")
                print(f"    Current: {current_val}")

                if label == "suggestion":
                    print(f"    Options: approve / revise:data / revise:logic / revise:writing")
                    new_val = input("    New value (Enter to keep): ").strip()
                    if new_val and new_val in ("approve", "revise:data", "revise:logic", "revise:writing"):
                        setter(working, new_val)
                    elif new_val:
                        print(f"    [X] Invalid suggestion. Keeping: {current_val}")
                    # else: empty input, keep current
                elif label == "issues":
                    new_val = input("    New value (semicolon-separated, Enter to keep): ").strip()
                    if new_val:
                        setter(working, new_val)
                    # else: keep current
                else:
                    new_val = input("    New value (Enter to keep): ").strip()
                    if new_val:
                        setter(working, new_val)
                    # else: keep current

            # Show summary of changes
            print(f"\n  {'─'*46}")
            print(f"  Updated Review:")
            print(f"  {'─'*46}")
            for i, (label, getter, _) in enumerate(editable_fields, 1):
                new_val = getter(working)
                orig_val = editable_fields[i - 1][1](llm_review)
                changed = " (modified)" if new_val != orig_val else ""
                display_val = new_val if len(str(new_val)) <= 35 else str(new_val)[:32] + "..."
                print(f"    {i}. {label:<12}: {display_val}{changed}")
            print(f"  {'─'*46}")

            # Confirm or go back
            try:
                confirm = input("  Confirm changes? (y=yes / b=back to re-edit): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return llm_review

            if confirm in ("b", "back"):
                # Reset working copy and re-edit
                working = copy.deepcopy(llm_review)
                continue
            # Any other input (including 'y', empty, etc.) = confirm
            return working

    def _get_blueprint_feedback(self, blueprint: Dict[str, Any], auto_mode: bool = False) -> Dict:
        """
        Get user feedback on the strategic blueprint.

        Displays the blueprint content (mission, SWOT, TOWS, pillars, KPIs)
        and collects approval or revision request.

        Args:
            blueprint: Strategic blueprint dict
            auto_mode: If True, automatically approve

        Returns:
            Dict with 'decision', 'feedback_type', and 'comments' keys
        """
        # Format blueprint for display
        print(f"\n{'─' * 60}")
        print("[BLUEPRINT] Strategic Blueprint Review")
        print(f"{'─' * 60}")

        # Mission
        mission = blueprint.get("mission", "N/A")
        print(f"\n[TARGET] Core Mission: {mission}")

        # SWOT
        swot = blueprint.get("swot_analysis", {})
        if swot:
            print("\n[SWOT] SWOT Analysis:")
            for cat, label in [("strengths", "Strengths"), ("weaknesses", "Weaknesses"),
                               ("opportunities", "Opportunities"), ("threats", "Threats")]:
                items = swot.get(cat, swot.get(label, []))
                if items:
                    print(f"  {label}:")
                    for item in items:
                        print(f"    - {item}")

        # TOWS strategies
        tows = blueprint.get("tows_strategies", {})
        if tows:
            print("\n[TOWS] TOWS Strategies:")
            for key in ["SO", "WO", "ST", "WT"]:
                strategies = tows.get(key, [])
                if strategies:
                    print(f"  {key}:")
                    for s in strategies:
                        print(f"    - {s}")

        # Strategic pillars
        pillars = blueprint.get("strategic_pillars", [])
        if pillars:
            print("\n[PILLARS] Strategic Pillars:")
            for p in pillars:
                print(f"  - {p}")

        # KPIs
        kpis = blueprint.get("kpis", {})
        if kpis:
            print("\n[KPIs] KPIs (BSC Dimensions):")
            for dim, indicators in kpis.items():
                if indicators:
                    print(f"  {dim}:")
                    for name, target in indicators.items():
                        print(f"    - {name}: {target}")

        print(f"\n{'─' * 60}\n")

        if auto_mode:
            print("[AUTO] Auto mode: Approving blueprint...\n")
            return {
                "decision": "approve_blueprint",
                "feedback_type": "auto",
                "comments": "Auto-approved"
            }

        # Interactive mode
        while True:
            print("Please review the strategic blueprint:")
            print("  1. Approve - Proceed to initiatives phase")
            print("  2. Revise  - Request blueprint revision with comments\n")
            try:
                choice = input("Your choice (1/2): ").strip()

                if choice == "1":
                    if _confirm_or_go_back("Approve strategic blueprint"):
                        print()
                        return {
                            "decision": "approve_blueprint",
                            "feedback_type": "user",
                            "comments": "Blueprint approved"
                        }

                elif choice == "2":
                    comments = input("Revision instructions: ").strip()
                    if not comments:
                        print("[X] Revision instructions required. Please try again.\n")
                        continue

                    if _confirm_or_go_back(f"Revise blueprint | {comments[:50]}"):
                        print()
                        return {
                            "decision": "revise_blueprint",
                            "feedback_type": "user",
                            "comments": comments
                        }
                else:
                    print("[X] Invalid choice. Please enter 1 or 2.\n")

            except (EOFError, KeyboardInterrupt):
                print("\n\n[WARN] Interrupt detected. Approving blueprint by default...\n")
                return {
                    "decision": "approve_blueprint",
                    "feedback_type": "interrupt",
                    "comments": "Auto-approved due to interrupt"
                }

    def _llm_evaluate_draft(self, draft: str, chapter_title: str, state: Dict[str, Any]) -> Dict:
        """
        LLM 评审草稿质量，返回多维度评分和针对性改进建议。

        使用独立的评审模型（硅基流动 Qwen）避免自评幻觉。
        评估 6 个维度：主题契合度、分析深度、写作专业度、引用充分性、内容真实性、上下文连贯性。

        根据阶段（诊断/蓝图/推演）使用不同的评分侧重点。

        Args:
            draft: 章节草稿文本
            chapter_title: 章节标题
            state: 当前工作流状态

        Returns:
            Dict with:
                - score: 总分 1-10
                - dimension_scores: 各维度评分 dict
                - issues: 问题列表
                - suggestion: approve / revise:data / revise:logic / revise:writing
                - improvement_hints: 针对各节点的改进提示 {researcher, analyst, writer}
        """
        # 统计草稿基本信息
        chinese_chars = sum(1 for c in draft if '\u4e00' <= c <= '\u9fff')
        citations = len(__import__('re').findall(r'\[来源:', draft))

        global_plan = state.get("global_plan", [])
        current_index = state.get("current_chapter_index", 0)
        chapter_meta = global_plan[current_index] if current_index < len(global_plan) else {}
        analysis_model = chapter_meta.get("analysis_model", "") if isinstance(chapter_meta, dict) else ""
        phase = chapter_meta.get("phase", "") if isinstance(chapter_meta, dict) else ""

        # 新增：读取上下文和蓝图用于评估连贯性和对齐度
        context_summary = state.get("context_summary", "")
        strategic_blueprint = state.get("strategic_blueprint", {})

        # 构建阶段感知的评分 prompt
        prompt = self._build_review_prompt(
            draft=draft,
            chapter_title=chapter_title,
            phase=phase,
            analysis_model=analysis_model,
            citations=citations,
            chinese_chars=chinese_chars,
            context_summary=context_summary,
            strategic_blueprint=strategic_blueprint
        )

        try:
            # Use independent reviewer client (SiliconFlow Qwen) if available
            if self.reviewer_client is not None:
                # Retry logic for reliability
                last_error = None
                for attempt in range(self.reviewer_max_retries + 1):
                    try:
                        response = self.reviewer_client.chat.completions.create(
                            model=self.reviewer_model,
                            messages=[
                                {"role": "system", "content": "你是国企战略规划报告评审专家，严格按照JSON格式输出评审结果。"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1024,
                            response_format={"type": "json_object"},
                            extra_body={'enable_thinking': False}
                        )
                        response_text = response.choices[0].message.content or ""
                        logger.info(f"Review by independent model: {self.reviewer_model} (attempt {attempt + 1})")
                        break
                    except Exception as retry_error:
                        last_error = retry_error
                        if attempt < self.reviewer_max_retries:
                            import time
                            wait_time = 5 * (attempt + 1)
                            logger.warning(f"Review attempt {attempt + 1} failed: {retry_error}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise last_error
            else:
                # Fallback to coordinator LLM
                response_text = self.reviewer_llm.invoke(prompt, temperature=0.3, max_tokens=1024)
                logger.info("Review by fallback coordinator LLM")

            # 解析 JSON 响应
            result = self._parse_review_response(response_text, citations, chinese_chars)
            logger.info(f"LLM Review: score={result['score']}, suggestion={result['suggestion']}, issues={result['issues']}")
            return result

        except Exception as e:
            logger.warning(f"LLM review failed: {e}. Defaulting to approve.")
            return {
                "score": 70,
                "dimension_scores": {},
                "issues": [],
                "suggestion": "approve",
                "improvement_hints": {}
            }

    def _build_review_prompt(
        self,
        draft: str,
        chapter_title: str,
        phase: str,
        analysis_model: str,
        citations: int,
        chinese_chars: int,
        context_summary: str,
        strategic_blueprint: Dict
    ) -> str:
        """
        根据阶段构建差异化的评审 prompt。

        Args:
            draft: 草稿文本
            chapter_title: 章节标题
            phase: 当前阶段 (diagnosis/initiatives)
            analysis_model: 分析模型
            citations: 引用数量
            chinese_chars: 中文字数
            context_summary: 上下文摘要
            strategic_blueprint: 战略蓝图

        Returns:
            格式化的评审 prompt
        """
        # 基础评分标准（所有阶段通用）
        base_criteria = f"""
## 评分维度与标准（满分100分，总分=各维度得分之和，≥70分为合格）

### 1. 主题契合度（15分）
章节内容是否紧扣标题和研究问题，是否存在偏离。
- 13-15分（优秀）：每段内容都直接服务于研究问题，无任何偏离
- 10-12分（良好）：基本扣题，偶有轻微延伸但不影响主线
- 7-9分（一般）：约30%内容偏离主题，存在与研究问题关联不强的段落
- 0-6分（差）：严重跑题，大部分内容与标题和问题无关

### 2. 分析深度（20分）
是否充分运用{analysis_model}分析框架，分析维度是否完整，是否有深入洞察。
- 17-20分（优秀）：分析模型所有维度完整覆盖，各维度有深入分析和独到洞察，逻辑链完整
- 13-16分（良好）：主要维度覆盖，分析有一定深度但个别维度较浅
- 8-12分（一般）：仅覆盖部分维度，分析停留在表面陈述，缺乏深入推理
- 0-7分（差）：未体现分析模型框架，内容仅为事实罗列，无分析逻辑

### 3. 写作专业度（15分）
是否符合国企公文语态和报告规范。
- 13-15分（优秀）：全文使用规范公文语态（"深入贯彻"、"全面落实"、"扎实推进"），结构清晰（总-分-总），语言凝练权威
- 10-12分（良好）：大部分符合公文风格，偶有口语化表述，结构基本完整
- 7-9分（一般）：语态不够统一，部分段落过于随意或学术化，结构有缺失
- 0-6分（差）：大量口语化表述，无国企公文特征，结构混乱

### 4. 引用充分性（15分）
引用数量与质量。当前引用{citations}个。
- 13-15分（优秀）：引用≥5个，全部为真实文件名，引用位置恰当且支撑对应论点
- 10-12分（良好）：引用4个，或≥5个但个别引用位置不够精准
- 7-9分（一般）：引用2-3个，支撑不够充分，或存在通用占位引用（如"来源文档_X"）
- 0-6分（差）：引用0-1个，或引用全部为占位符，无法溯源

### 5. 内容真实性（20分）
数据、政策名称、事实陈述是否有据可查，幻觉程度评估。
- 17-20分（优秀）：所有数据和政策名称均可溯源至引用文档，无幻觉内容
- 13-16分（良好）：大部分数据可溯源，偶有无法验证的表述但不影响结论
- 8-12分（一般）：存在1-2处明显无法验证的数据或政策名称，可能为幻觉
- 0-7分（差）：多处数据或事实明显虚构，政策名称与实际不符，严重幻觉

### 6. 上下文连贯性（15分）
与前序章节的数据和结论是否一致，是否存在矛盾或重复。
"""

        # 第一章特殊处理：无上下文，自动满分
        if not context_summary:
            base_criteria += """
**注意**：本章节为首章，无前序章节内容，因此：
- 上下文连贯性维度自动计为满分（15分）
- 总分基数为85分（其他5个维度），最终换算为100分制
- 评分时只需评估前5个维度，第6维度固定15分
"""
        else:
            base_criteria += """
- 13-15分（优秀）：与前序章节数据完全一致，在不重复的前提下自然深化，逻辑连贯
- 10-12分（良好）：基本一致，偶有轻微不一致但已有说明
- 7-9分（一般）：存在1处明显数据矛盾或大段重复前序内容
- 0-6分（差）：多处数据矛盾，大量重复前序章节，或与前序结论冲突且无解释
"""

        # 阶段差异化评估重点
        phase_specific = ""
        if phase == "diagnosis":
            phase_specific = f"""
## 诊断阶段特别评估

本章节属于诊断阶段，重点评估：
1. **信息覆盖度**: 检索信息是否准确覆盖章节主题
2. **分析模型完整性**: {analysis_model}的各维度是否都有分析内容
3. **数据准确性**: 政策名称、数据指标是否准确可查

前序章节摘要（用于评估连贯性）：
{context_summary[:500] if context_summary else "（无，为首章）"}
"""
        elif phase == "initiatives":
            # 提取蓝图关键信息
            mission = strategic_blueprint.get("mission", "") if strategic_blueprint else ""
            pillars = strategic_blueprint.get("strategic_pillars", []) if strategic_blueprint else []
            blueprint_info = f"""
核心使命: {mission}
战略支柱: {'; '.join(pillars[:3]) if pillars else '未设定'}
""" if strategic_blueprint else "（蓝图未生成或未批准）"

            phase_specific = f"""
## 推演阶段特别评估

本章节属于推演阶段，重点评估：
1. **蓝图对齐度**: 是否显式引用核心使命或战略支柱
2. **举措可执行性**: 提出的举措是否具体、可落地、有时限
3. **逻辑一致性**: 是否与前序诊断结论呼应

战略蓝图摘要：
{blueprint_info}

前序章节摘要：
{context_summary[:500] if context_summary else "（无）"}
"""
        else:
            phase_specific = f"""
前序章节摘要：
{context_summary[:500] if context_summary else "（无）"}
"""

        # 篇幅检查
        length_check = f"""
## 篇幅检查
- 目标字数: 1000-1800字
- 当前字数: {chinese_chars}字
- {"✓ 符合要求" if 1000 <= chinese_chars <= 1800 else "✗ 需调整篇幅"}
"""

        return f"""你是国企战略规划报告评审专家。请对以下章节进行专业评审。

## 章节信息
标题: {chapter_title}
阶段: {phase}
分析模型: {analysis_model}
引用数量: {citations}个
中文字数: {chinese_chars}字

{base_criteria}

{phase_specific}

{length_check}

## 草稿内容（前2000字）
{draft[:2000]}

---

## 输出要求

请严格按以下JSON格式输出评审结果：

```json
{{
    "total_score": <总分1-100>,
    "dimension_scores": {{
        "topic_relevance": <0-15>,
        "analysis_depth": <0-20>,
        "writing_quality": <0-15>,
        "citation_sufficiency": <0-15>,
        "groundedness": <0-20>,
        "context_coherence": <0-15>
    }},
    "issues": ["问题1", "问题2", ...],
    "suggestion": "approve 或 revise:data 或 revise:logic 或 revise:writing",
    "improvement_hints": {{
        "researcher": "针对数据检索的具体改进建议，如缺失哪类信息",
        "analyst": "针对分析的具体改进建议，如哪个维度不完整",
        "writer": "针对写作的具体改进建议，如语态或结构问题"
    }}
}}
```

**评分标准**:
- 总分≥70分 → approve
- 总分50-69分 → revise:xxx（根据主要问题类型选择）
- 总分<50分 → revise:writing

**注意**:
1. 如果引用数量<5个，citation_sufficiency必须扣分
2. 如果存在明显幻觉或无法溯源的数据，groundedness必须扣分
3. 如果分析模型维度不完整，analysis_depth必须扣分
4. improvement_hints要具体，不要泛泛而谈

只输出JSON，不要包含其他内容。
"""

    def _parse_review_response(self, response_text: str, citations: int, chinese_chars: int) -> Dict:
        """
        解析评审响应，多级容错兼容各种格式。

        解析策略：
        1. 直接 JSON 解析
        2. 提取 JSON 块后解析
        3. 修复常见 JSON 错误（尾逗号、注释）后解析
        4. 正则提取各字段
        5. 最终回退到旧格式 SCORE/ISSUES/SUGGESTION

        Args:
            response_text: LLM 响应文本
            citations: 引用数量
            chinese_chars: 中文字数

        Returns:
            结构化的评审结果
        """
        import json
        import re

        # 清理响应 - 去除 markdown 代码块标记
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 策略1: 直接解析
        result = self._try_parse_json(text)
        if result:
            return result

        # 策略2: 提取 { } 块后解析
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            result = self._try_parse_json(brace_match.group(0))
            if result:
                return result

        # 策略3: 修复常见错误后解析
        fixed = brace_match.group(0) if brace_match else text
        # 移除尾逗号（JSON标准不允许）
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        # 移除单行注释
        fixed = re.sub(r'//.*?\n', '\n', fixed)
        result = self._try_parse_json(fixed)
        if result:
            return result

        # 策略4: 正则提取各字段
        result = self._try_regex_extract(text)
        if result:
            return result

        # 策略5: 最终回退到旧格式
        logger.warning("All JSON parsing strategies failed. Using SCORE/ISSUES/SUGGESTION fallback.")
        return self._parse_old_format(text)

    def _try_parse_json(self, text: str) -> Dict:
        """
        尝试解析 JSON 并转换为标准结果格式。

        Args:
            text: 待解析的 JSON 文本

        Returns:
            标准结果 dict，或 None 如果解析失败
        """
        import json

        try:
            result = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        # 检查是否为新格式（包含 total_score 或 dimension_scores）
        if "total_score" not in result and "dimension_scores" not in result:
            return None

        # 提取总分（百分制）
        total_score = result.get("total_score", 70)
        score = max(0, min(100, total_score))

        # 提取维度评分
        dimension_scores = result.get("dimension_scores", {})

        # 提取问题
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []

        # 提取建议
        suggestion = result.get("suggestion", "approve").lower()
        if suggestion not in ("approve", "revise:data", "revise:logic", "revise:writing"):
            suggestion = "approve" if total_score >= 70 else "revise:writing"

        # 提取改进提示
        improvement_hints = result.get("improvement_hints", {})
        if not isinstance(improvement_hints, dict):
            improvement_hints = {}

        return {
            "score": score,
            "dimension_scores": dimension_scores,
            "issues": issues,
            "suggestion": suggestion,
            "improvement_hints": improvement_hints
        }

    def _try_regex_extract(self, text: str) -> Dict:
        """
        用正则从非标准 JSON 文本中提取评分字段。

        Args:
            text: 待提取的文本

        Returns:
            标准结果 dict，或 None 如果提取失败
        """
        import re

        # 提取 total_score
        ts_match = re.search(r'"total_score"\s*:\s*(\d+)', text)
        if not ts_match:
            return None

        total_score = int(ts_match.group(1))
        score = max(0, min(100, total_score))

        # 提取各维度评分
        dim_keys = ["topic_relevance", "analysis_depth", "writing_quality",
                     "citation_sufficiency", "groundedness", "context_coherence"]
        dimension_scores = {}
        for key in dim_keys:
            m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
            if m:
                dimension_scores[key] = int(m.group(1))

        # 提取 issues
        issues = []
        issues_match = re.search(r'"issues"\s*:\s*\[([^\]]*)\]', text)
        if issues_match:
            issues = [s.strip().strip('"').strip("'") for s in issues_match.group(1).split(',') if s.strip().strip('"')]

        # 提取 suggestion
        sug_match = re.search(r'"suggestion"\s*:\s*"([^"]+)"', text)
        suggestion = sug_match.group(1).lower() if sug_match else ("approve" if score >= 7 else "revise:writing")

        # 提取 improvement_hints
        improvement_hints = {}
        for role in ["researcher", "analyst", "writer"]:
            hint_match = re.search(rf'"{role}"\s*:\s*"([^"]+)"', text)
            if hint_match:
                improvement_hints[role] = hint_match.group(1)

        logger.info(f"Regex extraction succeeded: score={score}, dims={len(dimension_scores)}, hints={len(improvement_hints)}")

        return {
            "score": score,
            "dimension_scores": dimension_scores,
            "issues": issues,
            "suggestion": suggestion,
            "improvement_hints": improvement_hints
        }

    def _parse_old_format(self, text: str) -> Dict:
        """
        解析旧格式 SCORE/ISSUES/SUGGESTION。

        Args:
            text: 待解析的文本

        Returns:
            标准结果 dict（dimension_scores 和 improvement_hints 为空）
        """
        score = 70
        issues = []
        suggestion = "approve"

        for line in text.strip().split('\n'):
            line = line.strip()
            if line.upper().startswith('SCORE:'):
                try:
                    score_text = line.split(':', 1)[1].strip()
                    if '/' in score_text:
                        parts = score_text.split('/')
                        score = int(round(float(parts[0]) / float(parts[1]) * 100))
                    else:
                        score = int(''.join(c for c in score_text if c.isdigit()) or '70')
                    score = max(0, min(100, score))
                except (ValueError, IndexError):
                    score = 70
            elif line.upper().startswith('ISSUES:'):
                issues_text = line.split(':', 1)[1].strip()
                if issues_text.upper() != 'NONE':
                    issues = [i.strip() for i in issues_text.split(';') if i.strip()]
            elif line.upper().startswith('SUGGESTION:'):
                suggestion = line.split(':', 1)[1].strip().lower()

        if suggestion not in ("approve", "revise:data", "revise:logic", "revise:writing"):
            suggestion = "approve" if score >= 70 else "revise:writing"

        return {
            "score": score,
            "dimension_scores": {},
            "issues": issues,
            "suggestion": suggestion,
            "improvement_hints": {}
        }

    def _llm_evaluate_executive_summary(self, summary: str, full_report: str) -> Dict:
        """
        评审执行摘要质量。

        Args:
            summary: 执行摘要文本
            full_report: 完整报告（用于评估覆盖率）

        Returns:
            评审结果
        """
        chinese_chars = sum(1 for c in summary if '\u4e00' <= c <= '\u9fff')

        prompt = f"""你是国企战略规划报告评审专家。请评审以下执行摘要的质量。

## 执行摘要要求
1. **精简度**: 字数控制在800-1200字（当前{chinese_chars}字）
2. **关键数据保留**: 是否包含报告中的核心数据（投资额、增长率、KPI等）
3. **核心结论覆盖**: 是否概括了报告的主要结论和建议
4. **公文语态**: 是否符合国企公文风格

## 执行摘要内容
{summary}

## 输出格式
```json
{{
    "score": <1-10>,
    "issues": ["问题1", "问题2"],
    "suggestion": "approve 或 revise",
    "improvement_hints": "具体改进建议"
}}
```

只输出JSON。
"""

        try:
            if self.reviewer_client is not None:
                response = self.reviewer_client.chat.completions.create(
                    model=self.reviewer_model,
                    messages=[
                        {"role": "system", "content": "你是执行摘要评审专家，严格按JSON格式输出。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                    extra_body={'enable_thinking': False}
                )
                response_text = response.choices[0].message.content or ""
            else:
                response_text = self.reviewer_llm.invoke(prompt, temperature=0.3, max_tokens=512)

            # 解析响应
            import json
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result = json.loads(response_text.strip())
            return {
                "score": result.get("score", 7),
                "issues": result.get("issues", []),
                "suggestion": result.get("suggestion", "approve"),
                "improvement_hints": result.get("improvement_hints", "")
            }

        except Exception as e:
            logger.warning(f"Executive summary review failed: {e}")
            return {"score": 7, "issues": [], "suggestion": "approve", "improvement_hints": ""}

    def _review_executive_summary(self, report: str, state: Dict[str, Any]) -> str:
        """
        提取并质检执行摘要，不合格时重新生成（max 1 retry）。

        Args:
            report: 完整报告文本
            state: 最终状态

        Returns:
            可能更新后的报告文本
        """
        import re

        # 提取执行摘要部分
        summary_match = re.search(r'# 执行摘要\n+(.*?)(?=\n---\n|\n## 目录|\n# 第)', report, re.DOTALL)
        if not summary_match:
            logger.info("No executive summary found, skipping review")
            return report

        summary = summary_match.group(1).strip()
        if not summary or len(summary) < 100:
            logger.info("Executive summary too short, skipping review")
            return report

        print(f"\n[REVIEW] Reviewing executive summary ({sum(1 for c in summary if '\\u4e00' <= c <= '\\u9fff')} chars)...")

        review_result = self._llm_evaluate_executive_summary(summary, report)

        print(f"  Score: {review_result['score']}/10 | Suggestion: {review_result['suggestion']}")
        if review_result.get("issues"):
            print(f"  Issues: {'; '.join(review_result['issues'][:3])}")

        if review_result.get("suggestion") == "approve" or review_result.get("score", 7) >= 7:
            print(f"  [OK] Executive summary passed review")
            return report

        # 不合格时，用改进提示重新生成摘要
        logger.info(f"Executive summary needs revision: {review_result.get('improvement_hints', '')}")

        try:
            from rag_project.agent.llm_manager import LLMManager
            archiver_llm = LLMManager("archiver")

            revision_hint = review_result.get("improvement_hints", "")
            revised_summary = archiver_llm.invoke(
                prompt=f"""请根据评审反馈重写以下执行摘要。

评审反馈:
{'; '.join(review_result.get('issues', []))}
{revision_hint}

原始执行摘要:
{summary}

要求:
1. 精简浓缩，控制在800-1200字
2. 保留所有关键数据和核心结论
3. 使用国企公文语态
4. 只输出执行摘要正文，不要添加标题或其他标记

重写后的执行摘要:""",
                temperature=0.5
            )

            # 替换报告中的执行摘要
            new_report = report.replace(summary, revised_summary.strip())
            print(f"  [REVISED] Executive summary regenerated ({sum(1 for c in revised_summary if '\\u4e00' <= c <= '\\u9fff')} chars)")
            return new_report

        except Exception as e:
            logger.warning(f"Executive summary revision failed: {e}. Keeping original.")
            return report

    def save_report(self, report: str, output_path: str) -> None:
        """
        Save the final report to file.

        Args:
            report: The final report content
            output_path: Path to save the report

        Raises:
            IOError: If file write fails
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise
