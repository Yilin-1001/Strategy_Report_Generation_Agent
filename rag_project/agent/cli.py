"""
CLI interface for the report generation agent system.

Provides interactive and automated modes for running the multi-agent workflow.
"""

import os
from typing import Optional, Dict, Any
from langgraph.errors import GraphInterrupt

from rag_project.agent.graph import create_report_graph
from rag_project.agent.state import GraphState
from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


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

        Initializes the report graph and default configuration.
        """
        self.app = create_report_graph()
        self.config = {"configurable": {"thread_id": "report_session_001"}}

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
        print(f"📋 Report Generation Started")
        print(f"{'='*60}")
        print(f"Request: {user_input}")
        print(f"Mode: {'Auto' if auto_mode else 'Interactive'}")
        print(f"{'='*60}\n")

        try:
            # Start the workflow
            print("🚀 Starting workflow...\n")

            # Stream the workflow execution
            for event in self.app.stream(
                {"user_input": user_input},
                self.config,
                stream_mode="values"
            ):
                # Handle each node completion with progress indicators
                node_name = event.get("__start__", "")
                if "current_chapter_index" in event:
                    chapter_idx = event.get("current_chapter_index", 0)
                    print(f"📋 Processing chapter {chapter_idx + 1}...")

                if "chapter_title" in event and event["chapter_title"]:
                    print(f"🔍 Researching: {event['chapter_title']}")

                if "current_draft" in event and event["current_draft"]:
                    chapter = event.get("chapter_title", "Unknown")
                    print(f"✅ Completed: {chapter}")

                # Check if we're at human review point
                if "current_draft" in event and event["current_draft"]:
                    # Get user feedback
                    feedback = self._get_user_feedback(event, auto_mode)

                    # Update state with feedback and continue
                    if feedback:
                        print(f"📝 Feedback received, continuing...")
                        # The graph will continue with the feedback
                    else:
                        print("⏭️  Skipping feedback...")

            # Get the final state
            final_state = self.app.get_state(self.config)
            final_report = final_state.values.get("final_report", "")

            if not final_report:
                raise ValueError("No final report generated")

            print(f"\n{'='*60}")
            print(f"✅ Report Generation Complete!")
            print(f"{'='*60}\n")

            return final_report

        except GraphInterrupt:
            # Handle interrupt for human review
            logger.info("Workflow interrupted for human review")

            # Get current state
            state = self.app.get_state(self.config)

            # Get feedback
            feedback = self._get_user_feedback(state.values, auto_mode)

            # Continue with feedback
            if feedback:
                print("📝 Submitting feedback and continuing...")
                for event in self.app.stream(
                    None,  # None to continue from interrupt
                    self.config,
                    stream_mode="values"
                ):
                    if "current_draft" in event and event["current_draft"]:
                        chapter = event.get("chapter_title", "Unknown")
                        print(f"✅ Completed: {chapter}")

            # Get final state
            final_state = self.app.get_state(self.config)
            final_report = final_state.values.get("final_report", "")

            print(f"\n{'='*60}")
            print(f"✅ Report Generation Complete!")
            print(f"{'='*60}\n")

            return final_report

        except Exception as e:
            logger.error(f"Error during report generation: {e}")
            print(f"\n❌ Error: {e}")
            raise

    def _get_user_feedback(self, state: Dict[str, Any], auto_mode: bool = False) -> Dict:
        """
        Get user feedback on the current draft.

        In auto mode, returns default approval.
        In interactive mode, prompts user for input.

        Args:
            state: Current workflow state
            auto_mode: If True, automatically approve

        Returns:
            Dict with 'decision', 'feedback_type', and 'comments' keys
        """
        chapter = state.get("chapter_title", "Unknown")
        draft = state.get("current_draft", "")

        print(f"\n{'─'*60}")
        print(f"📖 Chapter: {chapter}")
        print(f"{'─'*60}")
        print(f"{draft}\n")
        print(f"{'─'*60}\n")

        if auto_mode:
            print("⏩ Auto mode: Approving chapter...\n")
            return {
                "decision": "approve",
                "feedback_type": "auto",
                "comments": "Auto-approved"
            }

        # Interactive mode
        print("Please review this chapter:")
        print("  1. Approve - Continue to next chapter")
        print("  2. Revise - Request revision with specific comments")
        print("  3. Skip - Skip this chapter\n")

        while True:
            try:
                choice = input("Your choice (1/2/3): ").strip()

                if choice == "1":
                    comments = input("Optional comments (press Enter to skip): ").strip()
                    print()
                    return {
                        "decision": "approve",
                        "feedback_type": "user",
                        "comments": comments or "Approved"
                    }

                elif choice == "2":
                    comments = input("Revision instructions: ").strip()
                    if not comments:
                        print("❌ Revision instructions required. Please try again.\n")
                        continue
                    print()
                    return {
                        "decision": "revise",
                        "feedback_type": "user",
                        "comments": comments
                    }

                elif choice == "3":
                    print()
                    return {
                        "decision": "skip",
                        "feedback_type": "user",
                        "comments": "Skipped by user"
                    }

                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.\n")

            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Interrupt detected. Approving by default...\n")
                return {
                    "decision": "approve",
                    "feedback_type": "interrupt",
                    "comments": "Auto-approved due to interrupt"
                }

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
