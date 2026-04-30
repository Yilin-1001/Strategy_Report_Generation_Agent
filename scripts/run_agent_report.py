#!/usr/bin/env python3
"""
Run script for the report generation agent system.

This script provides a command-line interface to generate reports using
the multi-agent LangGraph workflow.

Usage:
    python scripts/run_agent_report.py "your request here" [options]

Examples:
    # Interactive mode with default output
    python scripts/run_agent_report.py "Analyze China's 2024 transportation policy"

    # Auto mode with custom output
    python scripts/run_agent_report.py "Analyze China's 2024 transportation policy" -a -o custom_report.md

    # Interactive mode with custom output
    python scripts/run_agent_report.py "Generate a report on EV subsidies" --output reports/ev_subsidies.md
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_project.agent.cli import ReportGeneratorCLI
from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate reports using the multi-agent RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Analyze China's 2024 transportation policy"
  %(prog)s "Generate EV subsidy report" -a -o report.md
  %(prog)s "Create policy analysis" --auto --output reports/analysis.md
        """
    )

    parser.add_argument(
        "request",
        help="The report request or question to process"
    )

    parser.add_argument(
        "-o", "--output",
        default="output/report.md",
        help="Output file path for the generated report (default: output/report.md)"
    )

    parser.add_argument(
        "-a", "--auto",
        action="store_true",
        help="Run in automatic mode (approve all feedback without prompting)"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print("  [Agent] Multi-Agent Report Generation System")
    print("=" * 70)
    print(f"  Request: {args.request}")
    print(f"  Output:  {args.output}")
    print(f"  Mode:    {'Auto (non-interactive)' if args.auto else 'Interactive'}")
    print("=" * 70 + "\n")

    # Create output directory if needed
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {output_path.parent}")
    except Exception as e:
        print(f"[X] Error creating output directory: {e}")
        logger.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    # Initialize CLI
    try:
        cli = ReportGeneratorCLI()
    except Exception as e:
        print(f"[X] Error initializing CLI: {e}")
        logger.error(f"Failed to initialize CLI: {e}")
        sys.exit(1)

    # Generate report
    try:
        final_report = cli.generate_report(args.request, auto_mode=args.auto)

        # Save the report
        cli.save_report(final_report, args.output)

        print("\n" + "=" * 70)
        print("  [OK] Report Generation Complete!")
        print("=" * 70)
        print(f"  [File] Saved to: {args.output}")
        print(f"  [Info] Length:   {len(final_report)} characters")
        print("=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("  [WARN] Generation Interrupted by User")
        print("=" * 70 + "\n")
        logger.info("Report generation interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"  [X] Error: {e}")
        print("=" * 70 + "\n")
        logger.error(f"Error during report generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
