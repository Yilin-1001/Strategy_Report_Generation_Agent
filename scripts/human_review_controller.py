#!/usr/bin/env python3
"""
Human Review Test Controller

Step-by-step interactive review controller that communicates via files:
- output/review/draft.md     -> Current chapter draft for review
- output/review/feedback.json -> Expert feedback file (created by reviewer)
- output/review/status.json   -> Current workflow status
- output/review/log.txt       -> Processing log

Usage:
    python scripts/human_review_controller.py "report request"

Workflow:
    1. Script starts and generates Chapter 1
    2. Writes draft to output/review/draft.md
    3. Waits for output/review/feedback.json to appear
    4. Reads feedback, applies it, continues to next chapter
    5. Repeats until all chapters are done
"""

import json
import sys
import time
import os
import signal
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_project.agent.graph import create_report_graph
from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)

# Review communication directory
REVIEW_DIR = project_root / "output" / "review"
DRAFT_FILE = REVIEW_DIR / "draft.md"
FEEDBACK_FILE = REVIEW_DIR / "feedback.json"
STATUS_FILE = REVIEW_DIR / "status.json"
LOG_FILE = REVIEW_DIR / "log.txt"
REPORT_FILE = project_root / "output" / "human_review_report.md"


def log(msg: str):
    """Write to both console and log file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        # Windows GBK fallback - strip non-ASCII chars for console
        safe_line = line.encode('gbk', errors='replace').decode('gbk')
        print(safe_line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(status: dict):
    """Write current status to status file."""
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def write_draft(draft: str, chapter_title: str, chapter_index: int, phase: str):
    """Write draft to file for expert review."""
    header = f"""# 专家审核: {chapter_title}
> 章节索引: {chapter_index} | 阶段: {phase}
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 审核状态: 等待专家反馈

---

"""
    with open(DRAFT_FILE, "w", encoding="utf-8") as f:
        f.write(header + draft)

    write_status({
        "state": "waiting_for_review",
        "chapter_index": chapter_index,
        "chapter_title": chapter_title,
        "phase": phase,
        "timestamp": datetime.now().isoformat()
    })


def write_blueprint_draft(blueprint):
    """Write strategic blueprint for review - handles various formats."""
    import json as _json
    content = f"""# 专家审核: 战略蓝图 (Strategic Blueprint)
> 审核类型: 战略蓝图审批
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 审核状态: 等待专家反馈

---

"""
    if isinstance(blueprint, dict):
        # Try structured format first
        content += f"## 战略使命 (Mission)\n"
        content += f"{blueprint.get('mission', 'N/A')}\n\n"

        pillars = blueprint.get('strategic_pillars', [])
        if pillars:
            content += f"## 战略支柱 (Strategic Pillars)\n"
            for i, pillar in enumerate(pillars, 1):
                if isinstance(pillar, dict):
                    content += f"\n### 支柱 {i}: {pillar.get('name', 'N/A')}\n"
                    content += f"{pillar.get('description', 'N/A')}\n"
                else:
                    content += f"\n### 支柱 {i}: {pillar}\n"

        kpis = blueprint.get('kpis', {})
        if kpis:
            content += f"\n## 关键绩效指标 (KPIs)\n"
            if isinstance(kpis, dict):
                for dimension, indicators in kpis.items():
                    content += f"\n### {dimension}\n"
                    if isinstance(indicators, list):
                        for ind in indicators:
                            content += f"- {ind}\n"
                    else:
                        content += f"{indicators}\n"
            else:
                content += f"{kpis}\n"

        # Also show raw blueprint for reference
        content += f"\n\n---\n## 完整蓝图数据\n```json\n{_json.dumps(blueprint, ensure_ascii=False, indent=2)[:3000]}\n```\n"
    else:
        # Fallback: just dump the raw data
        content += f"## 原始蓝图数据\n```\n{str(blueprint)[:5000]}\n```\n"

    with open(DRAFT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    write_status({
        "state": "waiting_for_blueprint_review",
        "review_type": "strategic_blueprint",
        "timestamp": datetime.now().isoformat()
    })


def wait_for_feedback(timeout: int = 1800) -> dict:
    """Wait for expert feedback file to appear (default 30 min timeout)."""
    start = time.time()
    log("[WAIT] 等待专家审核反馈... (请编辑 output/review/feedback.json)")
    while time.time() - start < timeout:
        if FEEDBACK_FILE.exists():
            time.sleep(0.5)  # Brief pause to ensure file is fully written
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback = json.load(f)
            FEEDBACK_FILE.unlink()  # Clean up
            log(f"[OK] 收到专家反馈: {feedback.get('decision', 'unknown')}")
            return feedback
        time.sleep(2)
    raise TimeoutError("Timed out waiting for expert feedback")


def run_interactive_review(user_input: str):
    """Run the full interactive review process."""
    # Setup review directory
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous review files
    for f in [DRAFT_FILE, FEEDBACK_FILE, STATUS_FILE]:
        if f.exists():
            f.unlink()

    log("=" * 60)
    log("人工检索审核测试 - 启动")
    log(f"报告主题: {user_input}")
    log("=" * 60)

    # Create the workflow graph
    log("正在初始化工作流...")
    app = create_report_graph()
    config = {"configurable": {"thread_id": "human_review_test_001"}}

    is_first = True
    max_iterations = 80
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        input_data = {"user_input": user_input} if is_first else None
        is_first = False

        log(f"\n--- 迭代 {iteration} ---")

        # Stream until interrupt or completion
        for event in app.stream(
            input_data,
            config,
            stream_mode="values"
        ):
            if "current_chapter_index" in event:
                idx = event.get("current_chapter_index", 0)
                title = event.get("chapter_title", "")
                if title:
                    log(f"  处理中: 章节 {idx + 1} - {title}")

        # Check state after streaming
        state = app.get_state(config)
        state_values = state.values

        # Check if at human review interrupt
        if state.next and "human_review" in state.next:
            current_draft = state_values.get("current_draft", "")
            blueprint = state_values.get("strategic_blueprint", {})
            current_index = state_values.get("current_chapter_index", 0)
            chapter_title = state_values.get("chapter_title", f"章节 {current_index + 1}")

            # Determine if this is blueprint review
            is_blueprint_review = (
                current_index == 2 and
                blueprint and
                not blueprint.get("approved") and
                not current_draft
            )

            if is_blueprint_review:
                # Blueprint review
                log("\n" + "=" * 60)
                log("[BLUEPRINT] 战略蓝图审核")
                log("=" * 60)
                write_blueprint_draft(blueprint)
                log("蓝图已写入 output/review/draft.md，等待审核...")

                # Wait for expert feedback
                feedback = wait_for_feedback()
                decision = feedback.get("decision", "approve_blueprint")

                if decision == "revise_blueprint":
                    comments = feedback.get("comments", "")
                    log(f"修订蓝图: {comments}")
                    app.update_state(
                        config,
                        {
                            "review_decision": "revise_blueprint",
                            "human_feedback": {"comments": comments}
                        }
                    )
                else:
                    comments = feedback.get("comments", "")
                    log(f"批准蓝图: {comments}")
                    app.update_state(
                        config,
                        {"review_decision": "approve_blueprint"}
                    )
                continue

            elif current_draft:
                # Chapter review
                global_plan = state_values.get("global_plan", [])
                phase = "诊断阶段" if current_index < 3 else "推演阶段"

                log("\n" + "=" * 60)
                log(f"[CHAPTER] 章节 {current_index + 1} 审核: {chapter_title}")
                log(f"   阶段: {phase}")
                log("=" * 60)

                # Write draft for review
                write_draft(current_draft, chapter_title, current_index, phase)
                log(f"章节草稿已写入 output/review/draft.md，等待审核...")

                # Wait for expert feedback
                feedback = wait_for_feedback()
                decision = feedback.get("decision", "approve")
                comments = feedback.get("comments", "")
                feedback_type = feedback.get("feedback_type", "")

                log(f"专家决策: {decision}")
                log(f"专家意见: {comments}")

                if decision == "approve":
                    app.update_state(
                        config,
                        {"review_decision": "approve"}
                    )
                elif decision == "revise":
                    # Map feedback_type to revision route
                    if feedback_type == "data":
                        app.update_state(
                            config,
                            {
                                "review_decision": "revise:data",
                                "human_feedback": {"comments": comments}
                            }
                        )
                    elif feedback_type == "logic":
                        app.update_state(
                            config,
                            {
                                "review_decision": "revise:logic",
                                "human_feedback": {"comments": comments}
                            }
                        )
                    elif feedback_type == "writing":
                        app.update_state(
                            config,
                            {
                                "review_decision": "revise:writing",
                                "human_feedback": {"comments": comments}
                            }
                        )
                    else:
                        # Default to writing revision
                        app.update_state(
                            config,
                            {
                                "review_decision": "revise:writing",
                                "human_feedback": {"comments": comments}
                            }
                        )
                elif decision == "skip":
                    app.update_state(
                        config,
                        {"review_decision": "approve"}
                    )
                elif decision == "approve_blueprint":
                    app.update_state(
                        config,
                        {"review_decision": "approve_blueprint"}
                    )
                elif decision == "revise_blueprint":
                    app.update_state(
                        config,
                        {
                            "review_decision": "revise_blueprint",
                            "human_feedback": {"comments": comments}
                        }
                    )
                continue

            else:
                # No draft and not blueprint - check for final report
                if state_values.get("final_report"):
                    break
                log("无草稿内容，继续...")
                continue

        # Check if workflow is complete
        if not state.next or len(state.next) == 0:
            break

    # Get final report
    final_state = app.get_state(config)
    final_report = final_state.values.get("final_report", "")

    if final_report:
        # Save report
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(final_report)

        log("\n" + "=" * 60)
        log("[DONE] 报告生成完成!")
        log(f"   保存路径: {REPORT_FILE}")
        log(f"   报告长度: {len(final_report)} 字符")
        log("=" * 60)

        write_status({
            "state": "completed",
            "report_path": str(REPORT_FILE),
            "report_length": len(final_report),
            "timestamp": datetime.now().isoformat()
        })
    else:
        log("[ERR] 未生成最终报告")
        write_status({
            "state": "error",
            "error": "No final report generated",
            "timestamp": datetime.now().isoformat()
        })


if __name__ == "__main__":
    request = sys.argv[1] if len(sys.argv) > 1 else "江西交通投资集团2025年战略规划报告"

    try:
        run_interactive_review(request)
    except KeyboardInterrupt:
        log("[WARN] 用户中断")
        write_status({"state": "interrupted"})
    except Exception as e:
        log(f"[ERR] 错误: {e}")
        import traceback
        traceback.print_exc()
        write_status({"state": "error", "error": str(e)})