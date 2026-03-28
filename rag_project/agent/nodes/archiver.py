"""Archiver node for final report generation."""

from datetime import datetime
from typing import Dict, Any


def archiver_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge all chapters into the final report with cover and footer.

    Args:
        state: Current workflow state containing:
            - context_pool: List of completed chapter contents
            - user_input: User's query/topic for the report

    Returns:
        Updated state dictionary containing:
            - final_report: Complete report with cover, chapters, and footer
    """
    context_pool = state.get("context_pool", [])
    user_input = state.get("user_input", "")

    # Create cover
    cover = f"""# 江西交通投资集团战略规划报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日')}

**主题**: {user_input}

---

"""

    # Merge chapters with separator
    final_report_content = "\n\n---\n\n".join(context_pool)

    # Create footer with chapter list
    footer = "\n\n---\n\n## 章节\n\n"
    for i, chapter in enumerate(context_pool, 1):
        # Extract chapter title (first line starting with #)
        lines = chapter.split('\n')
        title = f"章节 {i}"
        for line in lines:
            if line.strip().startswith('#'):
                title = line.strip().lstrip('#').strip()
                break
        footer += f"{i}. {title}\n"

    # Combine all parts
    full_report = cover + final_report_content + footer

    # Update state with final report
    updated_state = state.copy()
    updated_state["final_report"] = full_report

    return updated_state
