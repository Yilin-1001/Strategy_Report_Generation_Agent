"""Archiver node for final report generation."""

from datetime import datetime


def archiver_node(context_pool: dict, user_input: str) -> dict:
    """
    Merge all chapters into the final report with cover and footer.

    Args:
        context_pool: Dictionary containing:
            - chapters: List of chapter contents to merge
        user_input: User's query/topic for the report

    Returns:
        Dictionary containing:
            - final_report: Complete report with cover, chapters, and footer
    """
    chapters = context_pool.get("chapters", [])

    # Create cover
    cover = f"""# 江西交通投资集团战略规划报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日')}

**主题**: {user_input}

---

"""

    # Merge chapters with separator
    final_report_content = "\n\n---\n\n".join(chapters)

    # Create footer with chapter list
    footer = "\n\n---\n\n## 章节\n\n"
    for i, chapter in enumerate(chapters, 1):
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

    return {"final_report": full_report}
