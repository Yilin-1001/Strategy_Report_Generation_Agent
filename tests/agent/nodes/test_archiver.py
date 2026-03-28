"""Tests for archiver node."""

import pytest
from datetime import datetime
from rag_project.agent.nodes.archiver import archiver_node


def test_archiver_node():
    """Test that archiver node merges all chapters with cover and footer."""
    # Setup test data
    state = {
        "context_pool": [
            "# 第一章\n\n第一章内容",
            "# 第二章\n\n第二章内容",
            "# 第三章\n\n第三章内容"
        ],
        "user_input": "测试主题"
    }

    # Execute archiver node
    result = archiver_node(state)

    # Verify structure
    assert "final_report" in result
    final_report = result["final_report"]

    # Verify cover is present
    assert "江西交通投资集团战略规划报告" in final_report
    assert "**生成时间**:" in final_report
    assert "**主题**: 测试主题" in final_report

    # Verify all chapters are merged with separator
    assert "# 第一章\n\n第一章内容" in final_report
    assert "# 第二章\n\n第二章内容" in final_report
    assert "# 第三章\n\n第三章内容" in final_report
    assert "\n\n---\n\n" in final_report

    # Verify footer is present
    assert "## 目录" in final_report or "## 章节" in final_report or "章节列表" in final_report
    assert "第一章" in final_report
    assert "第二章" in final_report
    assert "第三章" in final_report

    # Verify structure order: cover + chapters + footer
    cover_end = final_report.find("---\n\n") + len("---\n\n")
    assert cover_end > 0

    # Verify datetime format
    assert datetime.now().strftime('%Y年%m月%d日') in final_report
