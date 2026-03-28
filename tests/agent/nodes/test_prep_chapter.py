"""
Tests for Prepare Chapter node
"""

import pytest
from rag_project.agent.state import GraphState
from rag_project.agent.nodes.prep_chapter import prepare_chapter_node


class TestPrepareChapterNode:
    """Test suite for prepare_chapter_node function"""

    @pytest.fixture
    def sample_state(self):
        """Create sample GraphState for testing"""
        return GraphState(
            user_input="Generate a research report on China's transportation industry",
            global_plan=[
                "第一章：交通运输行业背景与发展历程",
                "第二章：政策环境与监管框架",
                "第三章：发展现状与市场规模分析",
                "第四章：存在的主要问题与挑战"
            ],
            current_chapter_index=0,
            context_pool=[],
            context_summary="",
            chapter_title="",
            chapter_scratchpad={},
            current_draft="",
            human_feedback={}
        )

    def test_prepare_chapter_node_first_chapter(self, sample_state):
        """Test that prepare_chapter_node initializes first chapter correctly"""
        result = prepare_chapter_node(sample_state)

        # Check return format
        assert isinstance(result, dict)
        assert "chapter_title" in result
        assert "chapter_scratchpad" in result
        assert "current_draft" in result

        # Check chapter_title is set from global_plan[0]
        assert result["chapter_title"] == "第一章：交通运输行业背景与发展历程"

        # CRITICAL: Check chapter_scratchpad is cleared to empty dict
        assert result["chapter_scratchpad"] == {}
        assert isinstance(result["chapter_scratchpad"], dict)

        # Check current_draft is cleared to empty string
        assert result["current_draft"] == ""

    def test_prepare_next_chapter(self, sample_state):
        """Test that prepare_chapter_node works for second chapter and clears scratchpad"""
        # Simulate state after completing first chapter
        sample_state["current_chapter_index"] = 1

        # Simulate dirty scratchpad from previous chapter
        sample_state["chapter_scratchpad"] = {
            "old_notes": "Should be cleared",
            "previous_data": "This should not persist"
        }
        sample_state["current_draft"] = "Previous chapter draft that should be cleared"

        result = prepare_chapter_node(sample_state)

        # Check chapter_title is set from global_plan[1]
        assert result["chapter_title"] == "第二章：政策环境与监管框架"

        # CRITICAL: Verify scratchpad is cleared even if it had data before
        assert result["chapter_scratchpad"] == {}
        assert "old_notes" not in result["chapter_scratchpad"]
        assert "previous_data" not in result["chapter_scratchpad"]

        # Verify draft is cleared
        assert result["current_draft"] == ""

    def test_prepare_chapter_state_isolation(self, sample_state):
        """Test that prepare_chapter_node enforces state isolation between chapters"""
        # Set up state with existing data
        sample_state["current_chapter_index"] = 2
        sample_state["chapter_scratchpad"] = {
            "research": "Old research data",
            "analysis": "Old analysis data",
            "drafts": ["old draft 1", "old draft 2"]
        }
        sample_state["current_draft"] = "This is old content that must be cleared"

        result = prepare_chapter_node(sample_state)

        # Verify complete isolation - all chapter-specific data cleared
        assert result["chapter_title"] == "第三章：发展现状与市场规模分析"
        assert result["chapter_scratchpad"] == {}
        assert result["current_draft"] == ""

        # Ensure no residual data from previous chapter
        assert len(result["chapter_scratchpad"]) == 0

    def test_prepare_chapter_last_chapter(self, sample_state):
        """Test that prepare_chapter_node works for last chapter"""
        sample_state["current_chapter_index"] = 3

        result = prepare_chapter_node(sample_state)

        assert result["chapter_title"] == "第四章：存在的主要问题与挑战"
        assert result["chapter_scratchpad"] == {}
        assert result["current_draft"] == ""

    def test_prepare_chapter_empty_scratchpad_stays_empty(self, sample_state):
        """Test that prepare_chapter_node maintains empty scratchpad"""
        sample_state["chapter_scratchpad"] = {}

        result = prepare_chapter_node(sample_state)

        # Should still be empty
        assert result["chapter_scratchpad"] == {}
        assert isinstance(result["chapter_scratchpad"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
