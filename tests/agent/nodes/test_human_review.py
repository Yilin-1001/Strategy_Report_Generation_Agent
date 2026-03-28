"""
Tests for human_review node and should_continue routing function.
"""

import pytest
from rag_project.agent.nodes.human_review import human_review_node, should_continue


class TestShouldContinue:
    """Tests for should_continue routing function."""

    def test_should_continue_approve_with_more_chapters(self):
        """Test approve with more chapters remaining returns 'continue'."""
        state = {
            "chapter_index": 0,
            "chapter_titles": ["Chapter 1", "Chapter 2", "Chapter 3"],
            "review_decision": "approve"
        }
        result = should_continue(state)
        assert result == "continue"

    def test_should_continue_approve_last_chapter(self):
        """Test approve with last chapter returns 'end'."""
        state = {
            "chapter_index": 2,
            "chapter_titles": ["Chapter 1", "Chapter 2", "Chapter 3"],
            "review_decision": "approve"
        }
        result = should_continue(state)
        assert result == "end"

    def test_should_continue_revise_data(self):
        """Test revise:data returns 'researcher'."""
        state = {
            "review_decision": "revise:data"
        }
        result = should_continue(state)
        assert result == "researcher"

    def test_should_continue_revise_logic(self):
        """Test revise:logic returns 'analyst'."""
        state = {
            "review_decision": "revise:logic"
        }
        result = should_continue(state)
        assert result == "analyst"

    def test_should_continue_revise_writing(self):
        """Test revise:writing returns 'writer'."""
        state = {
            "review_decision": "revise:writing"
        }
        result = should_continue(state)
        assert result == "writer"

    def test_should_continue_finished(self):
        """Test finished decision returns 'end'."""
        state = {
            "review_decision": "finished"
        }
        result = should_continue(state)
        assert result == "end"


class TestHumanReviewNode:
    """Tests for human_review_node function."""

    def test_human_review_approve(self):
        """Test approve adds draft to context_pool and increments index."""
        state = {
            "chapter_index": 0,
            "chapter_titles": ["Chapter 1", "Chapter 2"],
            "current_draft": "This is the content of chapter 1.",
            "context_pool": [],
            "scratchpad": ["Some notes"],
            "review_decision": "approve"
        }

        result = human_review_node(state)

        # Check that draft was added to context_pool
        assert len(result["context_pool"]) == 1
        assert "# Chapter 1" in result["context_pool"][0]
        assert "This is the content of chapter 1." in result["context_pool"][0]

        # Check that scratchpad was cleared
        assert result["scratchpad"] == []

        # Check that chapter_index was incremented
        assert result["chapter_index"] == 1

    def test_human_review_approve_last_chapter(self):
        """Test approve on last chapter still increments index."""
        state = {
            "chapter_index": 1,
            "chapter_titles": ["Chapter 1", "Chapter 2"],
            "current_draft": "This is the content of chapter 2.",
            "context_pool": ["Previous chapter content"],
            "scratchpad": ["Some notes"],
            "review_decision": "approve"
        }

        result = human_review_node(state)

        # Check that draft was added to context_pool
        assert len(result["context_pool"]) == 2
        assert "# Chapter 2" in result["context_pool"][1]

        # Check that scratchpad was cleared
        assert result["scratchpad"] == []

        # Check that chapter_index was incremented
        assert result["chapter_index"] == 2

    def test_human_review_revise_data(self):
        """Test revise:data keeps state and doesn't increment index."""
        state = {
            "chapter_index": 0,
            "chapter_titles": ["Chapter 1", "Chapter 2"],
            "current_draft": "Draft content",
            "context_pool": ["Previous content"],
            "scratchpad": ["Notes"],
            "review_decision": "revise:data"
        }

        result = human_review_node(state)

        # Check that state is preserved
        assert result["chapter_index"] == 0
        assert result["current_draft"] == "Draft content"
        assert result["context_pool"] == ["Previous content"]
        assert result["scratchpad"] == ["Notes"]

    def test_human_review_revise_logic(self):
        """Test revise:logic keeps state and doesn't increment index."""
        state = {
            "chapter_index": 1,
            "chapter_titles": ["Chapter 1", "Chapter 2"],
            "current_draft": "Draft content",
            "context_pool": [],
            "scratchpad": [],
            "review_decision": "revise:logic"
        }

        result = human_review_node(state)

        # Check that state is preserved
        assert result["chapter_index"] == 1
        assert result["current_draft"] == "Draft content"

    def test_human_review_revise_writing(self):
        """Test revise:writing keeps state and doesn't increment index."""
        state = {
            "chapter_index": 2,
            "chapter_titles": ["Chapter 1", "Chapter 2", "Chapter 3"],
            "current_draft": "Draft content",
            "context_pool": ["Chapter 1", "Chapter 2"],
            "scratchpad": ["Notes"],
            "review_decision": "revise:writing"
        }

        result = human_review_node(state)

        # Check that state is preserved
        assert result["chapter_index"] == 2
        assert result["context_pool"] == ["Chapter 1", "Chapter 2"]
        assert result["scratchpad"] == ["Notes"]

    def test_human_review_builds_full_chapter(self):
        """Test that full chapter includes title and draft."""
        state = {
            "chapter_index": 0,
            "chapter_titles": ["Introduction to AI"],
            "current_draft": "Artificial intelligence is...",
            "context_pool": [],
            "scratchpad": [],
            "review_decision": "approve"
        }

        result = human_review_node(state)

        # Check full chapter format
        expected_chapter = "# Introduction to AI\n\nArtificial intelligence is..."
        assert result["context_pool"][0] == expected_chapter
