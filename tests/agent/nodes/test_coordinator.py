"""
Tests for Coordinator node
"""

import os
import pytest
from rag_project.agent.state import GraphState
from rag_project.agent.nodes.coordinator import coordinator_node
from rag_project.agent.llm_manager import LLMManager


class TestCoordinatorNode:
    """Test suite for coordinator_node function"""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or skip test"""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set in environment")
        return api_key

    @pytest.fixture
    def sample_state(self):
        """Create sample GraphState for testing"""
        return GraphState(
            user_input="Generate a research report on China's transportation industry in 2024",
            global_plan=[],
            current_chapter_index=0,
            context_pool=[],
            context_summary="",
            chapter_title="",
            chapter_scratchpad={},
            current_draft="",
            human_feedback={}
        )

    def test_coordinator_node_generates_plan(self, sample_state, api_key):
        """Test that coordinator node generates a valid global plan"""
        llm_manager = LLMManager(agent_type="coordinator")

        result = coordinator_node(sample_state, llm_manager)

        # Check return format
        assert isinstance(result, dict)
        assert "global_plan" in result
        assert "current_chapter_index" in result

        # Check global_plan structure
        global_plan = result["global_plan"]
        assert isinstance(global_plan, list)
        assert len(global_plan) >= 3, "Should have at least 3 chapters"
        assert len(global_plan) <= 8, "Should have at most 8 chapters"

        # Check that all chapters are strings
        for chapter in global_plan:
            assert isinstance(chapter, str)
            assert len(chapter) > 0

        # Check current_chapter_index
        assert result["current_chapter_index"] == 0

    def test_coordinator_node_content_coverage(self, sample_state, api_key):
        """Test that generated plan covers required topics"""
        llm_manager = LLMManager(agent_type="coordinator")

        result = coordinator_node(sample_state, llm_manager)
        global_plan = result["global_plan"]

        # Convert to lowercase for easier matching
        plan_text = " ".join(global_plan).lower()

        # Check for key topic coverage (at least some should be present)
        keywords = [
            "背景", "background",
            "政策", "policy",
            "现状", "status", "analysis",
            "问题", "problem", "challenge",
            "建议", "战略", "suggestion", "strategy"
        ]

        # At least 3 different keyword categories should be covered
        matched_keywords = [kw for kw in keywords if kw in plan_text]
        assert len(matched_keywords) >= 3, f"Plan should cover key topics. Found: {matched_keywords}"

    def test_coordinator_node_fallback_on_error(self, sample_state, api_key):
        """Test that coordinator node falls back to default plan on LLM error"""
        # Create a mock LLM manager that raises an exception
        class MockLLMManager:
            def invoke(self, prompt, **kwargs):
                raise Exception("Simulated LLM error")

        mock_manager = MockLLMManager()
        result = coordinator_node(sample_state, mock_manager)

        # Should still return a valid plan
        assert isinstance(result, dict)
        assert "global_plan" in result
        assert isinstance(result["global_plan"], list)
        assert len(result["global_plan"]) >= 3

    def test_coordinator_node_default_plan_structure(self, sample_state, api_key):
        """Test that default plan has proper structure"""
        class MockLLMManager:
            def invoke(self, prompt, **kwargs):
                raise Exception("Force fallback")

        mock_manager = MockLLMManager()
        result = coordinator_node(sample_state, mock_manager)
        global_plan = result["global_plan"]

        # Check default plan chapters
        assert len(global_plan) >= 5
        assert any("背景" in c or "Background" in c for c in global_plan)
        assert any("政策" in c or "Policy" in c for c in global_plan)
        assert any("现状" in c or "Status" in c or "分析" in c for c in global_plan)
        assert any("问题" in c or "Problem" in c for c in global_plan)
        assert any("建议" in c or "战略" in c or "Strategy" in c for c in global_plan)

    def test_coordinator_node_json_parsing(self, sample_state, api_key):
        """Test that coordinator node can parse JSON response from LLM"""
        # This test verifies the JSON parsing logic works correctly
        llm_manager = LLMManager(agent_type="coordinator")

        result = coordinator_node(sample_state, llm_manager)

        # Should successfully parse and return list
        assert isinstance(result["global_plan"], list)
        assert all(isinstance(chapter, str) for chapter in result["global_plan"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
