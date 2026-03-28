"""
Tests for LLM Manager with DeepSeek API integration
"""

import os
import pytest
from rag_project.agent.llm_manager import LLMManager


class TestLLMManager:
    """Test suite for LLMManager class"""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or skip test"""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set in environment")
        return api_key

    def test_coordinator_initialization(self, api_key):
        """Test LLM manager initialization with coordinator agent"""
        manager = LLMManager(agent_type="coordinator")
        assert manager is not None
        assert manager.agent_type == "coordinator"
        assert manager.temperature == 0.3
        assert "Coordinator Agent" in manager.system_prompt

    def test_researcher_initialization(self, api_key):
        """Test LLM manager initialization with researcher agent"""
        manager = LLMManager(agent_type="researcher")
        assert manager is not None
        assert manager.agent_type == "researcher"
        assert manager.temperature == 0.1
        assert "Researcher Agent" in manager.system_prompt

    def test_analyst_initialization(self, api_key):
        """Test LLM manager initialization with analyst agent"""
        manager = LLMManager(agent_type="analyst")
        assert manager is not None
        assert manager.agent_type == "analyst"
        assert manager.temperature == 0.5
        assert "Analyst Agent" in manager.system_prompt

    def test_writer_initialization(self, api_key):
        """Test LLM manager initialization with writer agent"""
        manager = LLMManager(agent_type="writer")
        assert manager is not None
        assert manager.agent_type == "writer"
        assert manager.temperature == 0.7
        assert "Writer Agent" in manager.system_prompt

    def test_invalid_agent_type(self, api_key):
        """Test that invalid agent type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown agent type"):
            LLMManager(agent_type="invalid_agent")

    def test_invoke_coordinator(self, api_key):
        """Test invoking coordinator agent with simple query"""
        manager = LLMManager(agent_type="coordinator")
        response = manager.invoke("What is the capital of France?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_invoke_researcher(self, api_key):
        """Test invoking researcher agent with research query"""
        manager = LLMManager(agent_type="researcher")
        response = manager.invoke("Find information about climate change")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_invoke_with_custom_parameters(self, api_key):
        """Test invoking with custom parameters"""
        manager = LLMManager(agent_type="analyst")
        response = manager.invoke(
            "Analyze this data",
            temperature=0.3,
            max_tokens=500
        )
        assert response is not None
        assert isinstance(response, str)

    def test_system_prompt_format(self, api_key):
        """Test that system prompts are properly formatted"""
        manager = LLMManager(agent_type="writer")
        assert isinstance(manager.system_prompt, str)
        assert len(manager.system_prompt) > 50
        assert "Writer Agent" in manager.system_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
