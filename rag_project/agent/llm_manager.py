"""
LLM Manager for handling DeepSeek API integration with agent-specific configurations
"""

import os
import time
from typing import Optional
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError

from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global token registry: tracks all LLMManager instances by agent_type
_global_token_registry = {}


def get_global_token_stats() -> dict:
    """Return aggregated token stats from all LLMManager instances."""
    return dict(_global_token_registry)


def reset_global_token_stats():
    """Reset the global token registry."""
    _global_token_registry.clear()


class LLMManager:
    """
    Manages LLM interactions with DeepSeek API using OpenAI SDK.

    Supports different agent types with specific configurations:
    - coordinator: temperature=0.3, strategic and focused
    - researcher: temperature=0.1, precise and thorough
    - analyst: temperature=0.5, balanced and analytical
    - writer: temperature=0.7, creative and clear
    """

    # Agent-specific configurations
    AGENT_CONFIGS = {
        "coordinator": {
            "temperature": 0.3,
            "max_tokens": 2048,
            "system_prompt": """You are the Coordinator Agent for a RAG (Retrieval-Augmented Generation) system.
Your role is to:
- Analyze user queries and determine the appropriate research strategy
- Coordinate between different specialized agents
- Synthesize information from multiple sources
- Ensure responses are comprehensive and well-structured

Be concise, strategic, and focused on effective task delegation."""
        },
        "researcher": {
            "temperature": 0.1,
            "max_tokens": 4096,
            "system_prompt": """You are the Researcher Agent for a RAG system.
Your role is to:
- Conduct thorough information retrieval from the knowledge base
- Find relevant documents and evidence
- Extract key facts and details
- Verify information accuracy

Be precise, thorough, and focus on finding the most relevant information."""
        },
        "analyst": {
            "temperature": 0.5,
            "max_tokens": 3072,
            "system_prompt": """You are the Analyst Agent for a RAG system.
Your role is to:
- Analyze retrieved information critically
- Identify patterns and relationships
- Compare and contrast different sources
- Provide insights and conclusions

Be analytical, balanced, and focus on deeper understanding."""
        },
        "writer": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "system_prompt": """You are the Writer Agent for a RAG system.
Your role is to:
- Synthesize information into clear, coherent responses
- Structure content logically
- Ensure readability and engagement
- Adapt tone to the context

Be creative, clear, and focus on effective communication."""
        },
        "strategist": {
            "temperature": 0.5,
            "max_tokens": 4096,
            "system_prompt": """You are the Strategist Agent for strategic planning and blueprint generation.
Your role is to:
- Analyze SWOT analysis from diagnostic chapters
- Apply TOWS matrix analysis to generate strategic options
- Formulate mission statements and strategic pillars
- Define SMART KPIs across balanced scorecard dimensions

Be strategic, analytical, and focus on creating actionable strategic blueprints."""
        },
        "archiver": {
            "temperature": 0.5,
            "max_tokens": 3072,
            "system_prompt": """You are the Archiver Agent for report compilation and summarization.
Your role is to:
- Synthesize completed chapters into a coherent final report
- Generate executive summaries that highlight key insights
- Ensure proper formatting and structure
- Compile appendices and reference materials

Be systematic, clear, and focus on producing polished final deliverables."""
        }
    }

    def __init__(self, agent_type: str = "coordinator"):
        """
        Initialize LLM Manager with agent-specific configuration.

        Args:
            agent_type: Type of agent (coordinator, researcher, analyst, writer)

        Raises:
            ValueError: If agent_type is not recognized
        """
        if agent_type not in self.AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}. "
                           f"Must be one of: {list(self.AGENT_CONFIGS.keys())}")

        self.agent_type = agent_type
        agent_config = self.AGENT_CONFIGS[agent_type]

        self.temperature = agent_config["temperature"]
        self.max_tokens = agent_config["max_tokens"]
        self.system_prompt = agent_config["system_prompt"]

        # Load LLM configuration
        try:
            config = load_config("config/agent_config.yaml")
            llm_config = config.get("llm", {})
            self.model = llm_config.get("model", "deepseek-chat")
            self.base_url = llm_config.get("base_url", "https://api.deepseek.com")
            self.timeout = llm_config.get("timeout", 180)
            api_key_env = llm_config.get("api_key_env", "DEEPSEEK_API_KEY")
        except Exception as e:
            logger.warning(f"Failed to load agent config: {e}. Using defaults.")
            self.model = "deepseek-chat"
            self.base_url = "https://api.deepseek.com"
            self.timeout = 180
            api_key_env = "DEEPSEEK_API_KEY"

        # Get API key from environment
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Please set {api_key_env} environment variable."
            )

        # Initialize OpenAI client for DeepSeek API
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

        logger.info(f"Initialized LLM Manager for {agent_type} agent "
                   f"(model={self.model}, temp={self.temperature})")

    def get_token_stats(self) -> dict:
        """Return token usage statistics."""
        return {
            "agent_type": self.agent_type,
            "call_count": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset_token_stats(self):
        """Reset token counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def invoke(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Invoke the LLM with the given prompt, with retry on transient errors.

        Args:
            prompt: User prompt/question
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            max_retries: Maximum number of retry attempts (default 3)
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: LLM response

        Raises:
            Exception: If API call fails after all retries
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(max_retries):
            try:
                logger.debug(f"Invoking {self.agent_type} agent (attempt {attempt+1}/{max_retries}) with prompt: {prompt[:100]}...")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok,
                    **kwargs
                )

                result = response.choices[0].message.content

                # Track token usage
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens or 0
                    self.total_completion_tokens += response.usage.completion_tokens or 0
                    self.total_tokens += response.usage.total_tokens or 0
                    self.call_count += 1
                    # Update global registry
                    _global_token_registry[self.agent_type] = {
                        "agent_type": self.agent_type,
                        "call_count": self.call_count,
                        "prompt_tokens": self.total_prompt_tokens,
                        "completion_tokens": self.total_completion_tokens,
                        "total_tokens": self.total_tokens,
                    }

                logger.debug(f"Received response: {result[:100]}...")

                return result

            except (APITimeoutError, RateLimitError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"LLM API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM API failed after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error invoking LLM: {e}")
                raise
