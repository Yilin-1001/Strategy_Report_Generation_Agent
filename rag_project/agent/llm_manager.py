"""
LLM Manager for handling DeepSeek API integration with agent-specific configurations
"""

import os
from typing import Optional
from openai import OpenAI

from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


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
            config = load_config("agent_config.yaml")
            llm_config = config.get("llm", {})
            self.model = llm_config.get("model", "deepseek-chat")
            self.base_url = llm_config.get("base_url", "https://api.deepseek.com")
            self.timeout = llm_config.get("timeout", 30)
            api_key_env = llm_config.get("api_key_env", "DEEPSEEK_API_KEY")
        except Exception as e:
            logger.warning(f"Failed to load agent config: {e}. Using defaults.")
            self.model = "deepseek-chat"
            self.base_url = "https://api.deepseek.com"
            self.timeout = 30
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

        logger.info(f"Initialized LLM Manager for {agent_type} agent "
                   f"(model={self.model}, temp={self.temperature})")

    def invoke(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Invoke the LLM with the given prompt.

        Args:
            prompt: User prompt/question
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: LLM response

        Raises:
            Exception: If API call fails
        """
        try:
            # Use provided parameters or defaults from agent config
            temp = temperature if temperature is not None else self.temperature
            max_tok = max_tokens if max_tokens is not None else self.max_tokens

            logger.debug(f"Invoking {self.agent_type} agent with prompt: {prompt[:100]}...")

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
            logger.debug(f"Received response: {result[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise
