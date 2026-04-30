"""
Shared LLM Client

Simple OpenAI-compatible client for DeepSeek API.
Used by all ablation groups for text generation.
"""

import os
import time
import json
from typing import Optional
from openai import OpenAI

from ablation_experiment.config import GEN_API_KEY, GEN_BASE_URL, GEN_MODEL, GEN_TIMEOUT


class LLMClient:
    """Simple LLM client wrapper for DeepSeek API with token tracking."""

    def __init__(
        self,
        model: str = GEN_MODEL,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        system_prompt: str = "You are a helpful assistant."
    ):
        if not GEN_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=GEN_API_KEY,
            base_url=GEN_BASE_URL,
            timeout=GEN_TIMEOUT
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def get_token_stats(self) -> dict:
        """Return token usage statistics."""
        return {
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
        max_retries: int = 3
    ) -> str:
        """
        Invoke LLM with prompt.

        Args:
            prompt: User prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            max_retries: Max retry attempts

        Returns:
            LLM response text
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tok
                )

                # Track token usage
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens or 0
                    self.total_completion_tokens += response.usage.completion_tokens or 0
                    self.total_tokens += response.usage.total_tokens or 0
                    self.call_count += 1

                return response.choices[0].message.content

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  [WARN] LLM API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


def parse_json_response(response: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dict

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    response = response.strip()

    # Remove markdown code blocks
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    return json.loads(response)


def parse_queries_response(response: str) -> list:
    """
    Parse LLM query generation response into list of queries.
    Source: researcher.py:185-211

    Args:
        response: Raw LLM response

    Returns:
        List of query strings
    """
    lines = response.strip().split("\n")
    queries = []

    for line in lines:
        line = line.strip()
        line = line.lstrip("0123456789.-*•°")
        line = line.strip()

        if len(line) < 3:
            continue

        queries.append(line)

    return queries
