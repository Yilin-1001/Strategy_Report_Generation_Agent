"""
Embedding API Client for external embedding services.

Supports:
- SiliconFlow API (https://api.siliconflow.cn)
- OpenAI-compatible APIs
- Custom embedding endpoints
"""

import numpy as np
import requests
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger


class EmbeddingClient:
    """
    Client for external embedding API services.

    Supports SiliconFlow, OpenAI, and other OpenAI-compatible APIs.
    """

    # API provider configurations
    API_PROVIDERS = {
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "BAAI/bge-m3",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "text-embedding-3-small",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer ",
        },
        "local": {
            "base_url": "http://localhost:8080",
            "default_model": "bge-m3",
            "auth_header": None,
        }
    }

    def __init__(
        self,
        config_path: str = "config/milvus_config.yaml",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize embedding client

        Args:
            config_path: Path to configuration file
            provider: API provider name (siliconflow, openai, local)
            api_key: API key (if required by provider)
            base_url: Override API base URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.config = load_config(config_path)
        self.embedding_config = self.config.get('embedding', {})

        # Get provider from config or parameter
        self.provider = provider or self.embedding_config.get('api_provider', 'siliconflow')

        # Get provider config
        if self.provider not in self.API_PROVIDERS:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Supported: {list(self.API_PROVIDERS.keys())}"
            )

        provider_config = self.API_PROVIDERS[self.provider]

        # Set base URL
        self.base_url = base_url or self.embedding_config.get(
            'api_base_url',
            provider_config['base_url']
        )
        self.base_url = self.base_url.rstrip('/')

        # Set model
        self.model = model or self.embedding_config.get(
            'api_model',
            provider_config['default_model']
        )

        # Set API key
        self.api_key = api_key or self.embedding_config.get('api_key')

        # Check if API key is required
        if provider_config['auth_header'] and not self.api_key:
            # Try environment variable
            import os
            env_var = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.environ.get(env_var)

            if not self.api_key:
                raise ValueError(
                    f"API key required for {self.provider}. "
                    f"Please set 'api_key' in config or {env_var} environment variable."
                )

        # Set auth header config
        self.auth_header = provider_config['auth_header']
        self.auth_prefix = provider_config.get('auth_prefix', '')

        self.timeout = timeout
        self.dimension = None

        # Test connection and get dimension
        self._test_connection()

        logger.info(f"EmbeddingClient initialized: provider={self.provider}, model={self.model}")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json"
        }

        if self.auth_header:
            headers[self.auth_header] = f"{self.auth_prefix}{self.api_key}"

        return headers

    def _test_connection(self):
        """Test connection to embedding API"""
        # For external APIs, we can't test without making a real request
        # Set default dimensions based on model
        model_dimensions = {
            "BAAI/bge-m3": 1024,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        self.dimension = model_dimensions.get(
            self.model,
            1024  # Default dimension
        )

        logger.info(f"Using model: {self.model}, dimension: {self.dimension}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        response = self._call_api({
            "input": text,
            "model": self.model
        })

        # Extract embedding
        embedding = response['data'][0]['embedding']

        return np.array(embedding, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts with automatic batching.

        Args:
            texts: List of input texts

        Returns:
            Embedding vectors as numpy array (N, D)
        """
        if not texts:
            return np.array([])

        batch_size = self.embedding_config.get('batch_size', 32)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)")

            response = self._call_api({
                "input": batch,
                "model": self.model
            })

            batch_embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Embed LangChain documents

        Args:
            documents: List of LangChain Document objects

        Returns:
            Embedding vectors as numpy array (N, D)
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    def _call_api(self, payload: Dict) -> Dict:
        """
        Call embedding API

        Args:
            payload: Request payload

        Returns:
            API response
        """
        url = f"{self.base_url}/embeddings"

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Embedding API timeout after {self.timeout}s")
            raise

        except requests.exceptions.HTTPError as e:
            logger.error(f"Embedding API HTTP error: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model

        Returns:
            Dictionary with model information
        """
        return {
            'type': 'api_client',
            'provider': self.provider,
            'base_url': self.base_url,
            'model': self.model,
            'dimension': self.dimension,
            'timeout': self.timeout,
        }
