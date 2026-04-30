import requests
from typing import List, Dict, Optional
from rag_project.reranker.reranker_base import BaseReranker
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger


class SiliconFlowReranker(BaseReranker):
    """Reranker using SiliconFlow's /rerank API"""

    def __init__(self, config_path: str = "config/reranker_config.yaml"):
        self.config = load_config(config_path).get('reranker', {})

        self.model = self.config.get('model', 'BAAI/bge-reranker-v2-m3')
        self.api_base_url = self.config.get('api_base_url', 'https://api.siliconflow.cn/v1')
        self.api_key = self.config.get('api_key')
        self.timeout = self.config.get('api_timeout', 30)
        self.max_chunks_per_doc = self.config.get('max_chunks_per_doc', 1)
        self.overlap_tokens = self.config.get('overlap_tokens', 40)

        if not self.api_key:
            import os
            self.api_key = os.environ.get('SILICONFLOW_API_KEY')
            if not self.api_key:
                raise ValueError(
                    "SiliconFlow API key required. Set in reranker_config.yaml "
                    "or SILICONFLOW_API_KEY env var."
                )

        logger.info(f"SiliconFlowReranker initialized: model={self.model}")

    @classmethod
    def from_config_dict(cls, config: Dict) -> "SiliconFlowReranker":
        """Create a reranker from a config dict directly (no YAML file needed)."""
        instance = cls.__new__(cls)
        reranker_conf = config.get('reranker', {})
        instance.model = reranker_conf.get('model', 'BAAI/bge-reranker-v2-m3')
        instance.api_base_url = reranker_conf.get('api_base_url', 'https://api.siliconflow.cn/v1')
        instance.api_key = reranker_conf.get('api_key')
        instance.timeout = reranker_conf.get('api_timeout', 30)
        instance.max_chunks_per_doc = reranker_conf.get('max_chunks_per_doc', 1)
        instance.overlap_tokens = reranker_conf.get('overlap_tokens', 40)

        if not instance.api_key:
            import os
            instance.api_key = os.environ.get('SILICONFLOW_API_KEY')
            if not instance.api_key:
                raise ValueError("SiliconFlow API key required.")

        logger.info(f"SiliconFlowReranker created: model={instance.model}")
        return instance

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents via SiliconFlow API.

        Args:
            query: Search query string
            documents: Milvus search results [{'text', 'score', 'metadata'}]
            top_k: Number of results to return

        Returns:
            Reranked results in same format, with new 'original_score' and 'original_rank' fields
        """
        if not documents:
            return []

        doc_texts = [doc['text'] for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_k, len(documents)),
            "return_documents": True,
        }

        # BGE-specific params
        if 'bge' in self.model.lower():
            payload["max_chunks_per_doc"] = self.max_chunks_per_doc
            payload["overlap_tokens"] = self.overlap_tokens

        try:
            response = requests.post(
                f"{self.api_base_url}/rerank",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Rerank API error: {e}")
            logger.warning("Falling back to original order")
            return documents[:top_k]

        # Build reranked results
        reranked = []
        for item in data.get('results', []):
            idx = item['index']
            original_doc = documents[idx]
            reranked.append({
                'text': item.get('document', {}).get('text', original_doc['text']),
                'score': item['relevance_score'],
                'original_score': original_doc['score'],
                'original_rank': idx,
                'metadata': original_doc['metadata'],
            })

        logger.info(f"Reranked {len(documents)} -> {len(reranked)} results")
        return reranked

    def get_model_info(self) -> Dict:
        return {
            'type': 'siliconflow_reranker',
            'model': self.model,
            'api_base_url': self.api_base_url,
        }
