from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Single reranked result"""
    text: str
    score: float
    original_index: int
    metadata: Dict


class BaseReranker(ABC):
    """Abstract base class for rerankers"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank a list of retrieved documents.

        Args:
            query: The search query
            documents: List of dicts with 'text', 'score', 'metadata' keys
                       (same format as MilvusManager.search() output)
            top_k: Number of results to return after reranking

        Returns:
            List of dicts with same format: {'text', 'score', 'metadata'}
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model info for logging"""
        pass
