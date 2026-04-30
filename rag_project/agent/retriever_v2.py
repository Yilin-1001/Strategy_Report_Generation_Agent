"""
Optimized RAG Retriever - Only for retrieval, not indexing

This version avoids loading unnecessary components like Chunker and ChunkStorage.
"""

from typing import List, Dict, Optional
from rag_project.embeddings.embedding_model import EmbeddingModel
from rag_project.embeddings.embedding_client import EmbeddingClient
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger


class OptimizedRAGRetriever:
    """
    Optimized retriever that only loads components needed for retrieval.

    Uses only:
    - EmbeddingModel/EmbeddingClient: For query embedding
    - MilvusManager: For vector search

    Does NOT load:
    - ConfigurableChunker: Only needed for indexing
    - ChunkStorage: Only needed for indexing
    """

    def __init__(
        self,
        milvus_config_path: str = "config/milvus_config.yaml"
    ):
        """
        Initialize retriever with minimal components.

        Args:
            milvus_config_path: Path to Milvus configuration
        """
        self.config = load_config(milvus_config_path)
        self.embedding_config = self.config.get('embedding', {})

        # Initialize ONLY what's needed for retrieval

        # 1. Embedding model (for query embedding)
        embedding_mode = self.embedding_config.get('mode', 'local')

        if embedding_mode == 'api':
            logger.info("Using Embedding API client")
            self.embedding_model = EmbeddingClient(milvus_config_path)
        else:
            logger.info("Using local Embedding model")
            self.embedding_model = EmbeddingModel(milvus_config_path, load_on_init=True)

        # 2. Milvus manager (for vector search)
        self.milvus_manager = MilvusManager(milvus_config_path)

        logger.info("OptimizedRAGRetriever initialized (retrieval only)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"doc_type": ["news"]})

        Returns:
            List of results, each containing:
                - text: Document chunk text
                - score: Similarity score
                - metadata: Document metadata dict
        """
        logger.info(f"Searching for: {query} (top_k={top_k})")

        # Generate query embedding
        query_vector = self.embedding_model.embed_text(query)

        # Search Milvus
        results = self.milvus_manager.search(
            query_vector.tolist(),
            top_k=top_k,
            filters=filters
        )

        # Transform to standard format
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.get('text', ''),
                'score': result.get('score', 0.0),
                'metadata': {
                    'doc_type': result.get('doc_type'),
                    'source': result.get('source'),
                    'publish_date': result.get('publish_date'),
                    'page_number': result.get('page_number'),
                    'title': result.get('title'),
                }
            })

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def search_multiple(
        self,
        queries: List[str],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        Search for multiple queries

        Args:
            queries: List of search queries
            top_k: Number of results per query
            filters: Optional metadata filters

        Returns:
            Dictionary mapping query to list of results
        """
        logger.info(f"Searching {len(queries)} queries")

        all_results = {}
        for query in queries:
            all_results[query] = self.search(query, top_k=top_k, filters=filters)

        return all_results
