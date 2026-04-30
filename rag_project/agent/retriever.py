"""RAG Retriever wrapper for agents

This module provides a simplified interface to the RAG Pipeline
for use by AI agents. Supports both dense-only and hybrid retrieval
strategies via agent_config.yaml.
"""

from typing import List, Dict, Optional
from rag_project.utils.logger import logger
from rag_project.utils.config_loader import load_config


class RAGRetriever:
    """Wrapper around RAGPipeline for agent consumption

    Supports retrieval strategy selection via config:
    - "dense": Pure vector search (RAGPipeline)
    - "hybrid": Dense + BM25 hybrid search (HybridRAGPipeline)
    """

    def __init__(
        self,
        chunking_config_path: str = "config/chunking_config.yaml",
        milvus_config_path: str = "config/milvus_config.yaml",
        knowledge_base_path: str = "知识库/知识库",
        agent_config_path: str = "config/agent_config.yaml",
    ):
        """Initialize retriever with appropriate RAG pipeline

        Args:
            chunking_config_path: Path to chunking configuration
            milvus_config_path: Path to Milvus configuration
            knowledge_base_path: Path to knowledge base directory
            agent_config_path: Path to agent configuration (retrieval strategy)
        """
        # Read retrieval strategy from agent config
        agent_config = load_config(agent_config_path)
        retrieval_cfg = agent_config.get("retrieval", {})
        strategy = retrieval_cfg.get("strategy", "dense")

        # Store hybrid params for search calls
        self._hybrid_ranker = retrieval_cfg.get("hybrid_ranker", "rrf")
        self._dense_weight = retrieval_cfg.get("dense_weight", 0.7)
        self._sparse_weight = retrieval_cfg.get("sparse_weight", 0.3)
        self._strategy = strategy

        if strategy == "hybrid":
            from rag_project.pipeline_hybrid import HybridRAGPipeline
            self.pipeline = HybridRAGPipeline(
                chunking_config_path=chunking_config_path,
                milvus_config_path=milvus_config_path,
                knowledge_base_path=knowledge_base_path,
            )
            logger.info(
                f"RAGRetriever initialized with HYBRID strategy "
                f"(ranker={self._hybrid_ranker}, "
                f"dense={self._dense_weight}, sparse={self._sparse_weight})"
            )
        else:
            from rag_project.pipeline import RAGPipeline
            self.pipeline = RAGPipeline(
                chunking_config_path=chunking_config_path,
                milvus_config_path=milvus_config_path,
                knowledge_base_path=knowledge_base_path,
            )
            logger.info("RAGRetriever initialized with DENSE strategy")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for relevant documents

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
        logger.info(f"Searching for: {query} (top_k={top_k}, strategy={self._strategy})")

        results = self.pipeline.search(
            query=query,
            top_k=top_k,
            filters=filters,
            use_reranker=True,
        )

        # Transform to standard format
        formatted_results = []
        for result in results:
            metadata = result.get('metadata', {})
            formatted_results.append({
                'text': result.get('text', ''),
                'score': result.get('score', 0.0),
                'metadata': {
                    'doc_type': metadata.get('doc_type'),
                    'source': metadata.get('source'),
                    'publish_date': metadata.get('publish_date'),
                    'page_number': metadata.get('page_number'),
                    'title': metadata.get('title'),
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
        """Search for multiple queries

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
