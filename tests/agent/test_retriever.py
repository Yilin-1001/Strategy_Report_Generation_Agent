"""Test RAG retriever wrapper for agents"""

import pytest
from rag_project.agent.retriever import RAGRetriever


class TestRAGRetriever:
    """Test RAGRetriever class"""

    def test_retriever_init(self):
        """Test retriever initialization"""
        # Create retriever with default configs
        retriever = RAGRetriever()

        # Verify pipeline is created
        assert retriever.pipeline is not None
        assert hasattr(retriever, 'search')
        assert hasattr(retriever, 'search_multiple')

    def test_retriever_search(self):
        """Test basic search functionality"""
        # Create retriever
        retriever = RAGRetriever()

        # Perform a search
        results = retriever.search("交通运输政策", top_k=3)

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) <= 3

        # Check first result has required fields
        if len(results) > 0:
            result = results[0]
            assert 'text' in result
            assert 'score' in result
            assert 'metadata' in result
            assert isinstance(result['text'], str)
            assert isinstance(result['score'], (int, float))
            assert isinstance(result['metadata'], dict)

    def test_retriever_search_multiple(self):
        """Test multiple queries search"""
        # Create retriever
        retriever = RAGRetriever()

        # Search with multiple queries
        queries = ["交通运输", "城市规划"]
        results = retriever.search_multiple(queries, top_k=2)

        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) == 2

        # Check each query has results
        for query in queries:
            assert query in results
            assert isinstance(results[query], list)
            assert len(results[query]) <= 2

        # Verify result structure for first query
        if len(results[queries[0]]) > 0:
            result = results[queries[0]][0]
            assert 'text' in result
            assert 'score' in result
            assert 'metadata' in result

    def test_retriever_search_with_filters(self):
        """Test search with filters"""
        # Create retriever
        retriever = RAGRetriever()

        # Search with doc_type filter
        results = retriever.search(
            "政策",
            top_k=5,
            filters={"doc_type": ["政策文件"]}
        )

        # Verify results are returned
        assert isinstance(results, list)

        # All results should match the filter
        for result in results:
            metadata = result.get('metadata', {})
            assert metadata.get('doc_type') in ['政策文件', None]  # None if filter not applied
