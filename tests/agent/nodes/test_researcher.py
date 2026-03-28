"""
Tests for Researcher node.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_project.agent.nodes.researcher import researcher_node


@pytest.fixture
def mock_retriever():
    """Mock RAG retriever."""
    retriever = Mock()
    return retriever


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager."""
    llm_manager = Mock()
    return llm_manager


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents."""
    return [
        {"text": "Document 1 content", "metadata": {"source": "doc1.pdf", "page": 1}, "score": 0.95},
        {"text": "Document 2 content", "metadata": {"source": "doc2.pdf", "page": 2}, "score": 0.90},
        {"text": "Document 3 content", "metadata": {"source": "doc1.pdf", "page": 3}, "score": 0.85},
    ]


def test_researcher_node(mock_retriever, mock_llm_manager, sample_retrieved_docs):
    """Test researcher node with multi-query retrieval and deduplication."""
    # Setup mock responses
    query_list = ["query 1", "query 2", "query 3", "query 4"]
    mock_llm_manager.generate_response.return_value = "\n".join(query_list)

    # Mock retriever to return different docs for each query
    def mock_retrieve(query, top_k=20):
        # Add some duplicate documents across queries to test deduplication
        if "query 1" in query:
            return sample_retrieved_docs + [
                {"text": "Duplicate doc", "metadata": {"source": "dup.pdf"}, "score": 0.80}
            ]
        elif "query 2" in query:
            return [
                {"text": "Document 4 content", "metadata": {"source": "doc3.pdf", "page": 1}, "score": 0.88},
                {"text": "Duplicate doc", "metadata": {"source": "dup.pdf"}, "score": 0.82},
            ]
        elif "query 3" in query:
            return [
                {"text": "Document 5 content", "metadata": {"source": "doc4.pdf", "page": 1}, "score": 0.87},
            ]
        else:
            return sample_retrieved_docs[:2]

    mock_retriever.retrieve.side_effect = mock_retrieve

    # Prepare state
    state = {
        "chapter_question": "What is the capital of France?",
        "chapter_context": "Geography of Europe",
        "chapter_scratchpad": {}
    }

    # Execute researcher node
    result = researcher_node(state, mock_retriever, mock_llm_manager)

    # Verify structure
    assert "chapter_scratchpad" in result
    assert "queries" in result["chapter_scratchpad"]
    assert "retrieved_docs" in result["chapter_scratchpad"]

    # Verify queries were generated
    queries = result["chapter_scratchpad"]["queries"]
    assert len(queries) >= 3
    assert len(queries) <= 5
    assert all(isinstance(q, str) for q in queries)

    # Verify retriever was called for each query
    assert mock_retriever.retrieve.call_count == len(queries)

    # Verify documents were retrieved
    retrieved_docs = result["chapter_scratchpad"]["retrieved_docs"]
    assert len(retrieved_docs) > 0
    assert all("text" in doc for doc in retrieved_docs)
    assert all("metadata" in doc for doc in retrieved_docs)

    # Verify deduplication (no duplicate texts)
    texts = [doc["text"] for doc in retrieved_docs]
    unique_texts = set(texts)
    assert len(texts) == len(unique_texts), "Documents should be deduplicated by text hash"

    # Verify top 20 limit
    assert len(retrieved_docs) <= 20, "Should return at most 20 documents"


def test_researcher_node_with_empty_retrieval(mock_retriever, mock_llm_manager):
    """Test researcher node when retrieval returns no documents."""
    # Setup mock responses
    query_list = ["query 1", "query 2"]
    mock_llm_manager.generate_response.return_value = "\n".join(query_list)
    mock_retriever.retrieve.return_value = []

    # Prepare state
    state = {
        "chapter_question": "Unknown topic",
        "chapter_context": "",
        "chapter_scratchpad": {}
    }

    # Execute researcher node
    result = researcher_node(state, mock_retriever, mock_llm_manager)

    # Verify structure even with no results
    assert "chapter_scratchpad" in result
    assert "queries" in result["chapter_scratchpad"]
    assert "retrieved_docs" in result["chapter_scratchpad"]
    assert len(result["chapter_scratchpad"]["queries"]) >= 2
    assert len(result["chapter_scratchpad"]["retrieved_docs"]) == 0


def test_researcher_node_with_llm_error(mock_retriever, mock_llm_manager):
    """Test researcher node handles LLM errors gracefully."""
    # Setup mock to raise error
    mock_llm_manager.generate_response.side_effect = Exception("LLM API Error")

    # Prepare state
    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {}
    }

    # Execute researcher node - should handle error gracefully
    result = researcher_node(state, mock_retriever, mock_llm_manager)

    # Should still return structure with error info
    assert "chapter_scratchpad" in result
    # Should have fallback queries or empty state
    assert "queries" in result["chapter_scratchpad"]


def test_researcher_node_query_generation(mock_retriever, mock_llm_manager):
    """Test that queries are properly generated from the chapter question."""
    # Setup
    chapter_question = "What are the main causes of climate change?"
    mock_llm_manager.generate_response.return_value = "climate change causes\nglobal warming effects\nenvironmental factors"

    state = {
        "chapter_question": chapter_question,
        "chapter_context": "Environmental science",
        "chapter_scratchpad": {}
    }

    mock_retriever.retrieve.return_value = []

    # Execute
    result = researcher_node(state, mock_retriever, mock_llm_manager)

    # Verify LLM was called with proper prompt
    mock_llm_manager.generate_response.assert_called_once()
    call_args = mock_llm_manager.generate_response.call_args
    assert chapter_question in str(call_args)

    # Verify queries
    queries = result["chapter_scratchpad"]["queries"]
    assert len(queries) == 3
    assert "climate change causes" in queries
    assert "global warming effects" in queries
    assert "environmental factors" in queries


def test_researcher_node_deduplication_by_text_hash(mock_retriever, mock_llm_manager):
    """Test that documents are properly deduplicated by text hash."""
    # Setup
    mock_llm_manager.generate_response.return_value = "query 1\nquery 2"

    # Create documents with duplicate text
    doc1 = {"text": "Same content", "metadata": {"source": "doc1.pdf"}, "score": 0.95}
    doc2 = {"text": "Same content", "metadata": {"source": "doc2.pdf"}, "score": 0.90}
    doc3 = {"text": "Different content", "metadata": {"source": "doc3.pdf"}, "score": 0.85}

    mock_retriever.retrieve.side_effect = [
        [doc1, doc2],  # First query returns duplicates
        [doc2, doc3],  # Second query returns one duplicate and one unique
    ]

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {}
    }

    # Execute
    result = researcher_node(state, mock_retriever, mock_llm_manager)

    # Verify deduplication
    retrieved_docs = result["chapter_scratchpad"]["retrieved_docs"]
    texts = [doc["text"] for doc in retrieved_docs]

    # Should have only 2 unique documents
    assert len(retrieved_docs) == 2
    assert len(texts) == len(set(texts)), "Should have no duplicate texts"
    assert "Same content" in texts
    assert "Different content" in texts


def test_researcher_node_top_20_limit(mock_retriever, mock_llm_manager):
    """Test that researcher node returns at most 20 documents."""
    # Setup
    query_list = [f"query {i}" for i in range(5)]  # 5 queries
    mock_llm_manager.generate_response.return_value = "\n".join(query_list)

    # Each query returns 10 documents, total would be 50 without limit
    def mock_retrieve(query, top_k=20):
        return [
            {"text": f"Document {i} from {query}", "metadata": {"source": f"doc{i}.pdf"}, "score": 0.9 - i * 0.01}
            for i in range(10)
        ]

    mock_retriever.retrieve.side_effect = mock_retrieve

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {}
    }

    # Execute
    result = researcher_node(state, mock_retriever, mock_llm_manager)

    # Verify limit
    retrieved_docs = result["chapter_scratchpad"]["retrieved_docs"]
    assert len(retrieved_docs) <= 20, "Should return at most 20 documents"
