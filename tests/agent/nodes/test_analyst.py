"""
Tests for Analyst node.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rag_project.agent.nodes.analyst import analyst_node


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager."""
    llm_manager = Mock()
    return llm_manager


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents from researcher."""
    return [
        {"text": "China's transportation industry has grown rapidly in the past decade.", "metadata": {"source": "doc1.pdf", "page": 1}, "score": 0.95},
        {"text": "High-speed rail network reached 40,000 km by 2022.", "metadata": {"source": "doc2.pdf", "page": 2}, "score": 0.90},
        {"text": "Investment in infrastructure exceeded 3 trillion yuan.", "metadata": {"source": "doc1.pdf", "page": 3}, "score": 0.85},
        {"text": "Urban transit systems expanded to 50 cities.", "metadata": {"source": "doc3.pdf", "page": 1}, "score": 0.88},
    ]


@pytest.fixture
def sample_llm_response():
    """Sample LLM response with key facts and insights."""
    return """```json
{
    "key_facts": [
        "China's high-speed rail network reached 40,000 km by 2022",
        "Infrastructure investment exceeded 3 trillion yuan",
        "Urban transit systems expanded to 50 cities"
    ],
    "insights": [
        "The rapid expansion demonstrates China's commitment to infrastructure development",
        "High-speed rail has become a cornerstone of China's transportation strategy",
        "Urban transit expansion reflects rapid urbanization trends"
    ]
}
```"""


def test_analyst_node(mock_llm_manager, sample_retrieved_docs, sample_llm_response):
    """Test analyst node with retrieved documents."""
    # Setup mock response
    mock_llm_manager.invoke.return_value = sample_llm_response

    # Prepare state with retrieved docs from researcher
    state = {
        "chapter_question": "What is the current status of China's transportation industry?",
        "chapter_context": "Industry analysis",
        "chapter_scratchpad": {
            "queries": ["China transportation industry", "high-speed rail development"],
            "retrieved_docs": sample_retrieved_docs
        }
    }

    # Execute analyst node
    result = analyst_node(state, mock_llm_manager)

    # Verify structure
    assert "chapter_scratchpad" in result

    # Verify original data is preserved
    scratchpad = result["chapter_scratchpad"]
    assert "queries" in scratchpad
    assert "retrieved_docs" in scratchpad
    assert scratchpad["queries"] == ["China transportation industry", "high-speed rail development"]
    assert len(scratchpad["retrieved_docs"]) == 4

    # Verify new analysis results
    assert "key_facts" in scratchpad
    assert "insights" in scratchpad
    assert "document_summary" in scratchpad

    # Verify key facts
    key_facts = scratchpad["key_facts"]
    assert isinstance(key_facts, list)
    assert len(key_facts) == 3
    assert "high-speed rail" in key_facts[0].lower()

    # Verify insights
    insights = scratchpad["insights"]
    assert isinstance(insights, list)
    assert len(insights) == 3

    # Verify document summary (limited to 10 docs)
    doc_summary = scratchpad["document_summary"]
    assert isinstance(doc_summary, str)
    assert len(doc_summary) > 0


def test_analyst_node_with_many_documents(mock_llm_manager):
    """Test analyst node limits document summary to 10 documents."""
    # Create 15 sample documents
    many_docs = [
        {"text": f"Document content {i}", "metadata": {"source": f"doc{i}.pdf"}, "score": 0.9 - i * 0.01}
        for i in range(15)
    ]

    mock_llm_manager.invoke.return_value = """```json
{
    "key_facts": ["Fact 1", "Fact 2"],
    "insights": ["Insight 1"]
}
```"""

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["query 1"],
            "retrieved_docs": many_docs
        }
    }

    # Execute analyst node
    result = analyst_node(state, mock_llm_manager)

    # Verify document summary is limited
    scratchpad = result["chapter_scratchpad"]
    assert "document_summary" in scratchpad

    # Check that invoke was called with limited documents
    call_args = mock_llm_manager.invoke.call_args
    prompt = call_args[0][0]

    # Count how many documents are in the summary
    # Count lines starting with "Document X (Source:" to accurately count entries
    import re
    doc_entries = re.findall(r'Document \d+ \(Source:', prompt)
    doc_count = len(doc_entries)
    assert doc_count <= 10, f"Document summary should include at most 10 documents, but found {doc_count}"


def test_analyst_node_preserves_existing_scratchpad_data(mock_llm_manager, sample_retrieved_docs):
    """Test that analyst node preserves existing scratchpad data."""
    mock_llm_manager.invoke.return_value = """```json
{
    "key_facts": ["Fact 1"],
    "insights": ["Insight 1"]
}
```"""

    # Prepare state with additional data in scratchpad
    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["query 1", "query 2"],
            "retrieved_docs": sample_retrieved_docs,
            "custom_field": "custom_value",
            "another_field": 123
        }
    }

    # Execute analyst node
    result = analyst_node(state, mock_llm_manager)

    # Verify all original data is preserved
    scratchpad = result["chapter_scratchpad"]
    assert "queries" in scratchpad
    assert "retrieved_docs" in scratchpad
    assert scratchpad["custom_field"] == "custom_value"
    assert scratchpad["another_field"] == 123

    # Verify new analysis is added
    assert "key_facts" in scratchpad
    assert "insights" in scratchpad


def test_analyst_node_with_empty_retrieval(mock_llm_manager):
    """Test analyst node when no documents were retrieved."""
    mock_llm_manager.invoke.return_value = """```json
{
    "key_facts": [],
    "insights": ["No relevant documents found"]
}
```"""

    state = {
        "chapter_question": "Unknown topic",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["unknown query"],
            "retrieved_docs": []
        }
    }

    # Execute analyst node
    result = analyst_node(state, mock_llm_manager)

    # Verify structure even with no documents
    scratchpad = result["chapter_scratchpad"]
    assert "key_facts" in scratchpad
    assert "insights" in scratchpad
    assert "document_summary" in scratchpad


def test_analyst_node_with_llm_error(mock_llm_manager, sample_retrieved_docs):
    """Test analyst node handles LLM errors gracefully."""
    # Setup mock to raise error
    mock_llm_manager.invoke.side_effect = Exception("LLM API Error")

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["query 1"],
            "retrieved_docs": sample_retrieved_docs
        }
    }

    # Execute analyst node - should handle error gracefully
    result = analyst_node(state, mock_llm_manager)

    # Should still return structure with error info
    scratchpad = result["chapter_scratchpad"]
    assert "queries" in scratchpad
    assert "retrieved_docs" in scratchpad

    # Should have fallback analysis or error indication
    assert "key_facts" in scratchpad
    assert "insights" in scratchpad


def test_analyst_node_json_parsing(mock_llm_manager, sample_retrieved_docs):
    """Test that analyst node properly parses JSON from LLM response."""
    # Test with various JSON formats
    test_cases = [
        # Standard JSON with code blocks
        (
            '```json\n{"key_facts": ["fact1"], "insights": ["insight1"]}\n```',
            ["fact1"],
            ["insight1"]
        ),
        # Plain JSON without code blocks
        (
            '{"key_facts": ["fact2"], "insights": ["insight2"]}',
            ["fact2"],
            ["insight2"]
        ),
        # JSON with extra whitespace
        (
            '\n\n  {"key_facts": ["fact3"], "insights": ["insight3"]}  \n\n',
            ["fact3"],
            ["insight3"]
        ),
    ]

    for i, (response, expected_facts, expected_insights) in enumerate(test_cases):
        mock_llm_manager.invoke.reset_mock()
        mock_llm_manager.invoke.return_value = response

        state = {
            "chapter_question": f"Test question {i}",
            "chapter_context": "",
            "chapter_scratchpad": {
                "queries": ["query 1"],
                "retrieved_docs": sample_retrieved_docs
            }
        }

        result = analyst_node(state, mock_llm_manager)

        # Verify parsing worked
        scratchpad = result["chapter_scratchpad"]
        assert scratchpad["key_facts"] == expected_facts
        assert scratchpad["insights"] == expected_insights


def test_analyst_node_invalid_json_handling(mock_llm_manager, sample_retrieved_docs):
    """Test analyst node handles invalid JSON gracefully."""
    # Return invalid JSON
    mock_llm_manager.invoke.return_value = "This is not valid JSON at all"

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["query 1"],
            "retrieved_docs": sample_retrieved_docs
        }
    }

    # Execute analyst node - should handle error gracefully
    result = analyst_node(state, mock_llm_manager)

    # Should still return structure with fallback
    scratchpad = result["chapter_scratchpad"]
    assert "queries" in scratchpad
    assert "retrieved_docs" in scratchpad

    # Should have empty or fallback analysis
    assert "key_facts" in scratchpad
    assert "insights" in scratchpad


def test_analyst_node_generates_summary(mock_llm_manager, sample_retrieved_docs):
    """Test that analyst node generates document summary for LLM."""
    mock_llm_manager.invoke.return_value = """```json
{
    "key_facts": ["fact1"],
    "insights": ["insight1"]
}
```"""

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["query 1"],
            "retrieved_docs": sample_retrieved_docs
        }
    }

    # Execute analyst node
    result = analyst_node(state, mock_llm_manager)

    # Verify LLM was called
    assert mock_llm_manager.invoke.called

    # Verify the prompt includes document information
    call_args = mock_llm_manager.invoke.call_args
    prompt = call_args[0][0]

    # Check that prompt contains question and document references
    assert "Test question" in prompt
    assert "Document" in prompt or "document" in prompt.lower()


def test_analyst_node_temperature(mock_llm_manager, sample_retrieved_docs):
    """Test that analyst node uses appropriate temperature for LLM."""
    mock_llm_manager.invoke.return_value = """```json
{
    "key_facts": ["fact1"],
    "insights": ["insight1"]
}
```"""

    state = {
        "chapter_question": "Test question",
        "chapter_context": "",
        "chapter_scratchpad": {
            "queries": ["query 1"],
            "retrieved_docs": sample_retrieved_docs
        }
    }

    # Execute analyst node
    result = analyst_node(state, mock_llm_manager)

    # Verify LLM was called with temperature parameter
    call_kwargs = mock_llm_manager.invoke.call_args[1]
    assert "temperature" in call_kwargs
    # Analyst should use moderate temperature for balanced analysis
    assert 0.3 <= call_kwargs["temperature"] <= 0.7
