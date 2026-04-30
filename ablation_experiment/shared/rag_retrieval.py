"""
RAG Retrieval Wrapper

Reuses rag_project's RAGRetriever for document search.
Provides simplified interface for all ablation groups.
"""

import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.agent.retriever import RAGRetriever


def create_retriever() -> RAGRetriever:
    """Create a RAG retriever using the project's configuration."""
    return RAGRetriever(agent_config_path="config/agent_config.yaml")


def search_documents(retriever: RAGRetriever, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Search for documents using the RAG retriever."""
    try:
        return retriever.search(query, top_k=top_k)
    except Exception as e:
        print(f"  [WARN] Search failed for query '{query[:50]}...': {e}")
        return []


def deduplicate_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate documents by text hash.
    Source: researcher.py:214-242

    Args:
        documents: List of document dictionaries with 'text' field

    Returns:
        Deduplicated list of documents
    """
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        text = doc.get("text", "")
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if text_hash in seen_hashes:
            continue

        seen_hashes.add(text_hash)
        unique_docs.append(doc)

    return unique_docs


def multi_query_search(
    retriever: RAGRetriever,
    queries: List[str],
    top_k_per_query: int = 20,
    max_total: int = 20
) -> List[Dict[str, Any]]:
    """
    Execute multi-query retrieval and deduplicate results.
    Source: researcher.py:60-96 (simplified)

    Args:
        retriever: RAGRetriever instance
        queries: List of search queries
        top_k_per_query: Top K results per query
        max_total: Maximum total documents to return

    Returns:
        Deduplicated list of documents
    """
    all_docs = []
    for query in queries:
        docs = search_documents(retriever, query, top_k=top_k_per_query)
        all_docs.extend(docs)

    # Deduplicate
    unique_docs = deduplicate_documents(all_docs)

    # Sort by score if available
    if unique_docs and "score" in unique_docs[0]:
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

    return unique_docs[:max_total]
