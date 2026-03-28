"""
Researcher node for multi-query retrieval and deduplication.
"""

import hashlib
from typing import Dict, List, Any
from rag_project.agent.retriever import RAGRetriever
from rag_project.agent.llm_manager import LLMManager


def researcher_node(
    state: Dict[str, Any],
    retriever: RAGRetriever,
    llm_manager: LLMManager
) -> Dict[str, Any]:
    """
    Researcher node that generates multiple queries and retrieves documents from RAG system.

    This node:
    1. Generates 3-5 queries using LLM based on the chapter question
    2. Executes multi-query retrieval via RAGRetriever
    3. Deduplicates results by text hash
    4. Returns top 20 documents

    Args:
        state: Current workflow state containing:
            - chapter_question: The research question
            - chapter_context: Optional context for the question
            - chapter_scratchpad: Dict to store intermediate results
        retriever: RAGRetriever instance for document retrieval
        llm_manager: LLMManager instance for query generation

    Returns:
        Updated state with chapter_scratchpad containing:
            - queries: List of generated queries
            - retrieved_docs: List of deduplicated retrieved documents
    """
    chapter_question = state.get("chapter_question", "")
    chapter_context = state.get("chapter_context", "")
    chapter_scratchpad = state.get("chapter_scratchpad", {})

    # Step 1: Generate multiple queries using LLM
    queries = _generate_queries(chapter_question, chapter_context, llm_manager)

    # Step 2: Execute multi-query retrieval
    all_docs = []
    for query in queries:
        try:
            docs = retriever.retrieve(query, top_k=20)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error retrieving documents for query '{query}': {e}")
            # Continue with other queries
            continue

    # Step 3: Deduplicate documents by text hash
    unique_docs = _deduplicate_documents(all_docs)

    # Step 4: Limit to top 20 documents
    # Sort by score if available, otherwise keep original order
    if unique_docs and "score" in unique_docs[0]:
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

    top_docs = unique_docs[:20]

    # Update scratchpad
    chapter_scratchpad["queries"] = queries
    chapter_scratchpad["retrieved_docs"] = top_docs

    return {
        "chapter_scratchpad": chapter_scratchpad
    }


def _generate_queries(
    chapter_question: str,
    chapter_context: str,
    llm_manager: LLMManager
) -> List[str]:
    """
    Generate 3-5 diverse queries using LLM.

    Args:
        chapter_question: The main research question
        chapter_context: Optional context for the question
        llm_manager: LLMManager instance

    Returns:
        List of 3-5 query strings
    """
    # Construct prompt for query generation
    prompt = f"""Given the following research question, generate 3-5 diverse and specific search queries that would help retrieve relevant documents.

Research Question: {chapter_question}

Context: {chapter_context if chapter_context else "No additional context provided"}

Generate 3-5 search queries (one per line) that:
1. Rephrase the question in different ways
2. Include related keywords and concepts
3. Cover different aspects of the topic
4. Are specific enough to retrieve relevant documents

Queries:"""

    try:
        response = llm_manager.generate_response(
            prompt=prompt,
            temperature=0.7,  # Moderate temperature for diversity
            max_tokens=200
        )

        # Parse response into queries
        queries = _parse_query_response(response)

        # Ensure we have 3-5 queries
        if len(queries) < 3:
            # Fallback: add original question if not enough queries
            if chapter_question not in queries:
                queries.append(chapter_question)
            # Add simple variations
            queries.append(f"details about {chapter_question}")
            queries.append(f"{chapter_question} explanation")

        # Limit to 5 queries
        queries = queries[:5]

        return queries

    except Exception as e:
        print(f"Error generating queries: {e}")
        # Fallback: return original question and simple variations
        return [
            chapter_question,
            f"information about {chapter_question}",
            f"{chapter_question} overview"
        ]


def _parse_query_response(response: str) -> List[str]:
    """
    Parse LLM response into a list of queries.

    Args:
        response: LLM response string

    Returns:
        List of query strings
    """
    # Split by newlines and clean up
    lines = response.strip().split("\n")

    queries = []
    for line in lines:
        # Remove numbering, bullets, etc.
        line = line.strip()
        line = line.lstrip("0123456789.-*•°")
        line = line.strip()

        # Skip empty lines or too short queries
        if len(line) < 3:
            continue

        queries.append(line)

    return queries


def _deduplicate_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate documents by text hash.

    Args:
        documents: List of document dictionaries with 'text' field

    Returns:
        Deduplicated list of documents
    """
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        # Get text content
        text = doc.get("text", "")

        # Create hash of text content
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Skip if we've seen this document
        if text_hash in seen_hashes:
            continue

        # Add to unique documents
        seen_hashes.add(text_hash)
        unique_docs.append(doc)

    return unique_docs
