"""
Analyst Node - Extracts key facts and generates insights from retrieved documents

This node analyzes the documents retrieved by the Researcher node and:
- Generates a summary of the retrieved documents (limited to 10 docs)
- Extracts key facts from the documents
- Generates insights based on the analysis
- Merges results into the existing chapter scratchpad
"""

import json
import logging
from typing import Dict, List, Any

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def analyst_node(state: Dict[str, Any], llm_manager) -> Dict[str, Any]:
    """
    Analyze retrieved documents to extract key facts and generate insights.

    This node:
    1. Takes retrieved_docs from chapter_scratchpad (from Researcher node)
    2. Generates a document summary (limited to 10 documents)
    3. Calls LLM to extract key_facts and insights (JSON format)
    4. Merges results into existing scratchpad (preserves queries/retrieved_docs)
    5. Returns updated chapter_scratchpad

    Args:
        state: Current workflow state containing:
            - chapter_question: The research question
            - chapter_context: Optional context for the question
            - chapter_scratchpad: Dict with intermediate results including:
                - queries: List of search queries used
                - retrieved_docs: List of retrieved documents
        llm_manager: LLMManager instance for analysis

    Returns:
        Dict with updates:
            - chapter_scratchpad: Updated scratchpad with:
                - queries: Preserved original queries
                - retrieved_docs: Preserved original documents
                - document_summary: Summary of documents
                - key_facts: List of extracted key facts
                - insights: List of generated insights

    Example:
        >>> state = {
        ...     "chapter_question": "What is the status of China's transportation?",
        ...     "chapter_scratchpad": {
        ...         "queries": ["China transportation"],
        ...         "retrieved_docs": [{"text": "...", "metadata": {...}}]
        ...     }
        ... }
        >>> result = analyst_node(state, llm_manager)
        >>> scratchpad = result["chapter_scratchpad"]
        >>> "key_facts" in scratchpad
        True
        >>> "insights" in scratchpad
        True
    """
    chapter_question = state.get("chapter_question", "")
    chapter_context = state.get("chapter_context", "")
    chapter_scratchpad = state.get("chapter_scratchpad", {})

    # Get retrieved documents from scratchpad
    retrieved_docs = chapter_scratchpad.get("retrieved_docs", [])

    logger.info(f"Analyst node analyzing {len(retrieved_docs)} retrieved documents")

    # Step 1: Generate document summary (limit to 10 docs)
    document_summary = _generate_document_summary(retrieved_docs, limit=10)

    # Step 2: Extract key facts and insights using LLM
    try:
        key_facts, insights = _extract_facts_and_insights(
            chapter_question=chapter_question,
            chapter_context=chapter_context,
            document_summary=document_summary,
            llm_manager=llm_manager
        )
        logger.info(f"Extracted {len(key_facts)} key facts and {len(insights)} insights")

    except Exception as e:
        logger.error(f"Error extracting facts and insights: {e}. Using fallback.")
        key_facts, insights = _get_fallback_analysis(retrieved_docs)

    # Step 3: Merge results into scratchpad (preserve existing data)
    chapter_scratchpad["document_summary"] = document_summary
    chapter_scratchpad["key_facts"] = key_facts
    chapter_scratchpad["insights"] = insights

    return {
        "chapter_scratchpad": chapter_scratchpad
    }


def _generate_document_summary(documents: List[Dict[str, Any]], limit: int = 10) -> str:
    """
    Generate a summary of retrieved documents for LLM analysis.

    Args:
        documents: List of document dictionaries with 'text' and 'metadata' fields
        limit: Maximum number of documents to include in summary

    Returns:
        Formatted string summary of documents
    """
    # Limit documents to avoid overwhelming the LLM
    docs_to_summarize = documents[:limit]

    if not docs_to_summarize:
        return "No documents retrieved for analysis."

    summary_parts = []
    for i, doc in enumerate(docs_to_summarize, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        # Format each document
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")

        # Truncate text if too long (keep first 500 chars)
        text_preview = text[:500] if len(text) > 500 else text

        doc_entry = f"Document {i} (Source: {source}, Page: {page}):\n{text_preview}"
        summary_parts.append(doc_entry)

    return "\n\n".join(summary_parts)


def _extract_facts_and_insights(
    chapter_question: str,
    chapter_context: str,
    document_summary: str,
    llm_manager
) -> tuple[List[str], List[str]]:
    """
    Use LLM to extract key facts and generate insights from documents.

    Args:
        chapter_question: The research question
        chapter_context: Optional context
        document_summary: Summary of retrieved documents
        llm_manager: LLMManager instance

    Returns:
        Tuple of (key_facts, insights) as lists of strings

    Raises:
        Exception: If LLM call fails or response cannot be parsed
    """
    # Generate prompt for analysis
    prompt = _generate_analysis_prompt(chapter_question, chapter_context, document_summary)

    # Invoke LLM with moderate temperature for balanced analysis
    response = llm_manager.invoke(prompt, temperature=0.5)

    # Parse JSON response
    key_facts, insights = _parse_analysis_response(response)

    return key_facts, insights


def _generate_analysis_prompt(
    chapter_question: str,
    chapter_context: str,
    document_summary: str
) -> str:
    """
    Generate prompt for LLM analysis.

    Args:
        chapter_question: The research question
        chapter_context: Optional context
        document_summary: Summary of retrieved documents

    Returns:
        Formatted prompt string
    """
    context_section = f"\nContext: {chapter_context}" if chapter_context else ""

    return f"""Analyze the following retrieved documents to extract key facts and generate insights.

Research Question: {chapter_question}{context_section}

Retrieved Documents:
{document_summary}

Your task:
1. Extract 3-5 key facts from the documents that directly answer the research question
2. Generate 2-4 insights that provide deeper analysis or connect different pieces of information

Requirements for key_facts:
- Must be directly supported by the documents
- Should be specific and factual (not opinions)
- Should be concise (1-2 sentences each)
- Should directly address the research question

Requirements for insights:
- Should provide analysis beyond just facts
- Can identify patterns, trends, or relationships
- Can highlight implications or significance
- Should be thoughtful and add value

Return your response as a valid JSON object with this exact structure:
{{
    "key_facts": ["fact 1", "fact 2", "fact 3"],
    "insights": ["insight 1", "insight 2"]
}}

Return ONLY the JSON object, no additional text or explanation."""


def _parse_analysis_response(response: str) -> tuple[List[str], List[str]]:
    """
    Parse LLM response to extract key_facts and insights.

    Args:
        response: Raw LLM response string

    Returns:
        Tuple of (key_facts, insights) as lists of strings

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    # Clean up response
    response = response.strip()

    # Remove markdown code blocks if present
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    # Parse JSON
    try:
        result = json.loads(response)

        # Extract key_facts and insights
        key_facts = result.get("key_facts", [])
        insights = result.get("insights", [])

        # Ensure they are lists
        if not isinstance(key_facts, list):
            key_facts = [str(key_facts)]
        if not isinstance(insights, list):
            insights = [str(insights)]

        # Ensure all items are strings
        key_facts = [str(fact) for fact in key_facts]
        insights = [str(insight) for insight in insights]

        return key_facts, insights

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Response content: {response[:500]}")
        raise ValueError(f"Invalid JSON response: {e}")


def _get_fallback_analysis(documents: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    """
    Generate fallback analysis when LLM fails.

    Args:
        documents: List of retrieved documents

    Returns:
        Tuple of (key_facts, insights) as lists of strings
    """
    logger.warning("Using fallback analysis")

    # Extract basic facts from documents
    key_facts = []
    insights = []

    if documents:
        # Count documents
        key_facts.append(f"Retrieved {len(documents)} relevant documents")

        # Extract source information
        sources = set()
        for doc in documents[:5]:  # Check first 5 docs
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            sources.add(source)

        if sources:
            key_facts.append(f"Information来源于: {', '.join(list(sources)[:3])}")

        # Generate basic insights
        if len(documents) > 5:
            insights.append(f"Found substantial relevant documentation ({len(documents)} documents)")
        else:
            insights.append(f"Limited documentation available ({len(documents)} documents)")

        insights.append("Detailed analysis unavailable due to processing error")
    else:
        key_facts.append("No documents were retrieved for analysis")
        insights.append("Unable to generate insights without source documents")

    return key_facts, insights
