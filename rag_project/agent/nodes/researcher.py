"""
Researcher node for multi-query retrieval and deduplication.
"""

import hashlib
from typing import Dict, List, Any
from rag_project.agent.retriever import RAGRetriever
from rag_project.agent.llm_manager import LLMManager

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


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

    # 读取评审反馈（如果有 revise:data 的反馈）
    revision_feedback = chapter_scratchpad.get("revision_feedback")
    researcher_hint = ""
    if revision_feedback and revision_feedback.get("decision") == "revise:data":
        hints = revision_feedback.get("improvement_hints", {})
        researcher_hint = hints.get("researcher", "")
        if researcher_hint:
            logger.info(f"Researcher received revision hint: {researcher_hint[:100]}")

    # Step 1: Generate initial queries (use 5 for broader coverage)
    initial_queries = _generate_queries(chapter_question, chapter_context, llm_manager, researcher_hint)
    initial_queries = initial_queries[:5]  # Start with 5 queries for better recall

    # Step 2: Execute initial retrieval
    all_docs = []
    for query in initial_queries:
        try:
            docs = retriever.search(query, top_k=20)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Error retrieving documents for query '{query}': {e}")
            continue

    # Step 3: Deduplicate
    unique_docs = _deduplicate_documents(all_docs)

    # Step 4: Adaptive retrieval - evaluate sufficiency, optionally add supplementary queries
    queries = initial_queries
    if len(unique_docs) < 10 or not _evaluate_retrieval_sufficiency(chapter_question, unique_docs, llm_manager):
        logger.info(f"Initial retrieval insufficient ({len(unique_docs)} docs). Generating supplementary queries...")
        supp_queries = _generate_supplementary_queries(chapter_question, initial_queries, unique_docs, llm_manager)
        if supp_queries:
            for query in supp_queries:
                try:
                    docs = retriever.search(query, top_k=20)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Error retrieving documents for supplementary query '{query}': {e}")
                    continue
            unique_docs = _deduplicate_documents(all_docs)
            queries = initial_queries + supp_queries
            logger.info(f"After supplementary retrieval: {len(unique_docs)} unique docs with {len(queries)} queries")
    else:
        logger.info(f"Initial retrieval sufficient ({len(unique_docs)} docs). Skipping supplementary queries.")

    # Step 5: Limit to top 20 documents
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
    llm_manager: LLMManager,
    revision_hint: str = ""
) -> List[str]:
    """
    Generate 5-7 diverse queries using LLM.

    Args:
        chapter_question: The main research question
        chapter_context: Optional context for the question
        llm_manager: LLMManager instance
        revision_hint: Optional hint from reviewer for targeted retrieval

    Returns:
        List of 5-7 query strings
    """
    # Construct prompt for query generation
    revision_section = ""
    if revision_hint:
        revision_section = f"""
IMPORTANT - Previous retrieval was insufficient:
{revision_hint}

Make sure your queries specifically address these gaps.
"""

    prompt = f"""Given the following research question, generate 5-7 diverse and specific search queries that would help retrieve relevant documents.

Research Question: {chapter_question}

Context: {chapter_context if chapter_context else "No additional context provided"}
{revision_section}
Generate 5-7 search queries (one per line) that:
1. Rephrase the question in different ways
2. Include related keywords and concepts
3. Cover different aspects of the topic
4. Are specific enough to retrieve relevant documents

Queries:"""

    try:
        response = llm_manager.invoke(
            prompt=prompt,
            temperature=0.7,  # Moderate temperature for diversity
            max_tokens=200
        )

        # Parse response into queries
        queries = _parse_query_response(response)

        # Ensure we have at least 5 queries
        if len(queries) < 5:
            # Fallback: add original question if not enough queries
            if chapter_question not in queries:
                queries.append(chapter_question)
            # Add simple variations
            while len(queries) < 5:
                queries.append(f"details about {chapter_question}")
                queries.append(f"{chapter_question} overview")
                queries.append(f"{chapter_question} analysis")

        # Limit to 7 queries
        queries = queries[:7]

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


def _evaluate_retrieval_sufficiency(chapter_question: str, docs: List[Dict], llm_manager) -> bool:
    """
    LLM 评估检索结果是否信息充足。

    Args:
        chapter_question: 研究问题
        docs: 检索到的文档列表
        llm_manager: LLM管理器

    Returns:
        bool: True表示信息充足，False表示需要补充检索
    """
    doc_count = len(docs)
    # 取前 5 个文档的前 200 字作为预览
    preview = "\n".join(
        f"Doc {i+1}: {doc.get('text', '')[:200]}..."
        for i, doc in enumerate(docs[:5])
    )

    prompt = f"""评估以下检索结果是否能充分回答研究问题。

研究问题: {chapter_question}

检索到 {doc_count} 个文档:
{preview}

请判断: 这些文档的信息是否足以支撑对该问题的深入分析？
只回答 YES 或 NO。"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.1, max_tokens=10)
        return "YES" in response.upper()
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}. Defaulting to sufficient.")
        return True  # LLM 失败时默认充足，不阻塞流程


def _generate_supplementary_queries(chapter_question: str, existing_queries: List[str], docs: List[Dict], llm_manager) -> List[str]:
    """
    基于已有检索结果，生成补充查询。

    Args:
        chapter_question: 研究问题
        existing_queries: 已有查询列表
        docs: 已检索文档列表
        llm_manager: LLM管理器

    Returns:
        List[str]: 补充查询列表（最多2个）
    """
    # 提取已有查询覆盖的关键词
    existing_keywords = " ".join(existing_queries)

    prompt = f"""基于初始检索结果，生成 2 个补充搜索查询。

研究问题: {chapter_question}
已有查询: {existing_keywords}
已检索到 {len(docs)} 个文档，但信息不够充分。

请生成 2 个不同角度的补充查询（每行一个），聚焦尚未覆盖的方面:"""

    try:
        response = llm_manager.invoke(prompt, temperature=0.5, max_tokens=200)
        queries = [q.strip().lstrip('0123456789.-) ') for q in response.strip().split('\n') if q.strip()]
        return queries[:2]
    except Exception as e:
        logger.warning(f"Supplementary query generation failed: {e}")
        return []
