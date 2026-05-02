"""
Fact Verifier Module

A-type: Citation-based verification (RAG search + source filtering)
B-type: KB semantic search verification (RAG retriever)
LLM commonsense fallback when KB has no results.
"""

import re
import time
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI

from .config import (
    EVAL_API_KEY, EVAL_BASE_URL, VERIFY_MODEL, VERIFY_TIMEOUT,
    RAG_TOP_K, CITED_CHUNKS_TOP_K, MAX_CONTEXT_TOKENS,
    LLM_CALL_INTERVAL, RAG_CALL_INTERVAL
)
from .atomic_decomposer import AtomicFact
from .citation_parser import CitationParser


@dataclass
class Verdict:
    """Verification result for a single atomic fact."""
    fact_text: str           # Original fact text
    fact_type: str           # "FACT" or "ANALYSIS"
    citation: str            # Citation text (empty if none)
    category: str            # "A" (with citation) or "B" (without) or "C" (analytical)
    result: str              # Final verdict
    detail: str              # Human-readable explanation


# === Verification Prompts ===

CITATION_VERIFY_PROMPT = """你是一个事实核查专家。请判断以下原子事实是否被给定的参考文档所支持。

原子事实：{fact}

参考文档（被引用的原文片段）：
{source_chunks}

判断要求：
1. 如果参考文档中明确包含与原子事实相符的信息，输出：supported
2. 如果参考文档中的信息与原子事实矛盾，输出：contradicted
3. 如果参考文档中没有相关信息，输出：no_info

仅输出一个判断结果，不要解释。

判断结果："""

KB_VERIFY_PROMPT = """你是一个事实核查专家。请判断以下原子事实是否被给定的知识库检索结果所支持。

原子事实：{fact}

知识库检索结果：
{kb_chunks}

判断要求：
1. 如果检索结果明确支持该事实（信息相符），输出：supported
2. 如果检索结果明确矛盾（信息冲突），输出：contradicted
3. 如果检索结果没有相关信息或信息不足判断，输出：no_info

仅输出一个判断结果。

判断结果："""

LLM_COMMONSENSE_PROMPT = """你是一个事实核查专家。请判断以下原子事实的真实性。

原子事实：{fact}

判断要求：
基于你的知识判断该事实的合理性：
- 如果这是一个关于中国政策/经济/交通领域的具体断言（如人名、政策名、事件），判断其是否合理真实
- 如果这是一个一般性趋势描述（如"政策持续加码"），判断其是否符合常识
- 如果涉及具体数据/排名，而你不确定其真实性，输出 uncertain

仅输出以下之一：
- credible: 该事实合理可信，符合常识或一般知识
- uncertain: 无法确定真实性，可能需要进一步验证
- implausible: 该事实看起来不合理或可能有误

判断结果："""


class FactVerifier:
    """Verify atomic facts against KB and LLM knowledge."""

    def __init__(self, citation_parser: CitationParser):
        self.citation_parser = citation_parser
        self.llm_client = OpenAI(
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL
        )
        self._rag_retriever = None
        self._last_llm_call = 0
        self._last_rag_call = 0

    def _get_rag_retriever(self):
        """Lazy-load RAG retriever."""
        if self._rag_retriever is None:
            from ablation_experiment.shared.rag_retrieval import create_retriever
            self._rag_retriever = create_retriever()
        return self._rag_retriever

    def _rate_limit_llm(self):
        elapsed = time.time() - self._last_llm_call
        if elapsed < LLM_CALL_INTERVAL:
            time.sleep(LLM_CALL_INTERVAL - elapsed)
        self._last_llm_call = time.time()

    def _rate_limit_rag(self):
        elapsed = time.time() - self._last_rag_call
        if elapsed < RAG_CALL_INTERVAL:
            time.sleep(RAG_CALL_INTERVAL - elapsed)
        self._last_rag_call = time.time()

    def _llm_call(self, prompt: str) -> str:
        """Call LLM for verification with retry."""
        for attempt in range(3):
            self._rate_limit_llm()
            try:
                response = self.llm_client.chat.completions.create(
                    model=VERIFY_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个严谨的事实核查专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=50,
                    timeout=VERIFY_TIMEOUT
                )
                return response.choices[0].message.content.strip().lower()
            except Exception as e:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise

    def _rag_search(self, query: str, top_k: int = None) -> List[dict]:
        """Execute RAG semantic search."""
        self._rate_limit_rag()
        retriever = self._get_rag_retriever()

        from ablation_experiment.shared.rag_retrieval import search_documents
        try:
            return search_documents(retriever, query, top_k=top_k or RAG_TOP_K)
        except Exception:
            return []

    def verify_fact(self, fact: AtomicFact) -> Verdict:
        """
        Verify a single atomic fact.

        Routes to A-type (with citation) or B-type (without citation).
        Analytical facts are passed through without verification.
        """
        # Analytical facts: no verification needed
        if fact.fact_type == "ANALYSIS":
            return Verdict(
                fact_text=fact.text,
                fact_type=fact.fact_type,
                citation=fact.citation,
                category="C",
                result="analytical",
                detail="Analytical claim, excluded from hallucination metrics"
            )

        # Route based on citation presence
        if fact.citation:
            return self._verify_a_type(fact)
        else:
            return self._verify_b_type(fact)

    def _verify_a_type(self, fact: AtomicFact) -> Verdict:
        """A-type: Citation-based verification."""
        citation = fact.citation

        # Step 1: Verify citation authenticity
        is_valid, source_file = self.citation_parser.verify_citation(citation)

        if not is_valid:
            return Verdict(
                fact_text=fact.text,
                fact_type=fact.fact_type,
                citation=citation,
                category="A",
                result="citation_hallucination",
                detail=f"Cited source not found in KB: {citation}"
            )

        # Step 2: RAG search + source filtering
        cited_chunks = self._find_cited_chunks(fact.text, citation)

        if not cited_chunks:
            # No relevant chunks from cited source, fall back to B-type
            return self._verify_b_type_no_citation(fact, "A(no_match)")

        # Step 3: LLM verification against cited chunks
        source_text = self._format_chunks(cited_chunks)
        prompt = CITATION_VERIFY_PROMPT.format(
            fact=fact.text,
            source_chunks=source_text
        )
        result = self._llm_call(prompt)

        # Map result
        if result == "supported":
            return Verdict(
                fact_text=fact.text, fact_type=fact.fact_type,
                citation=citation, category="A",
                result="supported_by_source",
                detail=f"Verified against cited source: {source_file}"
            )
        elif result == "contradicted":
            return Verdict(
                fact_text=fact.text, fact_type=fact.fact_type,
                citation=citation, category="A",
                result="contradicted",
                detail=f"Contradicted by cited source: {source_file}"
            )
        else:
            return Verdict(
                fact_text=fact.text, fact_type=fact.fact_type,
                citation=citation, category="A",
                result="unsupported",
                detail=f"Cited source exists but does not support claim"
            )

    def _verify_b_type(self, fact: AtomicFact) -> Verdict:
        """B-type: KB semantic search verification."""
        return self._verify_b_type_no_citation(fact, "B")

    def _verify_b_type_no_citation(self, fact: AtomicFact, category: str) -> Verdict:
        """B-type verification (can be called as fallback from A-type)."""
        # Step 1: RAG semantic search
        kb_results = self._rag_search(fact.text)

        if kb_results and len(kb_results) >= 2:
            # Step 2: LLM verification against KB results
            kb_text = self._format_search_results(kb_results[:3])
            prompt = KB_VERIFY_PROMPT.format(
                fact=fact.text,
                kb_chunks=kb_text
            )
            result = self._llm_call(prompt)

            if result == "supported":
                return Verdict(
                    fact_text=fact.text, fact_type=fact.fact_type,
                    citation=fact.citation, category=category,
                    result="supported_by_kb",
                    detail="Supported by KB search results"
                )
            elif result == "contradicted":
                return Verdict(
                    fact_text=fact.text, fact_type=fact.fact_type,
                    citation=fact.citation, category=category,
                    result="contradicted",
                    detail="Contradicted by KB search results"
                )
            # no_info falls through to commonsense

        # Step 3: LLM commonsense judgment
        prompt = LLM_COMMONSENSE_PROMPT.format(fact=fact.text)
        result = self._llm_call(prompt)

        if result == "credible":
            return Verdict(
                fact_text=fact.text, fact_type=fact.fact_type,
                citation=fact.citation, category=category,
                result="supported_by_parametric",
                detail="Supported by LLM parametric knowledge (commonsense)"
            )
        elif result == "uncertain":
            return Verdict(
                fact_text=fact.text, fact_type=fact.fact_type,
                citation=fact.citation, category=category,
                result="unverifiable",
                detail="Cannot verify, LLM uncertain"
            )
        else:
            return Verdict(
                fact_text=fact.text, fact_type=fact.fact_type,
                citation=fact.citation, category=category,
                result="unsupported",
                detail="LLM judges as implausible"
            )

    def _find_cited_chunks(self, fact_text: str, citation: str) -> List[dict]:
        """
        Find relevant chunks from a cited source.
        Strategy: RAG search → filter by cited source.
        Fallback: ngram matching within cited source's chunks.
        """
        # Step 1: RAG search (get more results for better filtering)
        all_results = self._rag_search(fact_text, top_k=20)

        # Step 2: Filter for cited source
        is_valid, source_file = self.citation_parser.verify_citation(citation)
        if not is_valid:
            return []

        cited_results = []
        for doc in all_results:
            doc_source = doc.get('metadata', {}).get('source', '')
            if source_file in doc_source or doc_source in source_file:
                cited_results.append(doc)

        if cited_results:
            return cited_results[:CITED_CHUNKS_TOP_K]

        # Step 3: Fallback - ngram matching within source chunks
        source_chunks = self.citation_parser.source_index.get(source_file, [])
        return self._rank_by_ngram(source_chunks, fact_text, CITED_CHUNKS_TOP_K)

    def _rank_by_ngram(self, chunks: List[dict], fact_text: str, top_k: int) -> List[dict]:
        """Rank chunks by ngram overlap with fact text."""
        # Generate 4-grams from fact
        clean = re.sub(r'[，。、；：\u201c\u201d\u2018\u2019【】《》（）\[\]来源:\s]', '', fact_text)
        ngrams = [clean[i:i+4] for i in range(max(0, len(clean)-3))]

        if not ngrams:
            return chunks[:top_k]

        scored = []
        for chunk in chunks:
            text = chunk.get('text', '')
            score = sum(1 for ng in ngrams if ng in text)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:top_k] if s > 0]

    def _format_chunks(self, chunks: List[dict]) -> str:
        """Format chunks for LLM prompt."""
        texts = []
        total_len = 0
        for chunk in chunks:
            text = chunk.get('text', '')[:500]
            if total_len + len(text) > MAX_CONTEXT_TOKENS:
                break
            texts.append(text)
            total_len += len(text)
        return "\n---\n".join(texts)

    def _format_search_results(self, results: List[dict]) -> str:
        """Format RAG search results for LLM prompt."""
        texts = []
        total_len = 0
        for doc in results:
            text = doc.get('text', '')[:400] if isinstance(doc, dict) else str(doc)[:400]
            if total_len + len(text) > MAX_CONTEXT_TOKENS:
                break
            texts.append(text)
            total_len += len(text)
        return "\n---\n".join(texts)