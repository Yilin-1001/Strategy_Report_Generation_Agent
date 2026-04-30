"""
Ablation Experiment - Group 2: Multi-Agent, No Memory (WITH Strategist)

Full multi-agent workflow with three specialized LLM instances
(researcher, analyst, writer) each with different temperatures,
WITH strategic blueprint generation, but WITHOUT memory management.

Key ablation:
- Has strategist node (generates blueprint from context_pool after Chapter 3)
- NO rolling context compression (context_summary is always empty)
- Each chapter starts fresh without compressed context from previous chapters
"""

import os
import logging
from typing import Dict, Any, Optional

from ablation_experiment.config import (
    TEST_QUERY,
    CHAPTER_PLAN,
    REPORTS_DIR,
    AGENT_TEMPERATURES,
)
from ablation_experiment.shared.llm_client import (
    LLMClient,
    parse_json_response,
    parse_queries_response,
)
from ablation_experiment.shared.prompts import (
    generate_query_prompt,
    generate_analysis_prompt,
    generate_writing_prompt,
    generate_swot_extraction_prompt,
    generate_blueprint_prompt,
    generate_diagnosis_compression_prompt,
)
from ablation_experiment.shared.rag_retrieval import (
    create_retriever,
    multi_query_search,
)
from ablation_experiment.shared.report_utils import (
    generate_document_summary,
    extract_filename_mapping,
    extract_chapter_question,
    assemble_report,
)

logger = logging.getLogger(__name__)

# --- System prompts for the three specialized agents ---
# Plus strategist system prompt for blueprint generation

RESEARCHER_SYSTEM_PROMPT = (
    "You are the Researcher Agent for a RAG system. Your role is to:\n"
    "- Conduct thorough information retrieval from the knowledge base\n"
    "- Find relevant documents and evidence\n"
    "- Extract key facts and details\n"
    "- Verify information accuracy\n"
    "Be precise, thorough, and focus on finding the most relevant information."
)

ANALYST_SYSTEM_PROMPT = (
    "You are the Analyst Agent for a RAG system. Your role is to:\n"
    "- Analyze retrieved information critically\n"
    "- Identify patterns and relationships\n"
    "- Compare and contrast different sources\n"
    "- Provide insights and conclusions\n"
    "Be analytical, balanced, and focus on deeper understanding."
)

WRITER_SYSTEM_PROMPT = (
    "You are the Writer Agent for a RAG system. Your role is to:\n"
    "- Synthesize information into clear, coherent responses\n"
    "- Structure content logically\n"
    "- Ensure readability and engagement\n"
    "- Adapt tone to the context\n"
    "Be creative, clear, and focus on effective communication."
)

STRATEGIST_SYSTEM_PROMPT = (
    "You are the Strategist Agent for strategic planning and blueprint generation.\n"
    "Your role is to:\n"
    "- Analyze SWOT analysis from diagnostic chapters\n"
    "- Apply TOWS matrix analysis to generate strategic options\n"
    "- Formulate mission statements and strategic pillars\n"
    "- Define SMART KPIs across balanced scorecard dimensions\n"
    "Be strategic, analytical, and focus on creating actionable strategic blueprints."
)


def _safe_llm_call(llm: LLMClient, prompt: str, operation: str, fallback: str = "") -> str:
    """Safe LLM call with error handling."""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"{operation} failed: {e}")
        return fallback


def run_group2(run_id: int) -> str:
    """Run Group 2: Multi-agent workflow WITH strategist, WITHOUT memory.

    - Three specialized LLM instances (researcher, analyst, writer)
    - PLUS strategist LLM for blueprint generation
    - Same node sequence: retrieval -> analysis -> writing
    - NO rolling context (context_summary always empty) - KEY ABLATION
    - WITH strategic blueprint (generated after Chapter 3) - NOT ablated
    - Auto-approve all chapters

    Args:
        run_id: The run identifier for output file naming.

    Returns:
        The file path of the generated report.
    """
    logger.info(f"=== Group 2: Multi-Agent, No Memory + Strategist (run_id={run_id}) ===")

    # ----------------------------------------------------------------
    # 1. Create specialized LLM clients for each agent
    # ----------------------------------------------------------------
    researcher_llm = LLMClient(
        temperature=AGENT_TEMPERATURES["researcher"],
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
    )
    analyst_llm = LLMClient(
        temperature=AGENT_TEMPERATURES["analyst"],
        system_prompt=ANALYST_SYSTEM_PROMPT,
    )
    writer_llm = LLMClient(
        temperature=AGENT_TEMPERATURES["writer"],
        system_prompt=WRITER_SYSTEM_PROMPT,
    )
    strategist_llm = LLMClient(
        temperature=0.5,  # Strategist needs balanced temperature
        system_prompt=STRATEGIST_SYSTEM_PROMPT,
    )

    # ----------------------------------------------------------------
    # 2. Create retriever
    # ----------------------------------------------------------------
    retriever = create_retriever()

    # ----------------------------------------------------------------
    # 3. State variables
    # ----------------------------------------------------------------
    chapters_written: list[str] = []
    strategic_blueprint: Optional[Dict[str, Any]] = None
    # NOTE: context_pool stores full chapter texts for strategist
    # This is NOT the same as context_summary (compressed rolling context)
    context_pool: list[str] = []

    # ----------------------------------------------------------------
    # 4. Chapter loop
    # ----------------------------------------------------------------
    for chapter_index, chapter_spec in enumerate(CHAPTER_PLAN):
        chapter_title = chapter_spec.get("title", "Untitled Chapter")
        phase = chapter_spec.get("phase", "")
        analysis_model = chapter_spec.get("analysis_model", "")

        logger.info(f"Processing chapter {chapter_index + 1}: {chapter_title} (phase: {phase})")

        # --- 4a. Extract the question for this chapter ---
        try:
            chapter_question = extract_chapter_question(chapter_title)
        except Exception as exc:
            logger.warning(f"Failed to extract chapter question: {exc}")
            chapter_question = TEST_QUERY

        # --- 4b. Researcher: generate search queries ---
        try:
            query_prompt = generate_query_prompt(chapter_question, "")
            raw_queries = researcher_llm.invoke(query_prompt)
            queries = parse_queries_response(raw_queries)
            if not queries:
                queries = [chapter_question]
            logger.info(f"Researcher generated {len(queries)} queries")
        except Exception as exc:
            logger.warning(f"Query generation failed, using fallback: {exc}")
            queries = [chapter_question]

        # --- 4c. Multi-query retrieval ---
        try:
            search_results = multi_query_search(retriever, queries)
            logger.info(f"Retrieved {len(search_results)} documents")
        except Exception as exc:
            logger.warning(f"Retrieval failed: {exc}")
            search_results = []

        # --- 4d. Document summary & filename mapping ---
        try:
            doc_summary = generate_document_summary(search_results)
        except Exception as exc:
            logger.warning(f"Document summary generation failed: {exc}")
            doc_summary = "No documents retrieved."

        try:
            filename_mapping = extract_filename_mapping(doc_summary)
        except Exception as exc:
            logger.warning(f"Filename mapping failed: {exc}")
            filename_mapping = {}

        # --- 4e. Analyst: analyze retrieved context with model injection ---
        try:
            analysis_prompt = generate_analysis_prompt(
                chapter_question=chapter_question,
                chapter_context=f"基于用户请求: {TEST_QUERY}",
                document_summary=doc_summary,
                analysis_model=analysis_model,
                phase=phase,
            )
            raw_analysis = analyst_llm.invoke(analysis_prompt)
            analysis_result = parse_json_response(raw_analysis)
            key_facts = analysis_result.get("key_facts", [])
            insights = analysis_result.get("insights", [])
            logger.info(f"Analyst extracted facts and insights using {analysis_model}")
        except Exception as exc:
            logger.warning(f"Analysis failed, using fallback: {exc}")
            key_facts = []
            insights = []

        # --- 4f. Writer: compose the chapter ---
        # KEY ABLATION: context_summary is ALWAYS empty (no memory between chapters)
        # But strategic_blueprint is injected for initiatives phase (after Ch3)
        try:
            writing_prompt = generate_writing_prompt(
                chapter_title=chapter_title,
                chapter_question=chapter_question,
                chapter_context=f"基于用户请求: {TEST_QUERY}",
                key_facts=key_facts,
                insights=insights,
                context_summary="",  # <-- KEY ABLATION: NO MEMORY
                document_summary=doc_summary,
                filename_mapping=filename_mapping,
                analysis_model=analysis_model,
                phase=phase,
                strategic_blueprint=strategic_blueprint,  # Blueprint for Ch4-8
            )
            chapter_text = writer_llm.invoke(writing_prompt, temperature=0.7)
        except Exception as exc:
            logger.warning(f"Writing failed for chapter '{chapter_title}': {exc}")
            chapter_text = f"# {chapter_title}\n\n*Chapter generation failed.*"

        chapters_written.append(chapter_text)
        context_pool.append(chapter_text)  # Store full chapter for strategist
        logger.info(f"Finished chapter: {chapter_title}")

        # --- 4g. Strategist: Generate blueprint after Chapter 3 ---
        # This happens AFTER Chapter 3 is written and stored in context_pool
        if chapter_index == 2:  # After Chapter 3 (index 2)
            logger.info("Generating strategic blueprint from diagnosis chapters...")

            try:
                # Step 1: Compress diagnosis chapters (Chapters 1-3)
                diagnosis_text = "\n\n".join(context_pool[:3])
                compression_prompt = generate_diagnosis_compression_prompt(
                    user_input=TEST_QUERY,
                    chapters_text=diagnosis_text,
                )
                diagnosis_summary = strategist_llm.invoke(compression_prompt, temperature=0.3)

                # Step 2: Extract SWOT from diagnosis summary
                swot_prompt = generate_swot_extraction_prompt(diagnosis_summary)
                swot_response = strategist_llm.invoke(swot_prompt, temperature=0.3)
                swot_data = parse_json_response(swot_response)

                # Step 3: Generate full blueprint using TOWS analysis
                blueprint_prompt = generate_blueprint_prompt(swot_data, TEST_QUERY)
                blueprint_response = strategist_llm.invoke(blueprint_prompt, temperature=0.5)
                strategic_blueprint = parse_json_response(blueprint_response)

                # Mark as approved (auto-approve in ablation)
                strategic_blueprint["approved"] = True

                logger.info(f"Strategic blueprint generated with mission: {strategic_blueprint.get('mission', '')[:50]}...")
                logger.info(f"Pillars: {len(strategic_blueprint.get('strategic_pillars', []))}")

            except Exception as exc:
                logger.warning(f"Blueprint generation failed: {exc}. Using fallback.")
                strategic_blueprint = {
                    "mission": "服务交通强省战略，推动高质量发展",
                    "strategic_pillars": [
                        "主业提质：夯实交通投资建设主阵地",
                        "创新驱动：培育智慧绿色交通新动能",
                    ],
                    "kpis": {},
                    "approved": True,
                }

    # ----------------------------------------------------------------
    # 5. Assemble the final report WITH blueprint appendix
    # ----------------------------------------------------------------
    report = assemble_report(
        chapters=chapters_written,
        user_input=TEST_QUERY,
        strategic_blueprint=strategic_blueprint,  # Include blueprint appendix
    )

    # ----------------------------------------------------------------
    # 6. Save report
    # ----------------------------------------------------------------
    output_dir = REPORTS_DIR / "group2"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"run_{run_id}.md"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    logger.info(f"Group 2 report saved to: {output_path}")

    # Aggregate token stats from all agents
    all_stats = {}
    total_prompt = 0
    total_completion = 0
    total_total = 0
    total_calls = 0
    for name, agent_llm in [
        ("researcher", researcher_llm),
        ("analyst", analyst_llm),
        ("writer", writer_llm),
        ("strategist", strategist_llm),
    ]:
        s = agent_llm.get_token_stats()
        all_stats[name] = s
        total_prompt += s["prompt_tokens"]
        total_completion += s["completion_tokens"]
        total_total += s["total_tokens"]
        total_calls += s["call_count"]

    combined_stats = {
        "call_count": total_calls,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_total,
        "per_agent": all_stats,
    }
    print(f"  Token usage: {total_total:,} total "
          f"(prompt: {total_prompt:,}, completion: {total_completion:,}, calls: {total_calls})")
    for name, s in all_stats.items():
        print(f"    {name}: {s['total_tokens']:,} tokens ({s['call_count']} calls)")

    return report, combined_stats


if __name__ == "__main__":
    # Standalone test run
    logging.basicConfig(level=logging.INFO)
    run_group2(0)