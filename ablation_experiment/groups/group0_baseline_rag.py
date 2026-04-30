"""
Group 0: Traditional RAG Baseline (No Agent Architecture)

This group represents the simplest possible RAG pipeline:
- Single LLM instance does everything (no multi-agent workflow)
- No memory management (chapters are independent, context_summary is always "")
- No strategic blueprint (strategic_blueprint is always None)
- Same prompts as other groups (from shared/prompts.py)

Purpose: Establish a lower-bound baseline to measure the improvement
gained by the multi-agent architecture with memory and planning.
"""

import traceback

from ablation_experiment.config import TEST_QUERY, CHAPTER_PLAN, REPORTS_DIR, GEN_MODEL
from ablation_experiment.shared.llm_client import LLMClient, parse_json_response, parse_queries_response
from ablation_experiment.shared.prompts import (
    generate_query_prompt,
    generate_analysis_prompt,
    generate_writing_prompt,
)
from ablation_experiment.shared.rag_retrieval import create_retriever, multi_query_search
from ablation_experiment.shared.report_utils import (
    generate_document_summary,
    extract_filename_mapping,
    extract_chapter_question,
    assemble_report,
)


def run_group0(run_id: int) -> str:
    """Run Group 0: Traditional RAG baseline.

    - Single LLM instance (temperature=0.5)
    - For each of 8 chapters:
      1. RAG retrieval using same query generation prompts
      2. Generate document summary (same as analyst node)
      3. Analysis using analyst's prompt (same prompts)
      4. Writing using writer's prompt BUT with context_summary="" and blueprint=None
    - No context passing between chapters
    - No strategic blueprint
    - Save report to reports/group0/run_{run_id}.md

    Args:
        run_id: The run identifier (used for output file naming).

    Returns:
        The assembled report text.
    """
    print(f"\n{'='*60}")
    print(f"Group 0: Traditional RAG Baseline | Run {run_id}")
    print(f"{'='*60}")

    # Single LLM instance for the entire pipeline
    llm = LLMClient(model=GEN_MODEL, temperature=0.5)

    # Create the retriever once
    retriever = create_retriever()

    # No strategic blueprint and no context memory between chapters
    strategic_blueprint = None
    context_summary = ""  # Always empty -- no memory

    chapters = []

    for idx, chapter_spec in enumerate(CHAPTER_PLAN):
        chapter_title = chapter_spec.get("title", f"Chapter {idx+1}")
        chapter_num = idx + 1
        phase = chapter_spec.get("phase", "diagnosis")
        analysis_model = chapter_spec.get("analysis_model", "")
        print(f"\n--- Chapter {chapter_num}/{len(CHAPTER_PLAN)}: {chapter_title} ---")

        try:
            # Step 1: Extract the research question from the chapter title
            chapter_question = extract_chapter_question(chapter_title)
            print(f"  Question: {chapter_question}")

            # Step 2: Generate search queries using the shared prompt
            query_prompt = generate_query_prompt(
                chapter_question=chapter_question,
                chapter_context=TEST_QUERY,
            )
            query_response = llm.invoke(query_prompt)
            queries = parse_queries_response(query_response)
            print(f"  Generated {len(queries)} queries: {queries}")

            # Step 3: Multi-query RAG retrieval
            search_results = multi_query_search(retriever, queries)
            print(f"  Retrieved {len(search_results)} chunks")

            if not search_results:
                print(f"  WARNING: No search results for chapter '{chapter_title}'")
                chapters.append(f"# {chapter_title}\n\n（本章因检索结果不足，无法生成内容）")
                continue

            # Step 4: Generate document summary (same as analyst node)
            doc_summary = generate_document_summary(search_results)
            print(f"  Document summary: {len(doc_summary)} chars")

            # Step 5: Extract filename mapping for citations
            filename_mapping = extract_filename_mapping(doc_summary)

            # Step 6: Analysis using analyst's prompt (same prompts as other groups)
            analysis_prompt = generate_analysis_prompt(
                chapter_question=chapter_question,
                chapter_context=TEST_QUERY,
                document_summary=doc_summary,
                analysis_model=analysis_model,
                phase=phase,
            )
            analysis_response = llm.invoke(analysis_prompt)
            parsed_analysis = parse_json_response(analysis_response)

            key_facts = parsed_analysis.get("key_facts", [])
            insights = parsed_analysis.get("insights", [])
            print(f"  Analysis: {len(key_facts) if isinstance(key_facts, list) else 'structured'} key facts, {len(insights)} insights")

            # Step 7: Writing using writer's prompt
            # Key difference: context_summary="" and strategic_blueprint=None
            # This means the writer has NO memory of previous chapters and NO plan
            writing_prompt = generate_writing_prompt(
                chapter_title=chapter_title,
                chapter_question=chapter_question,
                chapter_context=TEST_QUERY,
                key_facts=key_facts,
                insights=insights,
                context_summary=context_summary,  # Always empty
                document_summary=doc_summary,
                filename_mapping=filename_mapping,
                analysis_model=analysis_model,
                phase=phase,
                strategic_blueprint=strategic_blueprint,  # Always None
            )
            chapter_content = llm.invoke(writing_prompt)
            print(f"  Written content: {len(chapter_content)} chars")

            chapters.append(chapter_content)

        except Exception as e:
            print(f"  ERROR in chapter '{chapter_title}': {e}")
            traceback.print_exc()
            chapters.append(f"# {chapter_title}\n\n（本章生成过程中出现错误：{e}）")

    # Assemble the final report (no blueprint)
    report_text = assemble_report(
        chapters=chapters,
        user_input=TEST_QUERY,
        strategic_blueprint=None,
    )
    print(f"\nTotal report length: {len(report_text)} chars")

    # Save to file
    output_dir = REPORTS_DIR / "group0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{run_id}.md"
    output_path.write_text(report_text, encoding="utf-8")

    # Print token stats
    stats = llm.get_token_stats()
    print(f"  Token usage: {stats['total_tokens']:,} total "
          f"(prompt: {stats['prompt_tokens']:,}, completion: {stats['completion_tokens']:,}, "
          f"calls: {stats['call_count']})")
    print(f"Report saved to: {output_path}")

    return report_text, stats


if __name__ == "__main__":
    # Allow running this group standalone for debugging
    report = run_group0(run_id=0)
    print("\n" + "=" * 60)
    print("Group 0 standalone run complete.")
    print("=" * 60)
