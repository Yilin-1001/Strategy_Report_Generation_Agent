"""
Group 1: Single Agent + Memory (Ablation Experiment)

Architecture:
- Single LLM node handles all tasks (retrieval, analysis, writing, blueprint generation)
- Rolling context compression between chapters (memory management)
- Strategic blueprint generated via prompting (same prompts as strategist node)
- Same prompts as all other groups for fair comparison

This group isolates the effect of multi-agent orchestration by collapsing
all agent roles into a single LLM while retaining memory and strategy capabilities.
"""

import time
import traceback

from ablation_experiment.config import (
    TEST_QUERY,
    CHAPTER_PLAN,
    REPORTS_DIR,
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
    generate_context_compression_prompt,
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
    build_chapter_knowledge,
    assemble_report,
)


def _safe_llm_call(llm, prompt, label, fallback="", max_retries=1):
    """Execute an LLM call with error handling and optional retry.

    Args:
        llm: LLMClient instance.
        prompt: The prompt string to send.
        label: Human-readable label for logging.
        fallback: Value to return on failure.
        max_retries: Number of retry attempts on failure.

    Returns:
        LLM response text, or fallback on failure.
    """
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"  [WARNING] LLM call failed for '{label}' (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
    print(f"  [ERROR] All attempts exhausted for '{label}', using fallback.")
    return fallback


def run_group1(run_id: int) -> str:
    """Run Group 1: Single agent with memory management.

    - Single LLM instance (temperature=0.5)
    - For each of 8 chapters:
      1. RAG retrieval
      2. Document summary
      3. Analysis using analyst's prompt
      4. Writing using writer's prompt WITH context_summary
      5. Rolling context compression after each chapter
    - After Chapter 3: Generate strategic blueprint via prompting (same strategist prompts)
      - Extract SWOT from Chapter 3 text
      - Generate TOWS strategies, mission, pillars, KPIs
      - Inject blueprint into writer prompts for Chapters 4-8

    Args:
        run_id: Unique identifier for this experimental run.

    Returns:
        The assembled report text.
    """
    print(f"\n{'='*60}")
    print(f"  GROUP 1: Single Agent + Memory | Run {run_id}")
    print(f"{'='*60}")

    start_time = time.time()

    # ----------------------------------------------------------------
    # 1. Initialise single LLM client (the "agent")
    # ----------------------------------------------------------------
    llm = LLMClient(temperature=0.5)
    print("[INIT] Single LLM client created (temperature=0.5)")

    # ----------------------------------------------------------------
    # 2. Initialise state
    # ----------------------------------------------------------------
    context_summary = ""
    strategic_blueprint = None
    chapter_results = []  # list of dicts with title, content, sources, analysis

    # ----------------------------------------------------------------
    # 3. Create RAG retriever
    # ----------------------------------------------------------------
    try:
        retriever = create_retriever()
        print("[INIT] RAG retriever created")
    except Exception as e:
        print(f"[ERROR] Failed to create retriever: {e}")
        retriever = None

    # ----------------------------------------------------------------
    # 4. Process each chapter
    # ----------------------------------------------------------------
    for chapter_idx, chapter_spec in enumerate(CHAPTER_PLAN):
        chapter_num = chapter_idx + 1
        chapter_title = chapter_spec.get("title", f"Chapter {chapter_num}")
        phase = chapter_spec.get("phase", "diagnosis")
        analysis_model = chapter_spec.get("analysis_model", "")
        print(f"\n--- Chapter {chapter_num}/{len(CHAPTER_PLAN)}: {chapter_title} ---")

        # 4a. Extract the research question for this chapter
        try:
            chapter_question = extract_chapter_question(chapter_title)
        except Exception:
            chapter_question = f"Analyse the topic: {chapter_title}"
        print(f"  Question: {chapter_question}")

        # 4b. Generate search queries
        try:
            query_prompt = generate_query_prompt(
                chapter_question=chapter_question,
                chapter_context=TEST_QUERY,
            )
            query_response = _safe_llm_call(
                llm, query_prompt, f"queries_ch{chapter_num}"
            )
            queries = parse_queries_response(query_response)
        except Exception as e:
            print(f"  [WARNING] Query generation failed: {e}")
            queries = [chapter_question]
        print(f"  Generated {len(queries)} queries")

        # 4c. RAG retrieval via multi-query search
        try:
            if retriever:
                search_results = multi_query_search(retriever, queries)
            else:
                search_results = []
        except Exception as e:
            print(f"  [WARNING] Retrieval failed: {e}")
            search_results = []
        print(f"  Retrieved {len(search_results)} document chunks")

        # 4d. Generate document summary
        try:
            doc_summary = generate_document_summary(search_results)
        except Exception:
            doc_summary = "No document summary available."

        # 4e. Build filename mapping
        try:
            filename_map = extract_filename_mapping(doc_summary)
        except Exception:
            filename_map = {}

        # 4f. Analysis phase
        try:
            analysis_prompt = generate_analysis_prompt(
                chapter_question=chapter_question,
                chapter_context=TEST_QUERY,
                document_summary=doc_summary,
                analysis_model=analysis_model,
                phase=phase,
            )
            analysis_response = _safe_llm_call(
                llm, analysis_prompt, f"analysis_ch{chapter_num}"
            )
            analysis_data = parse_json_response(analysis_response)
        except Exception as e:
            print(f"  [WARNING] Analysis failed: {e}")
            analysis_data = {
                "key_facts": [],
                "insights": ["Analysis unavailable due to error."],
            }
        key_facts = analysis_data.get("key_facts", [])
        insights = analysis_data.get("insights", [])
        print(f"  Analysis complete")

        # 4g. Writing phase
        try:
            writing_prompt = generate_writing_prompt(
                chapter_title=chapter_title,
                chapter_question=chapter_question,
                chapter_context=TEST_QUERY,
                key_facts=key_facts,
                insights=insights,
                context_summary=context_summary,
                document_summary=doc_summary,
                filename_mapping=filename_map,
                analysis_model=analysis_model,
                phase=phase,
                strategic_blueprint=strategic_blueprint,
            )
            chapter_content = _safe_llm_call(
                llm, writing_prompt, f"writing_ch{chapter_num}"
            )
            if not chapter_content:
                chapter_content = f"# {chapter_title}\n\nContent generation failed for this chapter."
        except Exception as e:
            print(f"  [WARNING] Writing failed: {e}")
            chapter_content = f"# {chapter_title}\n\nContent generation failed due to error: {e}"
        print(f"  Writing complete ({len(chapter_content)} chars)")

        # 4h. Store chapter results
        chapter_results.append({
            "title": chapter_title,
            "content": chapter_content,
        })

        # 4i. Build chapter knowledge for context compression
        try:
            chapter_knowledge = build_chapter_knowledge(
                chapter_title=chapter_title,
                key_facts=key_facts,
                insights=insights if isinstance(insights, list) else [],
            )
        except Exception:
            chapter_knowledge = f"Chapter {chapter_num} ({chapter_title}) was completed."

        # 4j. Rolling context compression after each chapter
        try:
            compression_prompt = generate_context_compression_prompt(
                existing_summary=context_summary,
                new_knowledge=chapter_knowledge,
            )
            context_summary = _safe_llm_call(
                llm,
                compression_prompt,
                f"context_compression_ch{chapter_num}",
                fallback=context_summary,
            )
        except Exception as e:
            print(f"  [WARNING] Context compression failed: {e}")
        print(f"  Context compressed ({len(context_summary)} chars)")

        # ------------------------------------------------------------
        # 5. After Chapter 3: Generate strategic blueprint
        # ------------------------------------------------------------
        if chapter_idx == 2:  # After Chapter 3 (0-indexed)
            print("\n  >>> Generating Strategic Blueprint <<<")

            try:
                # 5a. Collect text from chapters 1-3 for SWOT extraction
                chapters_text = "\n\n".join(
                    f"## {cr['title']}\n\n{cr['content']}"
                    for cr in chapter_results
                )

                # 5b. Extract SWOT via prompting
                swot_prompt = generate_swot_extraction_prompt(chapters_text)
                swot_response = _safe_llm_call(
                    llm, swot_prompt, "swot_extraction"
                )
                swot_data = parse_json_response(swot_response)

                # 5c. Generate full blueprint via prompting
                blueprint_prompt = generate_blueprint_prompt(
                    swot_data=swot_data,
                    user_input=TEST_QUERY,
                )
                blueprint_response = _safe_llm_call(
                    llm, blueprint_prompt, "blueprint_generation"
                )
                strategic_blueprint = parse_json_response(blueprint_response)

                # 5d. Mark as approved (bypasses human review in ablation)
                if strategic_blueprint is None:
                    strategic_blueprint = {}
                strategic_blueprint["approved"] = True

                print(f"  Blueprint generated successfully")

            except Exception as e:
                print(f"  [WARNING] Blueprint generation failed: {e}")
                traceback.print_exc()
                # Fallback: create minimal blueprint
                strategic_blueprint = {
                    "approved": True,
                    "mission": "Drive sustainable growth through strategic analysis.",
                    "strategic_pillars": [],
                    "kpis": {},
                    "swot": {
                        "strengths": [],
                        "weaknesses": [],
                        "opportunities": [],
                        "threats": [],
                    },
                }

    # ----------------------------------------------------------------
    # 6. Assemble and save the final report
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Assembling final report...")

    # Extract just the chapter content strings for assembly
    chapter_contents = [cr["content"] for cr in chapter_results]

    try:
        report_content = assemble_report(
            chapters=chapter_contents,
            user_input=TEST_QUERY,
            strategic_blueprint=strategic_blueprint,
        )
    except Exception as e:
        print(f"[ERROR] Report assembly failed: {e}")
        traceback.print_exc()
        # Fallback: concatenate chapter contents
        report_content = f"# Research Report\n\n## Query\n{TEST_QUERY}\n\n"
        for cr in chapter_results:
            report_content += f"{cr['content']}\n\n"

    # Save to file
    report_dir = REPORTS_DIR / "group1"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"run_{run_id}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    elapsed = time.time() - start_time

    # Print token stats
    stats = llm.get_token_stats()
    print(f"  Token usage: {stats['total_tokens']:,} total "
          f"(prompt: {stats['prompt_tokens']:,}, completion: {stats['completion_tokens']:,}, "
          f"calls: {stats['call_count']})")
    print(f"  Report saved to: {report_path}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return report_content, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_group1(run_id=0)
    print(f"\nReport preview (first 500 chars):\n{report[:500]}")
