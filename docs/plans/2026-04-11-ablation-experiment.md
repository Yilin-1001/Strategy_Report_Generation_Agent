# Ablation Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an isolated ablation experiment that tests the contribution of Memory Context Management and Multi-Agent Workflow modules independently, using 4 experiment groups with 3 runs each, evaluated by Qwen model.

**Architecture:** Create a new `ablation_experiment/` folder that imports shared infrastructure (RAG retrieval, embeddings, storage) from `rag_project/` but re-implements the workflow and memory logic for each ablation group. Each group uses identical prompts extracted from the original agent nodes.

**Tech Stack:** Python 3, LangGraph (for Groups 2-3), OpenAI SDK (DeepSeek API), SiliconFlow API (Qwen evaluator), Milvus vector DB (via rag_project pipeline)

---

## Experiment Design Summary

| Module | Group 0 | Group 1 | Group 2 | Group 3 |
|--------|---------|---------|---------|---------|
| Name | Baseline RAG | Single Agent+Memory | Multi-Agent No-Memory | Full System |
| RAG Retrieval | Yes | Yes | Yes | Yes |
| Multi-Agent Workflow | No | No | Yes | Yes |
| Memory (Rolling Context) | No | Yes | No | Yes |
| Strategic Blueprint | No | Yes (via prompt) | No | Yes (via node) |
| Identical Prompts | Yes | Yes | Yes | Yes |

---

## Folder Structure

```
ablation_experiment/
├── __init__.py
├── config.py                          # Shared config, API keys, paths
├── shared/
│   ├── __init__.py
│   ├── prompts.py                     # All prompts extracted from rag_project nodes
│   ├── chapter_plan.py                # Fixed 8-chapter plan (from coordinator)
│   ├── rag_retrieval.py               # RAG retrieval wrapper (reuses rag_project)
│   └── report_utils.py               # Report formatting utilities
├── groups/
│   ├── __init__.py
│   ├── group0_baseline_rag.py         # Traditional RAG, no workflow, no memory
│   ├── group1_single_agent_memory.py  # Single LLM node + rolling context + blueprint via prompt
│   ├── group2_multi_agent_no_memory.py # Full multi-agent, no context, no blueprint
│   └── group3_full_system.py          # Complete system (wraps rag_project)
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py                   # Qwen-based scoring with 5-dimension rubric
├── runner.py                          # Main experiment orchestrator
├── reports/                           # Generated reports by group
│   ├── group0/
│   ├── group1/
│   ├── group2/
│   └── group3/
└── results/                           # Evaluation CSV and summary
```

---

### Task 1: Create Folder Structure and Shared Config

**Files:**
- Create: `ablation_experiment/__init__.py`
- Create: `ablation_experiment/config.py`
- Create: `ablation_experiment/shared/__init__.py`
- Create: `ablation_experiment/groups/__init__.py`
- Create: `ablation_experiment/evaluation/__init__.py`
- Create directories: `reports/group0-3/`, `results/`

**Step 1: Create directory structure**

```bash
mkdir -p ablation_experiment/{shared,groups,evaluation,reports/{group0,group1,group2,group3},results}
touch ablation_experiment/__init__.py
touch ablation_experiment/shared/__init__.py
touch ablation_experiment/groups/__init__.py
touch ablation_experiment/evaluation/__init__.py
```

**Step 2: Write config.py**

`config.py` should contain:
- API key loading from env vars (DEEPSEEK_API_KEY, SILICONFLOW_API_KEY)
- LLM model config (deepseek-chat for generation, Qwen for evaluation)
- The fixed 8-chapter plan (identical to coordinator.py's global_plan)
- File paths for reports output
- Number of runs per group (3)
- The test query string

```python
import os

# Test query
TEST_QUERY = "生成江西交通投资集团的战略规划报告"

# Generation LLM (DeepSeek)
GEN_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEN_BASE_URL = "https://api.deepseek.com"
GEN_MODEL = "deepseek-chat"

# Evaluation LLM (Qwen via SiliconFlow)
EVAL_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
EVAL_BASE_URL = "https://api.siliconflow.cn/v1"
EVAL_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Experiment settings
NUM_RUNS = 3
REPORTS_DIR = "ablation_experiment/reports"
RESULTS_DIR = "ablation_experiment/results"

# Fixed 8-chapter plan (from coordinator.py)
CHAPTER_PLAN = [
    {"title": "第一章：宏观政策环境与时代要求", "phase": "diagnosis", "analysis_model": "PEST模型 (侧重P-政策与E-经济维度)", "index": 0},
    {"title": "第二章：区域战略与'交通强省'建设剖析", "phase": "diagnosis", "analysis_model": "无特定模型，侧重省级政策承接与区域占位分析", "index": 1},
    {"title": "第三章：行业演进趋势与当前内部诊断", "phase": "diagnosis", "analysis_model": "波特五力模型与SWOT分析 (强制要求在分析结尾输出结构化的SWOT矩阵)", "index": 2},
    {"title": "第四章：总体战略思路与政策响应目标", "phase": "initiatives", "analysis_model": "平衡计分卡(BSC)模型 (从财务、客户/民生、内部运营、学习与成长四个维度设定目标)", "index": 3},
    {"title": "第五章：主责主业：高质量建设与保通保畅举措", "phase": "initiatives", "analysis_model": "BCG波士顿矩阵 (将主业作为'现金牛'业务，侧重精益化与稳健回报)", "index": 4},
    {"title": "第六章：创新驱动：绿色低碳与智慧交投建设", "phase": "initiatives", "analysis_model": "安索夫矩阵 (将创新业务作为新产品/新市场拓展，侧重第二增长曲线)", "index": 5},
    {"title": "第七章：产业协同：交旅融合与服务地方经济", "phase": "initiatives", "analysis_model": "产业链协同与ESG社会责任模型", "index": 6},
    {"title": "第八章：治理效能：深化国企改革与党建引领", "phase": "initiatives", "analysis_model": "麦肯锡7S模型 (从结构、制度、风格、员工、技能等维度构建组织保障)", "index": 7},
]
```

**Step 3: Commit**

```bash
git add ablation_experiment/
git commit -m "feat(ablation): create folder structure and shared config"
```

---

### Task 2: Extract All Prompts into shared/prompts.py

**Files:**
- Create: `ablation_experiment/shared/prompts.py`
- Create: `ablation_experiment/shared/chapter_plan.py`
- Create: `ablation_experiment/shared/rag_retrieval.py`
- Create: `ablation_experiment/shared/report_utils.py`

**Step 1: Write prompts.py**

Extract ALL prompt-generating functions from rag_project agent nodes into a single shared module. These functions are used identically across all 4 experiment groups.

Key functions to extract (copy verbatim logic from rag_project source):
1. `generate_query_prompt(question, context)` - from researcher.py:135-147
2. `generate_analysis_prompt(question, context, doc_summary, analysis_model, phase)` - from analyst.py:404-509
3. `get_model_instruction(analysis_model)` - from analyst.py:512-736
4. `generate_writing_prompt(chapter_title, question, context, key_facts, insights, context_summary, doc_summary, analysis_model, phase, blueprint, revision_feedback)` - from writer.py:411-594
5. `get_model_writing_instruction(analysis_model, phase)` - from writer.py:673-742
6. `build_blueprint_constraint(blueprint)` - from writer.py:628-670
7. `build_facts_section(key_facts, analysis_model)` - from writer.py:597-625
8. `generate_blueprint_prompt(swot_data, user_input)` - from strategist.py:315-400
9. `generate_swot_extraction_prompt(chapter_text)` - from strategist.py:189-213
10. `compress_context_prompt(existing_summary, new_knowledge)` - from prep_chapter.py:69-85

**Step 2: Write chapter_plan.py**

Extract the fixed 8-chapter plan and chapter_question generation logic from coordinator.py and prep_chapter.py.

**Step 3: Write rag_retrieval.py**

Wrapper that imports and uses rag_project's RAGRetriever for document search. Provides `search_documents(query, top_k=20)` function.

```python
from rag_project.agent.retriever import RAGRetriever

def create_retriever():
    return RAGRetriever(agent_config_path="config/agent_config.yaml")

def search_documents(retriever, query, top_k=20):
    return retriever.search(query, top_k=top_k)
```

Also extract document deduplication and summary generation functions from researcher.py and analyst.py.

**Step 4: Write report_utils.py**

Extract report formatting utilities:
- `generate_document_summary_v2()` from analyst.py:213-285
- `smart_extract()` from analyst.py:187-210
- `deduplicate_documents()` from researcher.py:214-242
- `create_report()` - assemble chapters into final report with cover, TOC
- `save_report()` - write to file

**Step 5: Commit**

```bash
git add ablation_experiment/shared/
git commit -m "feat(ablation): extract shared prompts and utilities from rag_project"
```

---

### Task 3: Implement Group 0 - Baseline RAG

**Files:**
- Create: `ablation_experiment/groups/group0_baseline_rag.py`

**Design:**
- Single LLM instance (DeepSeek, temperature=0.5 middle ground)
- For each of 8 chapters:
  1. RAG retrieval (same multi-query logic as researcher, using same prompts)
  2. Generate document summary (same as analyst)
  3. **Skip** separate analysis step - pass documents directly to writing prompt
  4. Single LLM call using the **writer's prompt template** (same as writer.py's _generate_writing_prompt) but WITHOUT context_summary (always empty) and WITHOUT blueprint constraint (always None)
  5. Append draft to report
- No context passing between chapters
- No strategic blueprint generation

**Step 1: Implement group0_baseline_rag.py**

```python
def run_group0(run_id: int) -> str:
    """Run Group 0 experiment: Traditional RAG baseline.
    
    Returns the generated report text.
    """
    retriever = create_retriever()
    llm = LLMClient("deepseek-chat")  # Single LLM
    
    chapters = []
    for chapter_meta in CHAPTER_PLAN:
        # Step 1: RAG retrieval (same query generation prompts)
        queries = generate_queries(chapter_meta, llm)
        docs = multi_query_search(retriever, queries)
        doc_summary = generate_document_summary(docs)
        
        # Step 2: Write chapter using writer's prompt template
        # Key: context_summary="" and strategic_blueprint=None
        prompt = generate_writing_prompt(
            chapter_title=chapter_meta["title"],
            chapter_question=extract_question(chapter_meta["title"]),
            chapter_context=f"基于用户请求: {TEST_QUERY}",
            key_facts=[],  # No separate analysis step
            insights=[],
            context_summary="",  # NO MEMORY
            document_summary=doc_summary,
            analysis_model=chapter_meta["analysis_model"],
            phase=chapter_meta["phase"],
            strategic_blueprint=None,  # NO BLUEPRINT
        )
        draft = llm.invoke(prompt, temperature=0.7)
        chapters.append(draft)
    
    report = assemble_report(chapters, TEST_QUERY)
    return report
```

**Step 2: Commit**

```bash
git add ablation_experiment/groups/group0_baseline_rag.py
git commit -m "feat(ablation): implement Group 0 - baseline RAG"
```

---

### Task 4: Implement Group 1 - Single Agent + Memory

**Files:**
- Create: `ablation_experiment/groups/group1_single_agent_memory.py`

**Design:**
- Single LLM instance (same DeepSeek model, temperature=0.5)
- For each of 8 chapters:
  1. RAG retrieval (same multi-query logic)
  2. Document summary generation
  3. **Analysis step** using analyst's prompt template (same prompts)
  4. **Writing step** using writer's prompt template WITH context_summary
  5. **Rolling context compression** using prep_chapter's compression prompt
- After Chapter 3: Generate strategic blueprint using **same strategist prompts** (but all in the same LLM, no separate node)
- Blueprint is then injected into writer prompts for Chapters 4-8

**Step 1: Implement group1_single_agent_memory.py**

```python
def run_group1(run_id: int) -> str:
    """Run Group 1: Single agent with memory management.
    
    One LLM does everything, but WITH rolling context and strategic blueprint.
    """
    retriever = create_retriever()
    llm = LLMClient("deepseek-chat")  # SINGLE LLM for everything
    
    context_summary = ""
    context_pool = []
    strategic_blueprint = None
    
    for i, chapter_meta in enumerate(CHAPTER_PLAN):
        # Step 1: RAG retrieval (same prompts)
        queries = generate_queries(chapter_meta, llm)
        docs = multi_query_search(retriever, queries)
        doc_summary = generate_document_summary(docs)
        
        # Step 2: Analysis (using analyst's prompt template)
        key_facts, insights = analyze_documents(
            question=extract_question(chapter_meta["title"]),
            doc_summary=doc_summary,
            analysis_model=chapter_meta["analysis_model"],
            phase=chapter_meta["phase"],
            llm=llm
        )
        
        # Step 3: Write chapter (with context_summary and blueprint)
        prompt = generate_writing_prompt(
            chapter_title=chapter_meta["title"],
            chapter_question=extract_question(chapter_meta["title"]),
            chapter_context=f"基于用户请求: {TEST_QUERY}",
            key_facts=key_facts,
            insights=insights,
            context_summary=context_summary,  # WITH MEMORY
            document_summary=doc_summary,
            analysis_model=chapter_meta["analysis_model"],
            phase=chapter_meta["phase"],
            strategic_blueprint=strategic_blueprint,  # WITH BLUEPRINT (after Ch3)
        )
        draft = llm.invoke(prompt, temperature=0.7)
        chapters.append(draft)
        context_pool.append(draft)
        
        # Step 4: Rolling context compression (same prompt as prep_chapter.py)
        context_summary = compress_context(
            context_summary, chapter_meta["title"], key_facts, insights, llm
        )
        
        # Step 5: After Chapter 3, generate blueprint via prompting (NOT separate node)
        if i == 2:
            strategic_blueprint = generate_blueprint_via_prompt(
                context_pool, TEST_QUERY, llm
            )
            strategic_blueprint["approved"] = True
    
    report = assemble_report(chapters, TEST_QUERY, strategic_blueprint)
    return report
```

**Step 2: Commit**

```bash
git add ablation_experiment/groups/group1_single_agent_memory.py
git commit -m "feat(ablation): implement Group 1 - single agent with memory"
```

---

### Task 5: Implement Group 2 - Multi-Agent, No Memory

**Files:**
- Create: `ablation_experiment/groups/group2_multi_agent_no_memory.py`

**Design:**
- Multiple specialized LLM instances (same as original: coordinator, researcher, analyst, writer temps)
- Run the same node sequence: coordinator → prep_chapter → researcher → analyst → writer → auto-approve
- **REMOVE** rolling context compression (context_summary always stays empty)
- **REMOVE** strategic blueprint generation (skip strategist node entirely)
- **REMOVE** knowledge gap detection
- Auto-approve all chapters (no human review)

**Step 1: Implement group2_multi_agent_no_memory.py**

```python
def run_group2(run_id: int) -> str:
    """Run Group 2: Multi-agent workflow without memory management.
    
    Full multi-agent pipeline but NO rolling context and NO strategic blueprint.
    """
    retriever = create_retriever()
    researcher_llm = LLMClient("deepseek-chat", temperature=0.1)
    analyst_llm = LLMClient("deepseek-chat", temperature=0.5)
    writer_llm = LLMClient("deepseek-chat", temperature=0.7)
    
    context_pool = []
    
    for i, chapter_meta in enumerate(CHAPTER_PLAN):
        # Step 1: Prepare chapter (set title, clear workspace)
        chapter_title = chapter_meta["title"]
        chapter_question = extract_question(chapter_title)
        chapter_context = f"基于用户请求: {TEST_QUERY}"
        
        # Step 2: Researcher node (same query generation + retrieval)
        queries = generate_queries_researcher(chapter_question, chapter_context, researcher_llm)
        docs = multi_query_search(retriever, queries)
        doc_summary = generate_document_summary(docs)
        
        # Step 3: Analyst node (same analysis with model injection)
        key_facts, insights = analyze_documents(
            question=chapter_question,
            doc_summary=doc_summary,
            analysis_model=chapter_meta["analysis_model"],
            phase=chapter_meta["phase"],
            llm=analyst_llm
        )
        
        # Step 4: Writer node (same writing prompt BUT no context_summary, no blueprint)
        prompt = generate_writing_prompt(
            chapter_title=chapter_title,
            chapter_question=chapter_question,
            chapter_context=chapter_context,
            key_facts=key_facts,
            insights=insights,
            context_summary="",  # NO MEMORY - always empty
            document_summary=doc_summary,
            analysis_model=chapter_meta["analysis_model"],
            phase=chapter_meta["phase"],
            strategic_blueprint=None,  # NO BLUEPRINT
        )
        draft = writer_llm.invoke(prompt, temperature=0.7)
        context_pool.append(draft)
    
    # Step 5: Assemble report (simple, no executive summary with blueprint)
    report = assemble_report_simple(context_pool, TEST_QUERY)
    return report
```

**Step 2: Commit**

```bash
git add ablation_experiment/groups/group2_multi_agent_no_memory.py
git commit -m "feat(ablation): implement Group 2 - multi-agent without memory"
```

---

### Task 6: Implement Group 3 - Full System

**Files:**
- Create: `ablation_experiment/groups/group3_full_system.py`

**Design:**
- Import and use `rag_project.agent.graph.create_report_graph()` directly
- Auto-approve all chapters (auto_mode=True)
- Auto-approve strategic blueprint
- This is the complete system with all modules enabled

**Step 1: Implement group3_full_system.py**

```python
def run_group3(run_id: int) -> str:
    """Run Group 3: Full system with multi-agent + memory + blueprint.
    
    Uses the original rag_project agent system in auto-approve mode.
    """
    from rag_project.agent.graph import create_report_graph
    from rag_project.agent.state import GraphState
    
    app = create_report_graph()
    config = {"configurable": {"thread_id": f"ablation_group3_run{run_id}"}}
    
    # Initialize state
    state = {"user_input": TEST_QUERY}
    
    # Run with auto-approve for all chapters and blueprint
    result = run_auto_workflow(app, state, config)
    
    return result["final_report"]
```

The `run_auto_workflow` function needs to:
1. Invoke the graph
2. At each interrupt point (human_review), automatically approve
3. After Chapter 3, auto-approve the strategic blueprint
4. Continue until final_report is generated

**Step 2: Commit**

```bash
git add ablation_experiment/groups/group3_full_system.py
git commit -m "feat(ablation): implement Group 3 - full system wrapper"
```

---

### Task 7: Implement Evaluation Module

**Files:**
- Create: `ablation_experiment/evaluation/evaluator.py`

**Design:**
- Use Qwen model (SiliconFlow API) as evaluator
- Implement the 5-dimension scoring rubric provided by the user
- Each dimension scored 0-20, total 0-100
- Output structured evaluation with specific analysis per dimension

**Step 1: Write evaluator.py**

```python
SCORING_PROMPT = """【角色设定】你现在是一位拥有20年经验的国际顶尖战略咨询合伙人（如MBB级别）。
你需要以极其严苛、专业的商业视角，对输入的《战略规划报告》进行客观评估与打分。

【任务目标】请根据以下五个维度的评分标准，对报告进行深度剖析。你的输出必须包含对每个维度的
具体分析依据（指出报告中的亮点与致命漏洞），然后给出该维度的具体得分（0-20分）。
最后汇总总分，并给出一段改进建议。

[... full rubric from user's requirements ...]
"""

def evaluate_report(report_text: str) -> dict:
    """Evaluate a single report using Qwen model.
    
    Returns dict with scores for each dimension and total.
    """
    # Use SiliconFlow Qwen API
    client = OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)
    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": SCORING_PROMPT},
            {"role": "user", "content": f"请评估以下战略规划报告:\n\n{report_text}"}
        ],
        temperature=0.3,
        max_tokens=4096
    )
    # Parse structured scores from response
    return parse_evaluation(response.content)
```

**Step 2: Commit**

```bash
git add ablation_experiment/evaluation/evaluator.py
git commit -m "feat(ablation): implement Qwen evaluation module with 5-dimension rubric"
```

---

### Task 8: Implement Runner (Main Orchestrator)

**Files:**
- Create: `ablation_experiment/runner.py`

**Design:**
- Run all 4 groups x 3 runs = 12 total experiments
- Save each report to `reports/groupN/run_M.md`
- After all reports generated, evaluate each with Qwen
- Save evaluation results to `results/evaluation_report.csv`
- Generate summary statistics (mean, std per group per dimension)

**Step 1: Write runner.py**

```python
def main():
    results = []
    for group_id, run_fn in enumerate([run_group0, run_group1, run_group2, run_group3]):
        group_name = f"group{group_id}"
        for run_id in range(1, NUM_RUNS + 1):
            print(f"Running {group_name} run {run_id}/{NUM_RUNS}...")
            report = run_fn(run_id)
            
            # Save report
            save_report(report, group_name, run_id)
            
            # Evaluate
            scores = evaluate_report(report)
            results.append({
                "group": group_name,
                "run": run_id,
                **scores
            })
    
    # Save evaluation results
    save_results_csv(results)
    print_summary(results)
```

**Step 2: Commit**

```bash
git add ablation_experiment/runner.py
git commit -m "feat(ablation): implement experiment runner orchestrator"
```

---

### Task 9: Integration Test

**Step 1: Verify all imports work**

```bash
cd "E:/02 Final Year Project/RAG Project"
python -c "from ablation_experiment.config import CHAPTER_PLAN; print(len(CHAPTER_PLAN))"
```

**Step 2: Run a single group test (Group 0, 1 run)**

```bash
python -c "from ablation_experiment.groups.group0_baseline_rag import run_group0; print(run_group0(1)[:200])"
```

**Step 3: Run full experiment**

```bash
python -m ablation_experiment.runner
```

**Step 4: Verify reports and evaluation results**

Check that:
- `reports/group0/run_1.md` through `reports/group3/run_3.md` exist
- `results/evaluation_report.csv` has 12 rows (4 groups x 3 runs)
- Each row has 5 dimension scores and a total

**Step 5: Commit**

```bash
git add ablation_experiment/
git commit -m "feat(ablation): complete ablation experiment with integration test"
```
