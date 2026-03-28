# Agent System Architecture Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Design Principles](#design-principles)
- [Agent Responsibilities](#agent-responsibilities)
- [Routing Logic](#routing-logic)
- [Tech Stack](#tech-stack)
- [Extension Guide](#extension-guide)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                     (CLI / Python API)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     StateGraph                            │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │              GraphState (Global State)              │  │  │
│  │  │  • user_input                                        │  │  │
│  │  │  • global_plan                                       │  │  │
│  │  │  • context_pool (approved chapters)                  │  │  │
│  │  │  • chapter_scratchpad (current chapter workspace)    │  │  │
│  │  │  • current_draft                                      │  │  │
│  │  │  • human_feedback                                    │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  Nodes (Agents):                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │ Coordinator  │──▶│Prepare_Chapt │──▶│ Researcher   │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  │                                              │             │  │
│  │                                              ▼             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │  │
│  │  │   Archiver   │◀─│Human_Review  │◀─│   Writer     │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │  │
│  │                           ▲                                │  │
│  │                           │                                │  │
│  │                    ┌──────────────┐                       │  │
│  │                    │   Analyst    │                       │  │
│  │                    └──────────────┘                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LLM Manager    │  │  RAG Retriever  │  │  Checkpointer   │
│  (DeepSeek API) │  │  (Milvus)       │  │  (MemorySaver)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│  DeepSeek API   │  │  Milvus Vector  │
│  (LLM Service)  │  │  Database       │
└─────────────────┘  └─────────────────┘
```

### Core Components

#### 1. LangGraph Workflow Engine
- **File:** `rag_project/agent/graph.py`
- **Purpose:** Orchestrates multi-agent workflow
- **Key Features:**
  - State management via StateGraph
  - Conditional routing between nodes
  - Interrupt points for human intervention
  - Checkpoint-based persistence

#### 2. State Management
- **File:** `rag_project/agent/state.py`
- **Purpose:** Maintains global workflow state
- **Architecture:** Three-tier memory system
  - **Long-term Memory:** Milvus RAG (persistent knowledge)
  - **Working Memory:** `chapter_scratchpad` (per-chapter sandbox)
  - **Approved Memory:** `context_pool` (curated content)

#### 3. Agent Nodes
- **Directory:** `rag_project/agent/nodes/`
- **Purpose:** Specialized AI agents for specific tasks
- **Types:**
  - `coordinator.py` - Strategic planning
  - `prep_chapter.py` - Context preparation
  - `researcher.py` - Information retrieval
  - `analyst.py` - Critical analysis
  - `writer.py` - Content generation
  - `human_review.py` - HITL interface
  - `archiver.py` - Report compilation

#### 4. LLM Manager
- **File:** `rag_project/agent/llm_manager.py`
- **Purpose:** Manages LLM API interactions
- **Features:**
  - Agent-specific configurations
  - Temperature tuning per agent type
  - OpenAI-compatible API wrapper
  - Error handling and retry logic

#### 5. RAG Retriever
- **File:** `rag_project/agent/retriever.py`
- **Purpose:** Knowledge base queries
- **Features:**
  - Semantic search via embeddings
  - Metadata filtering
  - Multi-query support
  - Result formatting for agents

---

## Design Principles

### 1. State Isolation

**Problem:** Agents need different scopes of information access
- Researcher should see knowledge base but not other chapters
- Writer needs current chapter context but not raw search results
- Analyst needs clean evidence without writer's bias

**Solution:** Three-tier memory architecture

#### Tier 1: Long-term Memory (Milvus RAG)
```python
# Only accessible by Researcher node
class RAGRetriever:
    def search(self, query: str, top_k: int) -> List[Dict]:
        # Queries Milvus for relevant documents
        # Returns raw evidence from knowledge base
```

**Access Pattern:**
- Researcher ONLY
- Read-only
- Contains entire knowledge base
- Persistent across all chapters

#### Tier 2: Working Memory (chapter_scratchpad)
```python
# Created fresh for each chapter
chapter_scratchpad = {
    "queries": ["query1", "query2"],      # From Researcher
    "evidence": ["doc1", "doc2"],         # From Researcher
    "analysis": {                         # From Analyst
        "key_findings": [...],
        "trends": [...]
    }
}
```

**Access Pattern:**
- Researcher: Writes queries and evidence
- Analyst: Reads evidence, writes analysis
- Writer: Reads all, writes draft
- Cleared after each chapter (not added to context_pool)

**Key Design:** Scratchpad is "read-and-burn" - it exists only for current chapter, preventing contamination between chapters.

#### Tier 3: Approved Memory (context_pool)
```python
# Only contains human-approved chapters
context_pool = [
    "# Chapter 1: Industry Background\n\nContent...",
    "# Chapter 2: Policy Environment\n\nContent..."
]
```

**Access Pattern:**
- Written ONLY when human approves chapter
- Read by Prepare_Chapter for context summary
- Accumulates throughout workflow
- Used by Archiver for final report

**State Isolation Flow:**
```python
# Prepare_Chapter creates compressed summary
context_summary = compress(context_pool)

# Researcher CANNOT see context_pool
# Only uses RAG knowledge base

# Analyst reads scratchpad["evidence"]
# Cannot see other chapters in context_pool

# Writer reads scratchpad (evidence + analysis)
# Plus context_summary (brief context from prev chapters)
# NOT full context_pool (prevents copying)

# Only after human approve:
# current_draft → context_pool
# scratchpad → cleared
```

### 2. Chapter-by-Chapter Generation

**Problem:** Generating full report in one LLM call causes:
- Loss of coherence in long documents
- Token limit exceeded
- Hard to review and revise
- No quality control per section

**Solution:** Sequential chapter generation

#### Workflow:
```python
for chapter_index in range(len(global_plan)):
    # 1. Prepare context for THIS chapter only
    chapter_title = global_plan[chapter_index]
    context_summary = compress(context_pool)  # Brief, not detailed

    # 2. Create fresh scratchpad
    chapter_scratchpad = {}  # Empty, no previous chapter data

    # 3. Researcher: Fill scratchpad with evidence
    chapter_scratchpad["evidence"] = rag_search(chapter_title)

    # 4. Analyst: Analyze evidence in scratchpad
    chapter_scratchpad["analysis"] = analyze(chapter_scratchpad["evidence"])

    # 5. Writer: Generate draft from scratchpad
    current_draft = write(chapter_scratchpad, context_summary)

    # 6. Human review: Check quality
    if approve:
        context_pool.append(current_draft)  # Add to approved memory
        chapter_scratchpad = {}  # Clear for next chapter
    else:
        # Revise without advancing
        continue
```

**Benefits:**
1. **Bounded Context:** Each agent works with focused information
2. **Quality Control:** Human can review and refine each chapter
3. **Memory Efficiency:** Scratchpad cleared after each chapter
4. **Iterative Improvement:** Can revise individual chapters

**State Updates:**
```python
# On approve:
context_pool += [full_chapter]  # Accumulate approved chapters
scratchpad = {}  # Clear working memory
chapter_index += 1  # Advance

# On revise:
# No state changes
# Keep same scratchpad
# Re-run specific node (researcher/analyst/writer)
```

### 3. Human-in-the-Loop (HITL)

**Problem:** LLM outputs can have:
- Factual errors
- Logical inconsistencies
- Poor writing quality
- Missing important information

**Solution:** Mandatory human review between chapters

#### Interrupt Mechanism:
```python
# LangGraph interrupt configuration
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"]  # PAUSE here
)
```

#### Review Flow:
```python
# 1. Workflow executes until interrupt
state = app.invoke(initial_state, config)

# Workflow PAUSED at human_review node
# Execution stopped, waiting for input

# 2. Human reviews current_draft
print(state["current_draft"])

# 3. Human provides decision
state["review_decision"] = "approve"  # or revise:*
state["human_feedback"] = {...}

# 4. Resume workflow
state = app.invoke(state, config)  # Continues from interrupt
```

#### Decision Types:

**approve:**
```python
if decision == "approve":
    # Add to approved memory
    context_pool.append(current_draft)
    # Clear working memory
    scratchpad = {}
    # Advance to next chapter
    chapter_index += 1
```

**revise:data:**
```python
if decision == "revise:data":
    # Route back to researcher
    # Keep same scratchpad
    # Add new queries from human_feedback
    scratchpad["queries"].extend(human_feedback["new_queries"])
```

**revise:logic:**
```python
if decision == "revise:logic":
    # Route back to analyst
    # Keep evidence, re-analyze
    # Use human_feedback to guide analysis
```

**revise:writing:**
```python
if decision == "revise:writing":
    # Route back to writer
    # Keep evidence and analysis
    # Rewrite with new instructions
```

**Benefits:**
1. **Quality Assurance:** Human catches errors before propagation
2. **Iterative Refinement:** Multiple revision cycles possible
3. **Flexible Control:** Human can steer direction
4. **Trust Building:** User sees each chapter's evolution

### 4. Structured Reasoning

**Problem:** LLMs can:
- Jump to conclusions without evidence
- Hallucinate information
- Make logical leaps
- Miss important aspects

**Solution:** Chain-of-thought with structured reasoning steps

#### Reasoning Chain:

**Step 1: Decomposition (Coordinator)**
```python
# User request → Structured outline
user_input = "Analyze China's EV industry"

# Coordinator generates plan:
global_plan = [
    "Industry Background",
    "Policy Environment",
    "Market Analysis",
    "Key Players",
    "Technology Trends",
    "Challenges",
    "Future Outlook"
]
```

**Step 2: Evidence Gathering (Researcher)**
```python
# Chapter title → Search queries
chapter_title = "Market Analysis"

# Researcher generates structured queries:
queries = [
    "China EV market size 2024",
    "EV sales statistics by brand",
    "Market growth rate trends",
    "Regional market distribution"
]

# Retrieves evidence:
evidence = [
    {"text": "...", "source": "...", "score": 0.92},
    {"text": "...", "source": "...", "score": 0.89}
]
```

**Step 3: Analysis (Analyst)**
```python
# Evidence → Structured insights
analysis = {
    "key_findings": [
        "Market grew 35% YoY",
        "BYD leads with 31.8% share"
    ],
    "trends": [
        "Shift to premium segments",
        "Increased export activity"
    ],
    "insights": [
        "Policy support driving growth",
        "Intensifying competition"
    ],
    "data_gaps": [
        "Q4 2024 data incomplete"
    ]
}
```

**Step 4: Synthesis (Writer)**
```python
# Evidence + Analysis → Coherent content
# Writer must:
# 1. Reference specific evidence (citations)
# 2. Build on analysis insights
# 3. Structure logically (intro → body → conclusion)
# 4. Maintain professional tone

current_draft = """
# Chapter 3: Market Analysis

## 3.1 Market Size and Growth
According to [Source], the market grew 35%...

## 3.2 Competitive Landscape
BYD leads with 31.8% market share...

## 3.3 Key Trends
Analysis shows shift to premium segments...
"""
```

**Structured Reasoning Benefits:**
1. **Traceability:** Each step builds on previous
2. **Verifiability:** Evidence is explicit
3. **Debugging:** Can inspect intermediate outputs
4. **Quality:** Structured approach reduces errors

---

## Agent Responsibilities

### 1. Coordinator Agent

**File:** `rag_project/agent/nodes/coordinator.py`

**Purpose:** Strategic planning and task decomposition

**Input:**
- `user_input`: User's request text

**Output:**
- `global_plan`: List of 5-8 chapter titles
- `current_chapter_index`: Set to 0

**Responsibilities:**
1. Analyze user request intent
2. Determine report scope and structure
3. Generate comprehensive chapter outline
4. Ensure logical flow between chapters

**Configuration:**
```python
AGENT_CONFIGS["coordinator"] = {
    "temperature": 0.3,  # Low for consistency
    "max_tokens": 2048,
    "system_prompt": "Be concise, strategic, and focused..."
}
```

**Prompt Strategy:**
```python
prompt = f"""
Based on user request: {user_input}

Generate 5-8 chapter titles covering:
- Industry/Topic Background
- Policy Environment
- Current Status Analysis
- Problems and Challenges
- Strategic Suggestions
- Additional relevant topics

Return ONLY a JSON array of strings.
"""
```

**Error Handling:**
- JSON parse failure → Use fallback plan
- Validation failure → Use default template
- API error → Return default structure

**Example Output:**
```json
[
  "第一章：中国新能源汽车行业发展背景与历程",
  "第二章：政策环境与监管框架分析",
  "第三章：市场规模与销量现状分析",
  "第四章：主要企业竞争格局",
  "第五章：技术创新与智能化发展",
  "第六章：面临的挑战与制约因素",
  "第七章：发展建议与战略展望"
]
```

### 2. Prepare Chapter Agent

**File:** `rag_project/agent/nodes/prep_chapter.py`

**Purpose:** Initialize chapter-specific context

**Input:**
- `global_plan`: Full chapter list
- `current_chapter_index`: Current chapter number
- `context_pool`: List of approved chapters

**Output:**
- `chapter_title`: Current chapter title
- `context_summary`: Compressed summary of previous chapters
- `chapter_scratchpad`: Empty dict for this chapter

**Responsibilities:**
1. Extract current chapter title from plan
2. Compress context_pool into brief summary
3. Initialize clean scratchpad for this chapter
4. Prepare context for Researcher

**Context Compression Strategy:**
```python
def compress_context(context_pool):
    # Don't pass full chapters (too much context)
    # Create brief summary instead

    all_titles = [extract_title(chapter) for chapter in context_pool]

    summary_prompt = f"""
    Previous chapters covered: {all_titles}

    Provide a 2-3 sentence summary of key points
    that the current chapter should reference.
    """

    return llm.invoke(summary_prompt, max_tokens=200)
```

**State Isolation:**
- Creates FRESH scratchpad (no previous chapter data)
- Prevents information leakage between chapters
- Ensures each chapter starts clean

**Example Output:**
```python
{
    "chapter_title": "第三章：市场规模与销量现状分析",
    "context_summary": "前两章介绍了行业发展背景和政策支持框架。本章应重点引用政策支持对市场增长的推动作用。",
    "chapter_scratchpad": {}  # Empty, ready for Researcher
}
```

### 3. Researcher Agent

**File:** `rag_project/agent/nodes/researcher.py`

**Purpose:** Information retrieval from knowledge base

**Input:**
- `chapter_title`: Current chapter to research
- `chapter_scratchpad`: Empty dict (to be populated)

**Output:**
- `chapter_scratchpad`: Updated with queries, evidence, sources

**Responsibilities:**
1. Generate search queries from chapter title
2. Query RAG knowledge base (Milvus)
3. Retrieve relevant documents with metadata
4. Populate scratchpad with structured evidence

**Query Generation:**
```python
# Generate 3-5 diverse queries
prompt = f"""
Generate search queries for: {chapter_title}

Create 3-5 specific queries that:
1. Cover different aspects of the topic
2. Use relevant keywords
3. Target different document types (news, reports, regulations)
4. Include recent time periods if applicable

Return as JSON array.
"""

queries = json.loads(llm.invoke(prompt))
```

**RAG Search:**
```python
for query in queries:
    results = rag_retriever.search(
        query=query,
        top_k=5,
        filters={
            "doc_type": ["news", "report", "regulation"],
            "publish_date": ["2023-01-01", "2024-12-31"]
        }
    )
    all_results.extend(results)
```

**Evidence Structuring:**
```python
chapter_scratchpad = {
    "queries": queries,
    "evidence": [
        {
            "text": "相关统计数据...",
            "source": "中国汽车工业协会",
            "publish_date": "2024-12-10",
            "score": 0.92,
            "doc_type": "report"
        },
        # ... more evidence
    ],
    "sources": list(set([r["source"] for r in results]))
}
```

**Configuration:**
```python
AGENT_CONFIGS["researcher"] = {
    "temperature": 0.1,  # Very low for precision
    "max_tokens": 4096,
    "system_prompt": "Be precise, thorough, and focus on finding relevant information..."
}
```

**Key Design:**
- ONLY agent with access to Milvus RAG
- Cannot see context_pool (prevents bias)
- Populates scratchpad (shared with Analyst and Writer)

### 4. Analyst Agent

**File:** `rag_project/agent/nodes/analyst.py`

**Purpose:** Critical analysis and insight generation

**Input:**
- `chapter_scratchpad`: Contains queries and evidence from Researcher

**Output:**
- `chapter_scratchpad`: Updated with analysis field

**Responsibilities:**
1. Read evidence from scratchpad
2. Identify patterns and trends
3. Compare multiple sources
4. Generate structured insights
5. Note data gaps or inconsistencies

**Analysis Process:**
```python
evidence = scratchpad["evidence"]

prompt = f"""
Analyze the following evidence for chapter: {chapter_title}

Evidence:
{format_evidence(evidence)}

Provide structured analysis covering:
1. Key Findings: What are the main facts?
2. Trends: What patterns emerge?
3. Insights: What deeper understanding can be derived?
4. Data Gaps: What information is missing or contradictory?
5. Connections: How do different sources relate?

Return as JSON object with these fields.
"""

analysis = json.loads(llm.invoke(prompt, temperature=0.5))
```

**Structured Output:**
```python
chapter_scratchpad["analysis"] = {
    "key_findings": [
        "2024年1-11月新能源汽车产销同比增长超35%",
        "市场渗透率提升至36.7%"
    ],
    "trends": [
        "纯电动车型占据主导地位（71.1%）",
        "插电混动车型增速更快（82.6%）",
        "出口市场成为新增长点"
    ],
    "insights": [
        "政策支持与市场需求双轮驱动增长",
        "技术进步推动成本下降",
        "市场竞争格局加速重构"
    ],
    "data_gaps": [
        "部分企业12月数据尚未发布",
        "细分车型数据不够详细"
    ],
    "source_agreement": "各来源数据基本一致",
    "confidence_level": "high"
}
```

**Configuration:**
```python
AGENT_CONFIGS["analyst"] = {
    "temperature": 0.5,  # Balanced for creativity + accuracy
    "max_tokens": 3072,
    "system_prompt": "Be analytical, balanced, and focus on deeper understanding..."
}
```

**Key Design:**
- Reads scratchpad["evidence"]
- Writes scratchpad["analysis"]
- Cannot see context_pool (unbiased analysis)
- Adds structured reasoning layer

### 5. Writer Agent

**File:** `rag_project/agent/nodes/writer.py`

**Purpose:** Content generation and synthesis

**Input:**
- `chapter_title`: Current chapter title
- `context_summary`: Brief summary of previous chapters
- `chapter_scratchpad`: Contains evidence and analysis

**Output:**
- `current_draft`: Markdown-formatted chapter content

**Responsibilities:**
1. Read evidence and analysis from scratchpad
2. Reference context_summary for continuity
3. Structure content logically
4. Write clear, professional content
5. Ensure coherence and readability

**Writing Process:**
```python
evidence = scratchpad["evidence"]
analysis = scratchpad["analysis"]

prompt = f"""
Write chapter: {chapter_title}

Previous chapters summary: {context_summary}

Evidence:
{format_evidence(evidence)}

Analysis:
{format_analysis(analysis)}

Requirements:
1. Structure with clear headings and subheadings
2. Cite sources from evidence (use [Source: XXX])
3. Build on analysis insights
4. Use professional, academic tone
5. Ensure logical flow
6. Length: 800-1500 words

Write the chapter in markdown format.
"""

current_draft = llm.invoke(prompt, temperature=0.7)
```

**Content Structure:**
```markdown
# 第三章：市场规模与销量现状分析

## 3.1 整体市场规模
[Content with citations]

### 市场增长态势
[Content referencing analysis]

## 3.2 细分市场分析
[Structured breakdown]

### 纯电动车型
[Detailed analysis]

### 插电混动车型
[Detailed analysis]

## 3.3 区域市场分布
[Geographic analysis]

## 3.4 总结
[Chapter summary connecting to next chapter]
```

**Configuration:**
```python
AGENT_CONFIGS["writer"] = {
    "temperature": 0.7,  # Higher for creativity and flow
    "max_tokens": 4096,
    "system_prompt": "Be creative, clear, and focus on effective communication..."
}
```

**Key Design:**
- Synthesizes evidence + analysis
- Adds context_summary for continuity
- Cannot see full context_pool (prevents copying)
- Generates human-readable output

### 6. Human Review Node

**File:** `rag_project/agent/nodes/human_review.py`

**Purpose:** HITL interface for quality control

**Input:**
- `current_draft`: Chapter content from Writer
- `review_decision`: Human's decision (approve, revise:*)
- `human_feedback`: Structured feedback dict

**Output:**
- Updated state based on decision

**Responsibilities:**
1. Receive human decision
2. Update state accordingly
3. Route to appropriate next node

**State Update Logic:**
```python
if review_decision == "approve":
    # Add to approved memory
    full_chapter = f"# {chapter_title}\n\n{current_draft}"
    context_pool.append(full_chapter)

    # Clear working memory
    chapter_scratchpad = {}

    # Advance index
    chapter_index += 1

elif review_decision.startswith("revise"):
    # Keep state as-is
    # Don't add to context_pool
    # Don't clear scratchpad
    # Don't advance index
    # Routing handled by should_continue()
```

**Feedback Structure:**
```python
human_feedback = {
    "instruction": "Main instruction text",

    # For revise:data
    "query": "Additional search query",
    "aspect": "market_size",

    # For revise:logic
    "focus_area": "competitive_analysis",
    "missing_aspects": ["price_comparison", "technology_gap"],

    # For revise:writing
    "target_section": "conclusion",
    "tone": "professional_brief",
    "length": "shorter"
}
```

**Key Design:**
- Interrupt point in workflow
- Gatekeeper for context_pool
- Enables iterative refinement
- Routes to appropriate revision node

### 7. Archiver Agent

**File:** `rag_project/agent/nodes/archiver.py`

**Purpose:** Final report compilation

**Input:**
- `context_pool`: All approved chapters
- `user_input`: Original user request

**Output:**
- `final_report`: Complete markdown report

**Responsibilities:**
1. Compile all chapters from context_pool
2. Add title and metadata
3. Generate table of contents
4. Format final report
5. Save to file

**Report Structure:**
```markdown
# 中国新能源汽车行业研究报告 2024

**生成时间:** 2024-03-28
**报告类型:** 行业分析报告

---

## 目录

1. 行业背景与发展历程
2. 政策环境与监管框架
3. 市场规模与销量现状分析
4. 主要企业竞争格局
5. 技术创新与智能化发展
6. 面临的挑战与制约因素
7. 发展建议与战略展望

---

[Full chapters from context_pool]

---

**报告结束**

*本报告由AI辅助生成，内容基于公开资料整理。*
```

**Configuration:**
```python
def archiver_node(state):
    context_pool = state.get("context_pool", [])
    user_input = state.get("user_input", "")

    # Generate title
    title = generate_title(user_input)

    # Generate TOC
    toc = generate_toc(context_pool)

    # Compile report
    final_report = f"# {title}\n\n{toc}\n\n"
    final_report += "\n\n---\n\n".join(context_pool)
    final_report += "\n\n---\n\n**报告结束**"

    # Save to file
    output_path = save_report(final_report)

    return {"final_report": final_report, "output_path": output_path}
```

---

## Routing Logic

### Conditional Edges

**File:** `rag_project/agent/graph.py`

```python
workflow.add_conditional_edges(
    "human_review",
    should_continue,
    {
        "continue": "prepare_chapter",  # Next chapter
        "end": "archiver",              # Finalize report
        "researcher": "researcher",     # Data revision
        "analyst": "analyst",           # Logic revision
        "writer": "writer"              # Writing revision
    }
)
```

### Routing Function

**File:** `rag_project/agent/nodes/human_review.py`

```python
def should_continue(state) -> str:
    """
    Determines next destination based on review decision.

    Routing Logic:
    ┌─────────────────────────────────────┐
    │ review_decision = ?                 │
    └──────────────────┬──────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
  revise:data      revise:logic   revise:writing
       │               │               │
       ▼               ▼               ▼
  researcher       analyst         writer
       │               │               │
       └───────────────┴───────────────┘
                       │
                       ▼
            (revision complete, back to human_review)
                       │
       ┌───────────────┴───────────────┐
       │                               │
       ▼                               ▼
    approve                        finished
       │                               │
       ▼                               │
  more chapters?                        │
       │                               │
   ┌───┴───┐                           │
   │       │                           │
  Yes     No                           │
   │       │                           │
   │       └───────────────┬───────────┘
   │                       │
   ▼                       ▼
prepare_chapter         archiver
   │                       │
   └───────────────────────┘
           │
           ▼
      (workflow continues)
    """
    decision = state.get("review_decision")

    # Revision routes
    if decision == "revise:data":
        return "researcher"
    elif decision == "revise:logic":
        return "analyst"
    elif decision == "revise:writing":
        return "writer"
    elif decision == "finished":
        return "end"

    # Approve routes
    if decision == "approve":
        chapter_index = state.get("chapter_index", 0)
        chapter_titles = state.get("chapter_titles", [])

        if chapter_index + 1 < len(chapter_titles):
            return "continue"  # More chapters to process
        else:
            return "end"  # All chapters done

    return "end"  # Default fallback
```

### State Update Rules

**On approve:**
```python
# human_review_node
if decision == "approve":
    # Add current draft to approved memory
    context_pool.append(full_chapter)

    # Clear working memory for next chapter
    chapter_scratchpad = {}

    # Increment index
    chapter_index += 1
```

**On revise:**
```python
# No state changes
# Keep same scratchpad
# Keep same chapter_index
# Don't add to context_pool
```

**On archiver:**
```python
# Compile final report from context_pool
# No further state updates
```

---

## Tech Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Workflow Engine** | LangGraph | Latest | Multi-agent orchestration |
| **State Management** | LangGraph StateGraph | Latest | Global state and routing |
| **LLM Provider** | DeepSeek API | - | Language model |
| **LLM Client** | OpenAI SDK | 1.x | API interface |
| **Vector Database** | Milvus | 2.3+ | Semantic search |
| **Embeddings** | BGE-M3 | - | Vector embeddings |
| **Configuration** | PyYAML | 6.x | Config management |
| **Logging** | Python logging | Stdlib | Debug logging |
| **Environment** | python-dotenv | 1.x | Env variable management |

### Dependencies

**requirements.txt:**
```txt
# Core dependencies
langgraph>=0.0.20
langchain-core>=0.1.0
openai>=1.0.0
pymilvus>=2.3.0
python-dotenv>=1.0.0
pyyaml>=6.0

# Project dependencies
# (from main project)
sentence-transformers
pymilvus[binary]
```

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Agent System (Python)              │   │
│  │  • LangGraph workflow                           │   │
│  │  • Multi-agent nodes                            │   │
│  │  • State management                             │   │
│  └───────���─���───────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Service Layer                          │
│  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │  LLM Manager     │  │  RAG Retriever           │    │
│  │  • DeepSeek API  │  │  • Milvus Client         │    │
│  │  • OpenAI SDK    │  │  • Vector Search         │    │
│  └──────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                   │
│  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │  DeepSeek API    │  │  Milvus Vector DB        │    │
│  │  • LLM Service   │  │  • Docker Container      │    │
│  │  • HTTPS/REST    │  │  • Port 19530            │    │
│  └──────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input
    │
    ▼
GraphState Initialization
    │
    ▼
Coordinator Node → LLM Manager → DeepSeek API
    │
    ▼
Prepare Chapter Node
    │
    ▼
Researcher Node → RAG Retriever → Milvus
    │
    ▼
Analyst Node → LLM Manager → DeepSeek API
    │
    ▼
Writer Node → LLM Manager → DeepSeek API
    │
    ▼
Human Review (INTERRUPT)
    │
    ├─ approve → context_pool update → Prepare Chapter
    ├─ revise:data → Researcher
    ├─ revise:logic → Analyst
    └─ revise:writing → Writer
    │
    ▼ (all chapters approved)
Archiver Node → Final Report
```

---

## Extension Guide

### Adding New Agents

#### Step 1: Create Agent Node File

Create `rag_project/agent/nodes/new_agent.py`:

```python
"""
New Agent Node - Description of purpose

This node handles specific functionality...
"""

import logging
from typing import Dict

from rag_project.utils.logger import setup_logger

logger = setup_logger(__name__)


def new_agent_node(state: Dict) -> Dict:
    """
    Process state and return updates.

    Args:
        state: Current GraphState

    Returns:
        Dict with state updates
    """
    logger.info("New agent node executing")

    # Your logic here
    result = process_something(state)

    return {
        "some_field": result
    }


def process_something(state: Dict) -> str:
    """Helper function for processing"""
    # Implementation
    pass
```

#### Step 2: Export from __init__.py

Edit `rag_project/agent/nodes/__init__.py`:

```python
from rag_project.agent.nodes.new_agent import new_agent_node

__all__ = [
    # ... existing exports
    "new_agent_node",
]
```

#### Step 3: Add to Workflow

Edit `rag_project/agent/graph.py`:

```python
from rag_project.agent.nodes import new_agent_node

def create_report_graph():
    workflow = StateGraph(GraphState)

    # Add new node
    workflow.add_node("new_agent", new_agent_node)

    # Add edges
    workflow.add_edge("some_node", "new_agent")
    workflow.add_edge("new_agent", "next_node")

    # Add conditional edges if needed
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            # ... existing routes
            "revise:new": "new_agent"  # New revision route
        }
    )

    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"]
    )
```

#### Step 4: Update Routing Logic

Edit `rag_project/agent/nodes/human_review.py`:

```python
def should_continue(state) -> str:
    decision = state.get("review_decision")

    # Add new route
    if decision == "revise:new":
        return "new_agent"

    # ... existing logic
```

### Customizing Agent Prompts

#### Method 1: Via Config File

Edit `config/agent_config.yaml`:

```yaml
agents:
  writer:
    temperature: 0.7
    max_tokens: 4096
    system_prompt: |
      You are a specialized writer for Chinese policy research.

      Guidelines:
      - Use formal, academic tone
      - Cite sources with [Source: XXX]
      - Structure with clear headings
      - Include data tables when relevant
      - Keep paragraphs concise (3-5 sentences)

      Be professional, precise, and authoritative.
```

#### Method 2: Via Code

Edit `rag_project/agent/llm_manager.py`:

```python
class LLMManager:
    AGENT_CONFIGS = {
        "writer": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "system_prompt": """Custom prompt here..."""
        }
    }
```

### Adding New Feedback Types

#### Step 1: Extend Routing

Edit `rag_project/agent/nodes/human_review.py`:

```python
def should_continue(state) -> str:
    decision = state.get("review_decision")

    # Add new feedback type
    if decision == "revise:format":
        return "formatter"  # Route to new node

    # ... existing logic
```

#### Step 2: Create Handler Node

Create `rag_project/agent/nodes/formatter.py`:

```python
def formatter_node(state: Dict) -> Dict:
    """
    Handle formatting revisions.

    Reads human_feedback for formatting instructions
    and reformats current_draft accordingly.
    """
    feedback = state.get("human_feedback", {})
    current_draft = state.get("current_draft", "")

    # Apply formatting changes
    new_draft = apply_formatting(current_draft, feedback)

    return {"current_draft": new_draft}
```

#### Step 3: Add to Workflow

```python
from rag_project.agent.nodes.formatter import formatter_node

workflow.add_node("formatter", formatter_node)
workflow.add_edge("formatter", "human_review")
```

### Customizing State Schema

Edit `rag_project/agent/state.py`:

```python
from typing import TypedDict, List, Dict, Annotated
import operator

class GraphState(TypedDict):
    """Extended state schema"""

    # Existing fields
    user_input: str
    global_plan: List[str]
    current_chapter_index: int
    context_pool: Annotated[List[str], operator.add]
    context_summary: str
    chapter_title: str
    chapter_scratchpad: Dict
    current_draft: str
    human_feedback: Dict

    # New fields
    metadata: Dict  # Report metadata (author, date, version)
    revision_count: int  # Track number of revisions
    quality_score: float  # Quality metric
    custom_field: str  # Your custom field
```

### Adding Persistence

#### Save State to Database

```python
import pickle
import os

def save_state(state: Dict, thread_id: str):
    """Save state to file"""
    state_dir = "states"
    os.makedirs(state_dir, exist_ok=True)

    state_file = os.path.join(state_dir, f"{thread_id}.pkl")
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

def load_state(thread_id: str) -> Dict:
    """Load state from file"""
    state_file = os.path.join("states", f"{thread_id}.pkl")

    with open(state_file, "rb") as f:
        return pickle.load(f)
```

#### Use in Workflow

```python
# Save after each chapter
state = app.invoke(state, config=config)
save_state(state, config["configurable"]["thread_id"])

# Load later
state = load_state("report_001")
state = app.invoke(state, config=config)
```

### Integrating New LLM Providers

#### Example: Add Azure OpenAI

Edit `rag_project/agent/llm_manager.py`:

```python
class LLMManager:
    def __init__(self, agent_type: str = "coordinator", provider: str = "deepseek"):
        self.provider = provider

        if provider == "azure":
            self.client = OpenAI(
                api_key=os.environ["AZURE_API_KEY"],
                api_version="2023-05-15",
                azure_endpoint=os.environ["AZURE_ENDPOINT"]
            )
        elif provider == "deepseek":
            self.client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com"
            )
```

#### Configure Provider

```python
# Use Azure for writer (better quality)
llm = LLMManager("writer", provider="azure")

# Use DeepSeek for others (faster, cheaper)
llm = LLMManager("coordinator", provider="deepseek")
```

### Adding Evaluation Metrics

#### Create Evaluator

```python
def evaluate_chapter_quality(state: Dict) -> Dict:
    """Evaluate chapter quality on multiple dimensions"""

    current_draft = state.get("current_draft", "")
    evidence = state.get("chapter_scratchpad", {}).get("evidence", [])

    metrics = {
        "length": len(current_draft.split()),
        "citation_count": current_draft.count("[Source:"),
        "evidence_usage": len([e for e in evidence if e["text"] in current_draft]),
        "readability": calculate_readability(current_draft),
        "coherence": calculate_coherence(current_draft)
    }

    return {"quality_metrics": metrics}
```

#### Integrate into Workflow

```python
# Add evaluator node
workflow.add_node("evaluator", evaluate_chapter_quality)
workflow.add_edge("writer", "evaluator")
workflow.add_edge("evaluator", "human_review")
```

### Custom Output Formats

#### Export to PDF

```python
from weasyprint import HTML
import markdown

def export_to_pdf(markdown_content: str, output_path: str):
    """Convert markdown report to PDF"""

    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content)

    # Add CSS styling
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Microsoft YaHei', sans-serif; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    HTML(string=html).write_pdf(output_path)
```

#### Export to DOCX

```python
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def export_to_docx(markdown_content: str, output_path: str):
    """Convert markdown report to DOCX"""

    doc = Document()

    # Parse markdown and add to document
    lines = markdown_content.split("\n")

    for line in lines:
        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.strip():
            doc.add_paragraph(line)

    doc.save(output_path)
```

---

## Best Practices

### Agent Design

1. **Single Responsibility:** Each agent should do one thing well
2. **State Isolation:** Agents should only access needed state fields
3. **Error Handling:** Always handle LLM API failures gracefully
4. **Logging:** Log key decisions and state changes
5. **Testing:** Unit test each agent independently

### Prompt Engineering

1. **Be Specific:** Clear instructions reduce errors
2. **Use Examples:** Show desired output format
3. **Set Constraints:** Specify length, tone, structure
4. **Validate Output:** Check and parse LLM responses
5. **Iterate:** Refine prompts based on results

### Workflow Design

1. **Linear Flow:** Main flow should be simple and linear
2. **Conditional Routes:** Use conditional edges for branching
3. **Interrupt Points:** Clear points where human input is needed
4. **State Updates:** Batch state updates per node
5. **Checkpointing:** Save state for resumption

### Performance Optimization

1. **Caching:** Cache LLM responses when possible
2. **Parallel Processing:** Run independent queries in parallel
3. **Batch Operations:** Process multiple items together
4. **Token Management:** Use appropriate max_tokens per agent
5. **Async I/O:** Use async for API calls

---

## Troubleshooting

### Common Issues

**Issue:** State not updating between nodes
- **Cause:** Forgetting to return dict from node
- **Fix:** Ensure all nodes return state update dict

**Issue:** Routing not working
- **Cause:** Missing edge or incorrect route name
- **Fix:** Check all edges in workflow definition

**Issue:** LLM timeout
- **Cause:** API slow or overloaded
- **Fix:** Increase timeout in config, add retry logic

**Issue:** Memory leaks
- **Cause:** State growing too large
- **Fix:** Clear scratchpad after each chapter, compress context_pool

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [DeepSeek API Documentation](https://platform.deepseek.com/docs)

---

## Version History

- **v1.0** (2024-03-28): Initial agent system architecture
  - Multi-agent workflow with LangGraph
  - HITL support
  - Three-tier memory architecture
  - 7 specialized agents
