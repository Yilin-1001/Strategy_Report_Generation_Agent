# Agent System User Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Interactive Mode vs Auto Mode](#interactive-mode-vs-auto-mode)
- [Advanced Usage](#advanced-usage)
- [Workflow Diagram](#workflow-diagram)
- [Human Review Best Practices](#human-review-best-practices)
- [FAQ](#faq)
- [Example Reports](#example-reports)

---

## Quick Start

### Environment Preparation

#### 1. Prerequisites

Ensure you have the following installed:
- Python 3.9 or higher
- Git
- Milvus server (running locally or remotely)

#### 2. Install Dependencies

```bash
# Navigate to project directory
cd E:/02 Final Year Project/RAG Project

# Install core dependencies
pip install langgraph langchain-openai openai pymilvus python-dotenv pyyaml

# Install additional dependencies
pip install tqdm colorama
```

#### 3. Configure API Keys

Create a `.env` file in the project root:

```bash
# DeepSeek API (for LLM)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: Override default model
DEEPSEEK_MODEL=deepseek-chat
```

#### 4. Configure Milvus Connection

Edit `config/milvus_config.yaml`:

```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: rag_chunks
  embedding_dimension: 1024
```

#### 5. Verify Setup

Test your configuration:

```python
from rag_project.agent import LLMManager, RAGRetriever

# Test LLM connection
llm = LLMManager("coordinator")
response = llm.invoke("Hello, can you hear me?")
print(response)

# Test RAG connection
retriever = RAGRetriever()
results = retriever.search("test query", top_k=3)
print(f"Found {len(results)} results")
```

### Basic Usage

#### Running Your First Report Generation

```python
from rag_project.agent.graph import create_report_graph

# Create the workflow graph
app = create_report_graph()

# Initial state
initial_state = {
    "user_input": "Generate a comprehensive report on China's transportation industry in 2024",
    "chapter_index": 0,
    "chapter_titles": [],
    "context_pool": [],
    "scratchpad": [],
    "current_draft": "",
    "review_decision": None
}

# Run in auto mode (no human intervention)
config = {"configurable": {"thread_id": "report_001"}}
result = app.invoke(initial_state, config=config)

print(result["final_report"])
```

---

## Interactive Mode vs Auto Mode

### Interactive Mode (Recommended)

Interactive mode allows you to review and refine each chapter before proceeding.

#### Setup

```python
from rag_project.agent.graph import create_report_graph

app = create_report_graph()
config = {"configurable": {"thread_id": "report_interactive_001"}}
```

#### Workflow

1. **Start the workflow**

```python
# Run until first human review point
state = app.invoke(initial_state, config=config)

# Current chapter draft is ready for review
print(state["current_draft"])
```

2. **Review the chapter**

```python
chapter_title = state["chapter_titles"][state["chapter_index"]]
print(f"Chapter: {chapter_title}")
print(state["current_draft"])
```

3. **Provide feedback**

```python
# Option 1: Approve and continue
state["review_decision"] = "approve"
state = app.invoke(state, config=config)

# Option 2: Request revision with new data
state["review_decision"] = "revise:data"
state["human_feedback"] = {
    "query": "Add more recent statistics from 2024 Q4",
    "aspect": "market_size"
}
state = app.invoke(state, config=config)

# Option 3: Request logic revision
state["review_decision"] = "revise:logic"
state["human_feedback"] = {
    "instruction": "The analysis section needs deeper insights",
    "focus_area": "competitive_landscape"
}
state = app.invoke(state, config=config)

# Option 4: Request writing revision
state["review_decision"] = "revise:writing"
state["human_feedback"] = {
    "instruction": "Make the tone more professional and concise",
    "target_section": "conclusion"
}
state = app.invoke(state, config=config)
```

4. **Continue until completion**

```python
# Loop through all chapters
while True:
    state = app.invoke(state, config=config)

    # Check if workflow is complete
    if "final_report" in state:
        print("Report generation complete!")
        break

    # Review current chapter
    print(f"\n{'='*60}")
    print(f"Chapter: {state['chapter_titles'][state['chapter_index']]}")
    print(f"{'='*60}\n")
    print(state["current_draft"])

    # Get user input
    decision = input("\nDecision (approve/revise:data/revise:logic/revise:writing): ")

    if decision.startswith("revise"):
        feedback = input("Feedback: ")
        state["review_decision"] = decision
        state["human_feedback"] = {"instruction": feedback}
    else:
        state["review_decision"] = "approve"
        state["human_feedback"] = {}
```

### Auto Mode

Auto mode runs through all chapters without human intervention.

```python
from rag_project.agent.graph import create_report_graph

app = create_report_graph()
config = {"configurable": {"thread_id": "report_auto_001"}}

# Set auto-approve behavior
initial_state["review_decision"] = "approve"

# Run entire workflow
result = app.invoke(initial_state, config=config)

# Save report
with open("report_auto.md", "w", encoding="utf-8") as f:
    f.write(result["final_report"])

print("Auto-generated report saved to report_auto.md")
```

**Use Auto Mode when:**
- Generating draft reports for initial exploration
- Processing well-defined, standard topics
- Running batch report generation
- Time-constrained scenarios

**Use Interactive Mode when:**
- High-quality output is critical
- Topic requires domain expertise
- Complex analysis needed
- Client deliverables

---

## Advanced Usage

### Custom Output Path

```python
import os
from rag_project.agent.graph import create_report_graph

# Set custom output directory
output_dir = "reports/custom_topic_2024"
os.makedirs(output_dir, exist_ok=True)

app = create_report_graph()
config = {"configurable": {"thread_id": "custom_001"}}

result = app.invoke(initial_state, config=config)

# Save with custom filename
output_path = os.path.join(output_dir, "final_report.md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result["final_report"])

print(f"Report saved to {output_path}")
```

### Batch Report Generation

```python
from rag_project.agent.graph import create_report_graph
import json
from datetime import datetime

# Define multiple topics
topics = [
    "China's electric vehicle industry 2024",
    "Renewable energy policy in China",
    "Smart transportation systems",
    "Autonomous driving regulations"
]

app = create_report_graph()
results = []

for i, topic in enumerate(topics):
    print(f"\n{'='*60}")
    print(f"Generating report {i+1}/{len(topics)}: {topic}")
    print(f"{'='*60}\n")

    initial_state = {
        "user_input": f"Generate a comprehensive report on {topic}",
        "chapter_index": 0,
        "chapter_titles": [],
        "context_pool": [],
        "scratchpad": [],
        "current_draft": "",
        "review_decision": "approve"  # Auto mode
    }

    config = {"configurable": {"thread_id": f"batch_{i:03d}"}}
    result = app.invoke(initial_state, config=config)

    # Save individual report
    filename = f"batch_report_{i+1:02d}_{datetime.now().strftime('%Y%m%d')}.md"
    output_path = f"reports/batch/{filename}"

    os.makedirs("reports/batch", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["final_report"])

    results.append({
        "topic": topic,
        "filename": filename,
        "chapters": len(result.get("context_pool", [])),
        "timestamp": datetime.now().isoformat()
    })

# Save batch summary
with open("reports/batch/summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nBatch generation complete! {len(results)} reports generated.")
```

### Custom Agent Configuration

Override default agent settings:

```python
from rag_project.agent import LLMManager
from rag_project.agent.graph import create_report_graph

# Create custom LLM manager with different temperature
custom_llm = LLMManager("writer")
custom_llm.temperature = 0.9  # More creative
custom_llm.max_tokens = 6000

# Use in custom workflow
app = create_report_graph()
# ... proceed with normal usage
```

### Filtering RAG Results

```python
from rag_project.agent.retriever import RAGRetriever

retriever = RAGRetriever()

# Filter by document type
results = retriever.search(
    query="transportation policy",
    top_k=10,
    filters={"doc_type": ["regulation", "policy"]}
)

# Filter by date range
results = retriever.search(
    query="market analysis",
    top_k=10,
    filters={"publish_date": ["2024-01-01", "2024-12-31"]}
)

# Filter by source
results = retriever.search(
    query="industry trends",
    top_k=10,
    filters={"source": ["Ministry of Transport", "NDRC"]}
)
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│  "Generate a report on China's transportation industry 2024"    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      COORDINATOR NODE                            │
│  • Analyzes user request                                         │
│  • Generates 5-8 chapter outline                                 │
│  • Creates global plan                                          │
│  Output: global_plan = [                                         │
│    "Industry Background",                                        │
│    "Policy Environment",                                         │
│    "Current Status",                                             │
│    "Major Problems",                                             │
│    "Strategic Suggestions",                                      │
│    "Future Outlook"                                              │
│  ]                                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPARE CHAPTER NODE                          │
│  • Gets current chapter title from plan                          │
│  • Compresses context_pool into summary                          │
│  • Initializes chapter_scratchpad                                │
│  Output: chapter_title, context_summary, scratchpad              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RESEARCHER NODE                             │
│  • Generates search queries from chapter title                   │
│  • Queries RAG knowledge base (Milvus)                           │
│  • Retrieves relevant documents                                  │
│  • Populates chapter_scratchpad with evidence                    │
│  Output: scratchpad = {                                          │
│    "queries": [...],                                             │
│    "evidence": [...],                                            │
│    "sources": [...]                                              │
│  }                                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ANALYST NODE                               │
│  • Analyzes evidence in scratchpad                               │
│  • Identifies patterns and insights                              │
│  • Compares multiple sources                                     │
│  • Generates structured analysis                                 │
│  Output: scratchpad["analysis"] = {                              │
│    "key_findings": [...],                                        │
│    "trends": [...],                                              │
│    "insights": [...],                                            │
│    "data_gaps": [...]                                            │
│  }                                                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       WRITER NODE                                │
│  • Synthesizes evidence + analysis                               │
│  • Structures content logically                                  │
│  • Drafts chapter content                                        │
│  • Ensures coherence and readability                             │
│  Output: current_draft (markdown formatted chapter)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HUMAN REVIEW NODE                              │
│  ⚠️ WORKFLOW INTERRUPTS - AWAITING HUMAN INPUT                   │
│                                                                   │
│  Human reviews current_draft and provides decision:              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DECISION OPTIONS:                                       │   │
│  │                                                          │   │
│  │ approve  ──────► Add to context_pool                    │   │
│  │                     │                                    │   │
│  │                     ▼                                    │   │
│  │               Next chapter (prepare_chapter)             │   │
│  │                     │                                    │   │
│  │                     ◄─────┐                              │   │
│  │                            │                              │   │
│  │  All chapters done?         │                              │   │
│  │  Yes ──► archiver ──► END   │                              │   │
│  │  No ───► prepare_chapter    │                              │   │
│  │                            │                              │   │
│  │  revise:data ──► researcher │ (Add new queries)           │   │
│  │  revise:logic ─► analyst   │ (Refine analysis)            │   │
│  │  revise:writing ─► writer  │ (Improve writing)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼ (after all chapters approved)
┌─────────────────────────────────────────────────────────────────┐
│                      ARCHIVER NODE                               │
│  • Compiles all chapters from context_pool                       │
│  • Adds title, table of contents, metadata                       │
│  • Formats final report                                          │
│  • Saves to file                                                 │
│  Output: final_report (complete markdown document)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        END                                       │
│  Report saved, workflow complete                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Human Review Best Practices

### Decision Guidelines

#### When to APPROVE

- Chapter covers all key aspects of the topic
- Evidence is relevant and well-supported
- Analysis is logical and insightful
- Writing is clear and professional
- No factual errors or major omissions

**Example:**
```python
state["review_decision"] = "approve"
state["human_feedback"] = {}  # No feedback needed
```

#### When to Request DATA Revision (`revise:data`)

Use when:
- Missing critical information or statistics
- Evidence is outdated or insufficient
- Need more specific examples or case studies
- Important sources are not referenced

**Example:**
```python
state["review_decision"] = "revise:data"
state["human_feedback"] = {
    "query": "Find more recent data on EV sales in Q4 2024",
    "aspect": "market_size",
    "specific_sources": ["China Association of Automobile Manufacturers"]
}
```

#### When to Request LOGIC Revision (`revise:logic`)

Use when:
- Analysis is superficial or lacks depth
- Arguments are weak or poorly structured
- Missing connections between concepts
- Conclusions don't follow from evidence
- Need better synthesis of information

**Example:**
```python
state["review_decision"] = "revise:logic"
state["human_feedback"] = {
    "instruction": "The competitive landscape analysis needs to compare domestic vs foreign companies",
    "focus_area": "competitive_analysis",
    "missing_aspects": ["market_share_comparison", "technology_gap_analysis"]
}
```

#### When to Request WRITING Revision (`revise:writing`)

Use when:
- Tone is inappropriate (too casual, too academic)
- Structure is confusing or hard to follow
- Language is verbose or unclear
- Missing transitions between sections
- Formatting issues (headings, lists, etc.)

**Example:**
```python
state["review_decision"] = "revise:writing"
state["human_feedback"] = {
    "instruction": "Make the conclusion more concise and actionable",
    "target_section": "conclusion",
    "tone": "professional_brief",
    "length": "shorter"
}
```

### Review Checklist

For each chapter, check:

- **Content Quality**
  - [ ] Addresses the chapter title/topic
  - [ ] Sufficient depth and detail
  - [ ] Accurate and up-to-date information
  - [ ] Relevant examples and evidence

- **Analysis Quality**
  - [ ] Clear logical flow
  - [ ] Insights and patterns identified
  - [ ] Multiple perspectives considered
  - [ ] Data interpreted correctly

- **Writing Quality**
  - [ ] Clear and readable
  - [ ] Appropriate tone
  - [ ] Well-structured with headings
  - [ ] No grammatical errors

- **Completeness**
  - [ ] All key points covered
  - [ ] No major gaps or omissions
  - [ ] Sources cited appropriately
  - [ ] Figures/tables if needed

### Tips for Effective Feedback

1. **Be Specific**
   - Bad: "Make it better"
   - Good: "Add specific statistics on market growth rates from 2020-2024"

2. **Provide Context**
   - Include what's missing or what needs improvement
   - Reference specific sections or paragraphs

3. **Prioritize Issues**
   - Address major content gaps first
   - Then focus on analysis depth
   - Finally polish writing style

4. **Use Structured Feedback**
   ```python
   state["human_feedback"] = {
       "instruction": "Main request",
       "focus_area": "specific_section",
       "missing_aspects": ["item1", "item2"],
       "specific_sources": ["source1", "source2"],
       "tone": "professional",
       "length": "moderate"
   }
   ```

---

## FAQ

### Q1: How long does report generation take?

**A:** Depends on mode and topic complexity:
- Auto mode: ~5-10 minutes per report (6-8 chapters)
- Interactive mode: ~15-30 minutes per report (with review time)
- Main factors: API response time, number of chapters, revision cycles

### Q2: Can I pause and resume later?

**A:** Yes! The workflow state is automatically saved:

```python
# Pause
state = app.invoke(initial_state, config=config)
# State is saved with thread_id

# Resume later (same thread_id)
config = {"configurable": {"thread_id": "report_001"}}
state = app.get_state(config)
# Continue from where you left off
```

### Q3: What if the LLM generates invalid JSON?

**A:** The system has automatic fallback:
- Coordinator falls back to default chapter outline
- Other nodes log errors and retry with different prompts
- Check logs in `logs/agent.log` for details

### Q4: Can I use a different LLM provider?

**A:** Currently configured for DeepSeek API. To use another provider:
1. Modify `rag_project/agent/llm_manager.py`
2. Update `base_url` and API key handling
3. Adjust prompt formats if needed

### Q5: How do I improve report quality?

**A:** Several approaches:
1. **Use Interactive Mode** - Review and refine each chapter
2. **Improve Knowledge Base** - Add more relevant documents
3. **Customize Prompts** - Edit system prompts in `config/agent_config.yaml`
4. **Adjust Temperature** - Lower for factual, higher for creative
5. **Provide Better Feedback** - Be specific and structured

### Q6: Can I generate reports in English?

**A:** Yes, specify language in user input:
```python
initial_state["user_input"] = "Generate a comprehensive report in English on China's EV industry"
```

Or edit the system prompts to default to English.

### Q7: What happens if Milvus is down?

**A:** The researcher node will fail. Check:
1. Milvus server is running: `docker ps | grep milvus`
2. Connection config in `config/milvus_config.yaml`
3. Collection exists and has data

### Q8: Can I customize the chapter outline?

**A:** Two approaches:
1. **Provide custom outline in user input:**
   ```python
   initial_state["user_input"] = """
   Generate a report with these chapters:
   1. Market Overview
   2. Technology Analysis
   3. Key Players
   4. Future Trends
   """
   ```

2. **Edit coordinator prompts** in `coordinator.py`

### Q9: How do I handle very large reports?

**A:** For reports >20 chapters:
1. Break into multiple sub-reports
2. Use batch generation with related topics
3. Merge reports manually in post-processing

### Q10: Can I export to formats other than Markdown?

**A:** Yes, convert the markdown output:
```python
# To HTML
import markdown
html = markdown.markdown(result["final_report"])

# To PDF (requires weasyprint)
from weasyprint import HTML
HTML(string=markdown.markdown(result["final_report"])).write_pdf("report.pdf")

# To DOCX (requires python-docx)
from docx import Document
doc = Document()
# Add content...
doc.save("report.docx")
```

---

## Example Reports

### Example 1: Industry Analysis Report

**Input:**
```python
initial_state = {
    "user_input": "Generate a comprehensive analysis report on China's new energy vehicle industry in 2024, covering market size, policy support, key players, technological advancements, and future outlook"
}
```

**Generated Chapters:**
1. 新能源汽车行业发展背景与历程
2. 政策环境与监管框架分析
3. 市场规模与销量现状分析
4. 主要企业竞争格局
5. 技术创新与智能化发展
6. 面临的挑战与制约因素
7. 发展建议与战略展望

**Sample Output (Chapter 3 excerpt):**
```markdown
# 第三章：市场规模与销量现状分析

## 3.1 整体市场规模

2024年，中国新能源汽车市场继续保持强劲增长态势。根据中国汽车工业协会数据，2024年1-11月，新能源汽车产销分别完成705.8万辆和706.8万辆，同比分别增长33.6%和35.2%。

### 市场渗透率持续提升

新能源汽车的市场渗透率已从2020年的5.4%提升至2024年11月的36.7%，显示出消费者接受度的显著提高。

## 3.2 细分市场分析

### 纯电动车型主导市场
纯电动汽车（BEV）在2024年1-11月累计销售502.8万辆，占比71.1%...

### 插电混动车型快速增长
插电式混合动力汽车（PHEV）销量达到204.0万辆，同比增长82.6%...
```

### Example 2: Policy Research Report

**Input:**
```python
initial_state = {
    "user_input": "Analyze China's carbon neutrality policy framework, including national targets, implementation measures, regional progress, challenges, and international cooperation"
}
```

**Generated Chapters:**
1. 碳中和政策背景与目标体系
2. 政策框架与实施机制
3. 重点行业减排措施分析
4. 区域推进进展与差异
5. 面临的主要挑战
6. 国际合作与经验借鉴
7. 政策建议与展望

### Example 3: Technology Trend Report

**Input:**
```python
initial_state = {
    "user_input": "Research the development trends of artificial intelligence in China's healthcare sector, including applications, regulations, key companies, and future opportunities"
}
```

**Generated Chapters:**
1. 人工智能在医疗领域的应用背景
2. 核心技术与应用场景分析
3. 政策法规与监管环境
4. 主要企业与产品布局
5. 技术挑战与数据安全
6. 商业模式与市场前景
7. 发展建议与趋势预测

### Sample Review Session

**Chapter Draft:**
```markdown
# 第四章：主要企业竞争格局

比亚迪、特斯拉、蔚来、小鹏、理想等企业在市场竞争中各具特色...
[Content continues]
```

**Human Feedback (revise:data):**
```python
state["review_decision"] = "revise:data"
state["human_feedback"] = {
    "query": "Get specific market share data for top 10 NEV companies in 2024",
    "aspect": "market_share",
    "missing_companies": ["吉利汽车", "长安汽车", "广汽埃安"]
}
```

**Revised Chapter (after researcher node):**
```markdown
# 第四章：主要企业竞争格局

## 4.1 市场份额分析

根据2024年1-11月销售数据，中国新能源汽车市场TOP10企业市场份额如下：

1. 比亚迪：31.8%（224.7万辆）
2. 特斯拉中国：7.2%（50.9万辆）
3. 吉利汽车：5.4%（38.2万辆）
4. 广汽埃安：4.8%（33.9万辆）
5. 蔚来汽车：3.1%（21.9万辆）
...
```

**Human Feedback (revise:logic):**
```python
state["review_decision"] = "revise:logic"
state["human_feedback"] = {
    "instruction": "Add analysis comparing BYD's vertical integration strategy vs Tesla's technology-first approach",
    "focus_area": "competitive_strategy",
    "comparison_aspects": ["supply_chain", "R&D", "market_positioning"]
}
```

**Final Approval:**
```python
state["review_decision"] = "approve"
```

---

## Troubleshooting

### Issue: "API key not found" error

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Verify API key is set
echo $DEEPSEEK_API_KEY

# If not set, add to .env:
echo "DEEPSEEK_API_KEY=your_key_here" >> .env
```

### Issue: "Milvus connection failed"

**Solution:**
```bash
# Check Milvus is running
docker ps | grep milvus

# If not running, start Milvus:
docker-compose -f docker-compose-milvus.yml up -d

# Check connection
python -c "from rag_project.storage.milvus_manager import MilvusManager; m = MilvusManager(); print(m.connection.get_server_version())"
```

### Issue: Workflow stuck at human_review

**Solution:**
```python
# The workflow is designed to interrupt here
# You must provide review_decision to continue

state["review_decision"] = "approve"  # or other decision
state = app.invoke(state, config=config)
```

### Issue: Generated report has poor quality

**Solution:**
1. Use interactive mode to review each chapter
2. Provide specific feedback for revisions
3. Improve knowledge base with better documents
4. Adjust agent temperature settings
5. Customize system prompts in config

---

## Next Steps

- Explore the [Architecture Documentation](AGENT_ARCHITECTURE.md) for deeper technical details
- Check the code in `rag_project/agent/` for implementation details
- Review configuration options in `config/agent_config.yaml`
- Run example scripts in `examples/` directory

For issues or questions, check the logs in `logs/agent.log` or consult the project README.
