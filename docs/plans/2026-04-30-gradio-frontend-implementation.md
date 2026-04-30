# Gradio Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> **Design Skill:** Apply frontend-design aesthetics throughout — Executive Intelligence Dashboard style.

**Goal:** Build a Gradio frontend for the multi-agent RAG report generation system with split-panel workbench layout, full scratchpad transparency, and human-in-the-loop review.

**Architecture:** Backend adapter wraps LangGraph graph with pause/resume via Gradio events. Frontend uses Gradio Blocks with custom CSS for the Executive Intelligence Dashboard aesthetic. State managed through `gr.State()` + LangGraph `thread_id`.

**Tech Stack:** Gradio 4.x, Python 3.10+, custom CSS, LangGraph interrupt mechanism

---

## Task 1: Backend Adapter — WorkflowService

**Files:**
- Create: `rag_project/agent/workflow_service.py`
- Read: `rag_project/agent/graph.py`, `rag_project/agent/state.py`, `rag_project/agent/nodes/human_review.py`

**Step 1: Create WorkflowService class skeleton**

```python
"""Bridge between LangGraph workflow and Gradio frontend."""
import threading
from typing import Dict, Any, Optional, Generator
from langgraph.types import Command

from rag_project.agent.graph import create_report_graph


class WorkflowService:
    """Manages LangGraph workflow lifecycle for the Gradio frontend."""

    def __init__(self):
        self.graph = create_report_graph()
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def start_report(self, user_input: str, thread_id: str) -> Dict[str, Any]:
        """Start a new report generation, pause at first human_review."""
        ...

    def get_current_state(self, thread_id: str) -> Dict[str, Any]:
        """Get the current workflow state snapshot for UI rendering."""
        ...

    def submit_review(self, thread_id: str, decision: str, feedback: str = "") -> Dict[str, Any]:
        """Submit human review decision and resume workflow."""
        ...

    def get_report(self, thread_id: str) -> Optional[str]:
        """Get the final report if generation is complete."""
        ...

    def list_reports(self) -> list:
        """List all saved reports."""
        ...
```

**Step 2: Implement start_report**

Read `graph.py` to understand how `graph.stream()` works with `interrupt_before`. Implement:

```python
def start_report(self, user_input: str, thread_id: str) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"user_input": user_input}

    for event in self.graph.stream(initial_state, config=config):
        pass  # Stream until interrupt at human_review

    snapshot = self.graph.get_state(config)
    self._sessions[thread_id] = {
        "config": config,
        "snapshot": snapshot,
        "status": "waiting_review",
    }
    return self._extract_ui_state(snapshot, thread_id)
```

**Step 3: Implement _extract_ui_state helper**

Convert LangGraph state snapshot into UI-friendly structure:

```python
def _extract_ui_state(self, snapshot, thread_id: str) -> Dict[str, Any]:
    state = snapshot.values
    return {
        "thread_id": thread_id,
        "status": self._sessions[thread_id]["status"],
        "current_chapter_index": state.get("current_chapter_index", 0),
        "global_plan": state.get("global_plan", []),
        "current_phase": state.get("current_phase", "diagnosis"),
        "chapter_title": state.get("chapter_title", ""),
        "chapter_question": state.get("chapter_question", ""),
        "current_draft": state.get("current_draft", ""),
        "chapter_scratchpad": state.get("chapter_scratchpad", {}),
        "context_pool": state.get("context_pool", []),
        "strategic_blueprint": state.get("strategic_blueprint"),
        "is_blueprint_phase": self._is_blueprint_phase(snapshot),
        "final_report": state.get("final_report"),
        "review_decision": state.get("review_decision", ""),
        "human_feedback": state.get("human_feedback", {}),
    }
```

**Step 4: Implement submit_review**

```python
def submit_review(self, thread_id: str, decision: str, feedback: str = "") -> Dict[str, Any]:
    session = self._sessions[thread_id]
    config = session["config"]

    # Map UI decision to LangGraph review_decision
    feedback_data = {"decision": decision, "comment": feedback}

    self.graph.update_state(
        config,
        {"review_decision": decision, "human_feedback": feedback_data},
    )

    # Resume graph execution
    for event in self.graph.stream(None, config=config):
        pass

    snapshot = self.graph.get_state(config)
    session["snapshot"] = snapshot

    # Check if workflow completed
    if snapshot.values.get("final_report"):
        session["status"] = "completed"
    else:
        session["status"] = "waiting_review"

    return self._extract_ui_state(snapshot, thread_id)
```

**Step 5: Implement _is_blueprint_phase helper**

```python
def _is_blueprint_phase(self, snapshot) -> bool:
    state = snapshot.values
    blueprint = state.get("strategic_blueprint")
    if not blueprint:
        return False
    return not blueprint.get("approved", False) and state.get("current_chapter_index", 0) >= 2
```

**Step 6: Commit**

```bash
git add rag_project/agent/workflow_service.py
git commit -m "feat: add WorkflowService backend adapter for Gradio frontend"
```

---

## Task 2: Custom CSS Theme

**Files:**
- Create: `rag_project/agent/frontend/theme.css`
- Create: `rag_project/agent/frontend/__init__.py`

**Step 1: Write the Executive Intelligence Dashboard CSS**

Key design elements:
- Color palette: deep navy `#1a2332`, amber gold `#d4a574`, warm grays
- Paper texture background via subtle noise
- Milestone progress bar with pulse animation
- Card shadows and borders with warm tones
- Status colors: green completed, amber active, gray pending

```css
/* === Executive Intelligence Dashboard Theme === */

:root {
  --bg-primary: #1a2332;
  --bg-secondary: #1e2a3a;
  --bg-card: #243044;
  --bg-card-hover: #2a3750;
  --text-primary: #e8e0d8;
  --text-secondary: #9ca3af;
  --text-muted: #6b7280;
  --accent-gold: #d4a574;
  --accent-gold-light: #e8c9a0;
  --accent-gold-dim: #a07850;
  --status-done: #4ade80;
  --status-active: #d4a574;
  --status-pending: #6b7280;
  --danger: #ef4444;
  --border: #2d3d52;
  --border-light: #3a4d66;
  --shadow: rgba(0, 0, 0, 0.3);
}

/* Global overrides */
.gradio-container {
  font-family: 'Sora', 'Source Han Sans SC', 'Noto Sans SC', sans-serif !important;
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* Paper texture overlay */
.gradio-container::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  opacity: 0.03;
  background-image: url("data:image/svg+xml,..."); /* noise pattern */
  pointer-events: none;
  z-index: 0;
}

/* === Milestone Progress Bar === */
.milestone-node {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  transition: all 0.3s ease;
  cursor: pointer;
}

.milestone-node:hover {
  background: var(--bg-card-hover);
}

.milestone-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  flex-shrink: 0;
}

.milestone-done .milestone-icon {
  background: var(--status-done);
  color: var(--bg-primary);
}

.milestone-active .milestone-icon {
  background: var(--accent-gold);
  color: var(--bg-primary);
  animation: pulse-glow 2s ease-in-out infinite;
}

.milestone-pending .milestone-icon {
  background: var(--status-pending);
  opacity: 0.5;
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px var(--accent-gold-dim); }
  50% { box-shadow: 0 0 20px var(--accent-gold); }
}

/* === Cards === */
.doc-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 12px;
  transition: all 0.2s ease;
}

.doc-card:hover {
  border-color: var(--accent-gold-dim);
  box-shadow: 0 4px 16px var(--shadow);
}

/* === Action Buttons === */
.btn-approve {
  background: var(--status-done) !important;
  color: var(--bg-primary) !important;
  border: none !important;
  font-weight: 600 !important;
}

.btn-revise {
  background: var(--accent-gold) !important;
  color: var(--bg-primary) !important;
  border: none !important;
}

.btn-danger {
  background: transparent !important;
  color: var(--danger) !important;
  border: 1px solid var(--danger) !important;
}

/* === Score bars === */
.score-bar {
  height: 8px;
  border-radius: 4px;
  background: var(--border);
  overflow: hidden;
}

.score-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

/* === SWOT Matrix === */
.swot-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.swot-cell {
  padding: 16px;
  border-radius: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
}

/* === Blueprint Pillars === */
.pillar-card {
  background: var(--bg-card);
  border: 1px solid var(--accent-gold-dim);
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s ease;
}

.pillar-card:hover {
  border-color: var(--accent-gold);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px var(--shadow);
}
```

**Step 2: Commit**

```bash
git add rag_project/agent/frontend/
git commit -m "feat: add Executive Intelligence Dashboard CSS theme"
```

---

## Task 3: Progress Panel Component

**Files:**
- Create: `rag_project/agent/frontend/progress_panel.py`

**Step 1: Implement progress panel HTML renderer**

```python
"""Left sidebar: milestone progress tracker + draft library."""
from typing import List, Dict, Any


CHAPTER_META = [
    {"index": 0, "title": "宏观政策环境与时代要求", "model": "PEST", "phase": "diagnosis"},
    {"index": 1, "title": "区域战略与'交通强省'建设剖析", "model": "区域分析", "phase": "diagnosis"},
    {"index": 2, "title": "行业演进趋势与当前内部诊断", "model": "SWOT", "phase": "diagnosis"},
    {"index": 3, "title": "总体战略思路与政策响应目标", "model": "BSC", "phase": "initiatives"},
    {"index": 4, "title": "主责主业：高质量建设与保通保畅举措", "model": "BCG", "phase": "initiatives"},
    {"index": 5, "title": "创新驱动：绿色低碳与智慧交投建设", "model": "Ansoff", "phase": "initiatives"},
    {"index": 6, "title": "产业协同：交旅融合与服务地方经济", "model": "ESG", "phase": "initiatives"},
    {"index": 7, "title": "治理效能：深化国企改革与党建引领", "model": "McKinsey 7S", "phase": "initiatives"},
]


def render_progress_panel(
    current_index: int,
    phase: str,
    context_pool: List[str],
    is_blueprint_phase: bool,
    blueprint_approved: bool,
) -> str:
    """Render left sidebar HTML with milestone progress."""

    completed_count = len(context_pool)

    # Build milestone nodes
    diagnosis_nodes = ""
    initiatives_nodes = ""
    for ch in CHAPTER_META:
        idx = ch["index"]
        if idx < current_index and not (idx == current_index):
            status = "done"
            icon = "✅"
        elif idx == current_index and not is_blueprint_phase:
            status = "active"
            icon = "🔵"
        else:
            status = "pending"
            icon = "○"

        node = f"""
        <div class="milestone-node milestone-{status}">
            <div class="milestone-icon">{icon}</div>
            <div class="milestone-info">
                <div class="milestone-title">Ch{idx+1} {ch['title'][:12]}...</div>
                <div class="milestone-meta">{ch['model']} · {_status_text(status)}</div>
            </div>
        </div>
        """
        if ch["phase"] == "diagnosis":
            diagnosis_nodes += node
        else:
            initiatives_nodes += node

    # Blueprint node
    if is_blueprint_phase:
        bp_status = "active"
        bp_icon = "🔶"
    elif blueprint_approved:
        bp_status = "done"
        bp_icon = "✅"
    else:
        bp_status = "pending"
        bp_icon = "◇"

    blueprint_node = f"""
    <div class="milestone-node milestone-{bp_status}">
        <div class="milestone-icon">{bp_icon}</div>
        <div class="milestone-info">
            <div class="milestone-title">战略蓝图审核</div>
            <div class="milestone-meta">{_status_text(bp_status)}</div>
        </div>
    </div>
    """

    # Progress bar
    progress_pct = int((completed_count / 8) * 100)
    progress_bar = f"""
    <div class="progress-section">
        <div class="progress-label">📈 总进度: {progress_pct}%</div>
        <div class="score-bar">
            <div class="score-bar-fill" style="width:{progress_pct}%; background:var(--accent-gold);"></div>
        </div>
        <div class="progress-count">{completed_count}/8 章</div>
    </div>
    """

    # Draft library
    draft_items = ""
    for i, chapter_text in enumerate(context_pool):
        title = CHAPTER_META[i]["title"][:15] if i < len(CHAPTER_META) else f"章节{i+1}"
        draft_items += f"""
        <div class="draft-item">
            <span>▸ {title}...</span>
            <span style="color:var(--status-done)">✅</span>
        </div>
        """

    return f"""
    <div class="progress-panel">
        <div class="panel-header">
            <div class="panel-title">📊 战略报告生成</div>
        </div>

        <div class="phase-label">── 诊断阶段 ──────────</div>
        {diagnosis_nodes}

        {blueprint_node}

        <div class="phase-label">── 举措阶段 ──────────</div>
        {initiatives_nodes}

        {progress_bar}

        <div class="phase-label">── 报告草稿库 ──────────</div>
        {draft_items if draft_items else '<div class="text-muted">暂无已完成章节</div>'}
    </div>
    """
```

**Step 2: Implement _status_text helper**

```python
def _status_text(status: str) -> str:
    return {"done": "已完成", "active": "生成中...", "pending": "待处理"}.get(status, "未知")
```

**Step 3: Commit**

```bash
git add rag_project/agent/frontend/progress_panel.py
git commit -m "feat: add progress panel component with milestone rendering"
```

---

## Task 4: Workspace Tabs Components

**Files:**
- Create: `rag_project/agent/frontend/workspace_tabs.py`

**Step 1: Implement draft tab renderer**

```python
"""Right workspace: tab content renderers for scratchpad data."""
from typing import Dict, Any, List, Optional


def render_draft_tab(draft: str, metadata: Optional[Dict] = None) -> str:
    """Render chapter draft with metadata footer."""
    meta_html = ""
    if metadata:
        word_count = metadata.get("word_count", 0)
        gen_time = metadata.get("generation_time", 0)
        citations = metadata.get("citation_count", 0)
        meta_html = f"""
        <div class="draft-meta">
            📊 字数: {word_count} · 🕐 生成耗时: {gen_time}s · 📖 引用来源: {citations}处
        </div>
        """
    return f"""
    <div class="draft-content">
        <div class="markdown-body">{draft}</div>
        {meta_html}
    </div>
    """
```

**Step 2: Implement retrieved docs tab renderer**

```python
def render_docs_tab(docs: List[Dict]) -> str:
    """Render retrieved document cards with relevance scores."""
    if not docs:
        return '<div class="text-muted">暂无检索文档</div>'

    cards = ""
    for i, doc in enumerate(docs):
        score = doc.get("score", 0)
        title = doc.get("title", f"文档 #{i+1}")
        source = doc.get("source", "未知来源")
        content = doc.get("content", "")
        score_pct = int(score * 100)
        score_color = "#4ade80" if score > 0.85 else "#d4a574" if score > 0.7 else "#6b7280"

        cards += f"""
        <div class="doc-card">
            <div class="doc-header">
                <span class="doc-title">📄 {title}</span>
                <span class="doc-score" style="color:{score_color}">相关度: {score_pct}%</span>
            </div>
            <div class="doc-source">来源: {source}</div>
            <div class="doc-content">{content[:300]}{'...' if len(content) > 300 else ''}</div>
        </div>
        """

    return f'<div class="docs-list">{cards}</div>'
```

**Step 3: Implement analysis tab renderer**

```python
def render_analysis_tab(scratchpad: Dict) -> str:
    """Render analyst reasoning process."""
    analysis = scratchpad.get("analyst", {})
    if not analysis:
        return '<div class="text-muted">分析师尚未开始工作</div>'

    questions = analysis.get("questions", [])
    facts = analysis.get("key_facts", [])
    reasoning = analysis.get("reasoning_chain", [])
    insights = analysis.get("insights", [])

    questions_html = "".join(f"<li>{q}</li>" for q in questions) if questions else "<li>待生成</li>"
    facts_html = "".join(
        f'<div class="fact-item">• 事实#{i+1}: {f.get("content", "")} <span class="fact-source">来源: {f.get("source", "N/A")}</span></div>'
        for i, f in enumerate(facts)
    ) if facts else '<div class="text-muted">待提取</div>'

    reasoning_html = "".join(
        f'<div class="reasoning-step"><strong>Step {i+1}:</strong> {step}</div>'
        for i, step in enumerate(reasoning)
    ) if reasoning else '<div class="text-muted">待推理</div>'

    insights_html = "".join(f"<li>{ins}</li>" for ins in insights) if insights else "<li>待生成</li>"

    return f"""
    <div class="analysis-panel">
        <div class="analysis-section">
            <h4>【核心问题拆解】</h4>
            <ol>{questions_html}</ol>
        </div>
        <div class="analysis-section">
            <h4>【关键事实提取】</h4>
            {facts_html}
        </div>
        <div class="analysis-section">
            <h4>【分析推理链】</h4>
            <div class="reasoning-chain">{reasoning_html}</div>
        </div>
        <div class="analysis-section">
            <h4>💡 洞察要点</h4>
            <ul>{insights_html}</ul>
        </div>
    </div>
    """
```

**Step 4: Implement review tab renderer**

```python
def render_review_tab(review: Dict) -> str:
    """Render LLM review with dimension scores and issues."""
    if not review:
        return '<div class="text-muted">LLM评审尚未完成</div>'

    total_score = review.get("total_score", 0)
    dimensions = review.get("dimensions", {})
    issues = review.get("issues", [])

    # Score bars
    dim_html = ""
    for dim_name, dim_data in dimensions.items():
        score = dim_data.get("score", 0)
        warning = " ⚠️" if score < 75 else ""
        dim_html += f"""
        <div class="score-row">
            <span class="dim-name">{dim_name}</span>
            <div class="score-bar">
                <div class="score-bar-fill" style="width:{score}%; background:{_score_color(score)};"></div>
            </div>
            <span class="dim-score">{score}{warning}</span>
        </div>
        """

    # Issues
    issues_html = ""
    for i, issue in enumerate(issues):
        issues_html += f"""
        <div class="issue-card">
            <div class="issue-header">问题 #{i+1}: {issue.get('title', '')}</div>
            <div class="issue-location">位置: {issue.get('location', '未知')}</div>
            <div class="issue-desc">{issue.get('description', '')}</div>
            <div class="issue-suggestion">建议: {issue.get('suggestion', '')}</div>
        </div>
        """

    return f"""
    <div class="review-panel">
        <div class="review-summary">
            <div class="total-score">
                <span class="score-number">{total_score}</span>
                <span class="score-label">/100</span>
            </div>
        </div>
        <div class="dimension-scores">{dim_html}</div>
        <div class="issues-section">
            <h4>⚠️ 发现的问题</h4>
            {issues_html if issues_html else '<div class="text-muted">未发现严重问题</div>'}
        </div>
    </div>
    """


def _score_color(score: int) -> str:
    if score >= 85:
        return "#4ade80"
    if score >= 70:
        return "#d4a574"
    return "#ef4444"
```

**Step 5: Implement history tab renderer**

```python
def render_history_tab(history: List[Dict]) -> str:
    """Render version history timeline."""
    if not history:
        return '<div class="text-muted">本章尚未发生修改</div>'

    entries = ""
    for i, entry in enumerate(history):
        is_current = entry.get("is_current", False)
        version_class = "history-current" if is_current else "history-past"
        marker = "🔵" if is_current else f"版本 #{i+1}"

        entries += f"""
        <div class="history-entry {version_class}">
            <div class="history-header">{marker} · {entry.get('label', '')}</div>
            <div class="history-meta">
                时间: {entry.get('timestamp', 'N/A')}<br>
                操作: {entry.get('action', 'N/A')}<br>
                评审: {entry.get('score', 'N/A')}分
            </div>
        </div>
        """

    return f'<div class="history-timeline">{entries}</div>'
```

**Step 6: Commit**

```bash
git add rag_project/agent/frontend/workspace_tabs.py
git commit -m "feat: add workspace tab renderers for draft, docs, analysis, review, history"
```

---

## Task 5: Blueprint Review Component

**Files:**
- Create: `rag_project/agent/frontend/blueprint_panel.py`

**Step 1: Implement blueprint review renderer**

```python
"""Blueprint review page: structured display of strategic blueprint."""
from typing import Dict, Any, Optional


def render_blueprint_review(blueprint: Dict) -> str:
    """Render the full strategic blueprint for review."""
    if not blueprint:
        return '<div class="text-muted">蓝图数据不可用</div>'

    mission = blueprint.get("mission", "待定义")
    swot = blueprint.get("swot", {})
    tows = blueprint.get("tows", {})
    pillars = blueprint.get("pillars", [])
    kpis = blueprint.get("kpis", {})

    # Mission
    mission_html = f"""
    <div class="blueprint-section">
        <h3>🎯 企业使命</h3>
        <div class="mission-statement">{mission}</div>
    </div>
    """

    # SWOT matrix
    swot_html = f"""
    <div class="blueprint-section">
        <h3>📊 SWOT 分析矩阵</h3>
        <div class="swot-grid">
            <div class="swot-cell swot-strength">
                <h4>✅ 优势 (S)</h4>
                {_render_list(swot.get("strengths", []))}
            </div>
            <div class="swot-cell swot-opportunity">
                <h4>🚀 机会 (O)</h4>
                {_render_list(swot.get("opportunities", []))}
            </div>
            <div class="swot-cell swot-weakness">
                <h4>⚠️ 劣势 (W)</h4>
                {_render_list(swot.get("weaknesses", []))}
            </div>
            <div class="swot-cell swot-threat">
                <h4>🔥 威胁 (T)</h4>
                {_render_list(swot.get("threats", []))}
            </div>
        </div>
    </div>
    """

    # Strategic pillars
    pillar_cards = "".join(
        f'<div class="pillar-card"><div class="pillar-name">{p.get("name", "")}</div>'
        f'<div class="pillar-desc">{p.get("description", "")}</div></div>'
        for p in pillars
    )
    pillars_html = f"""
    <div class="blueprint-section">
        <h3>🏛️ 战略支柱</h3>
        <div class="pillar-grid">{pillar_cards}</div>
    </div>
    """

    # KPIs
    kpi_rows = ""
    for dim, indicators in kpis.items():
        if isinstance(indicators, list):
            kpi_rows += f"<tr><td>{dim}</td><td>{'、'.join(indicators)}</td></tr>"
        else:
            kpi_rows += f"<tr><td>{dim}</td><td>{indicators}</td></tr>"

    kpis_html = f"""
    <div class="blueprint-section">
        <h3>📈 关键指标 (BSC框架)</h3>
        <table class="kpi-table">
            <thead><tr><th>维度</th><th>指标</th></tr></thead>
            <tbody>{kpi_rows}</tbody>
        </table>
    </div>
    """

    return f"""
    <div class="blueprint-review">
        <div class="blueprint-header">
            <h2>🔷 战略蓝图审核</h2>
            <p>基于前三章诊断结果自动生成，审核通过后将为Ch4-8提供战略约束</p>
        </div>
        {mission_html}
        {swot_html}
        {pillars_html}
        {kpis_html}
    </div>
    """


def _render_list(items: list) -> str:
    return "".join(f"<div class='list-item'>• {item}</div>" for item in items)
```

**Step 2: Commit**

```bash
git add rag_project/agent/frontend/blueprint_panel.py
git commit -m "feat: add blueprint review panel with SWOT, pillars, KPIs"
```

---

## Task 6: Report Viewer Component

**Files:**
- Create: `rag_project/agent/frontend/report_viewer.py`

**Step 1: Implement report viewer renderer**

```python
"""Report viewing and export: final report display with TOC."""
from typing import List, Dict


def render_report_viewer(report: str, chapters: List[str]) -> str:
    """Render final report with sidebar TOC."""
    toc_items = ""
    chapter_titles = [
        "摘要", "Ch1: 宏观政策环境", "Ch2: 区域战略分析",
        "Ch3: 行业趋势诊断", "🔷 战略蓝图", "Ch4: 总体战略思路",
        "Ch5: 主责主业举措", "Ch6: 创新驱动", "Ch7: 产业协同", "Ch8: 治理效能",
    ]
    for i, title in enumerate(chapter_titles):
        toc_items += f'<div class="toc-item" data-chapter="{i}">{title}</div>'

    return f"""
    <div class="report-viewer">
        <div class="report-toc">
            <div class="toc-header">📋 目录</div>
            {toc_items}
        </div>
        <div class="report-content">
            <div class="markdown-body">{report}</div>
        </div>
    </div>
    """


def render_report_card(report_meta: Dict) -> str:
    """Render a single report card for history list."""
    status = report_meta.get("status", "unknown")
    status_icon = "✅" if status == "completed" else "⏸️"
    status_text = "已完成" if status == "completed" else "已终止"

    return f"""
    <div class="report-card">
        <div class="report-card-title">📄 {report_meta.get("title", "未命名报告")}</div>
        <div class="report-card-meta">
            生成时间: {report_meta.get("created_at", "N/A")} ·
            状态: {status_icon} {status_text} ·
            章节: {report_meta.get("chapters", "0")}/8 ·
            字数: {report_meta.get("word_count", "0")}
        </div>
    </div>
    """
```

**Step 2: Commit**

```bash
git add rag_project/agent/frontend/report_viewer.py
git commit -m "feat: add report viewer with TOC and history cards"
```

---

## Task 7: Main Gradio App Assembly

**Files:**
- Create: `rag_project/agent/frontend/app.py`

**Step 1: Build the main Gradio app with full layout**

This is the core file that assembles all components into the split-panel workbench.

```python
"""Gradio frontend: Executive Intelligence Dashboard for report generation."""
import uuid
import gradio as gr

from rag_project.agent.workflow_service import WorkflowService
from rag_project.agent.frontend.progress_panel import render_progress_panel
from rag_project.agent.frontend.workspace_tabs import (
    render_draft_tab, render_docs_tab, render_analysis_tab,
    render_review_tab, render_history_tab,
)
from rag_project.agent.frontend.blueprint_panel import render_blueprint_review
from rag_project.agent.frontend.report_viewer import render_report_viewer, render_report_card


def create_app():
    service = WorkflowService()

    # Load CSS
    with open("rag_project/agent/frontend/theme.css", encoding="utf-8") as f:
        css = f.read()

    with gr.Blocks(
        title="战略报告生成系统",
        css=css,
        theme=gr.themes.Base(),
    ) as app:
        # === State ===
        thread_id = gr.State(value=lambda: str(uuid.uuid4()))
        ui_state = gr.State(value={})

        # === Top Nav ===
        gr.HTML("""
        <div class="top-nav">
            <div class="nav-logo">📊 江西交投战略报告生成系统</div>
            <div class="nav-actions">
                <button class="nav-btn" id="btn-new">新建报告</button>
                <button class="nav-btn" id="btn-history">历史报告</button>
            </div>
        </div>
        """)

        # === Welcome Page ===
        with gr.Group(visible=True) as welcome_page:
            gr.Markdown("# 战略报告生成系统")
            topic_input = gr.Textbox(label="报告主题", placeholder="例如：江西交投2026年度战略规划报告")
            btn_start = gr.Button("🚀 开始生成", variant="primary")

        # === Main Workbench ===
        with gr.Group(visible=False) as workbench:
            with gr.Row():
                # Left panel (20%)
                with gr.Column(scale=1):
                    progress_html = gr.HTML()

                # Right panel (80%)
                with gr.Column(scale=4):
                    # Chapter header
                    chapter_header = gr.HTML()

                    # Workspace tabs (dynamic visibility)
                    with gr.Tabs() as workspace_tabs:
                        with gr.Tab("📄 草稿") as tab_draft:
                            draft_html = gr.HTML()
                        with gr.Tab("📚 检索文档") as tab_docs:
                            docs_html = gr.HTML()
                        with gr.Tab("🔍 分析过程") as tab_analysis:
                            analysis_html = gr.HTML()
                        with gr.Tab("⭐ 评审详情") as tab_review:
                            review_html = gr.HTML()
                        with gr.Tab("📜 修改历史") as tab_history:
                            history_html = gr.HTML()

                    # Blueprint review (hidden by default)
                    blueprint_html = gr.HTML(visible=False)

                    # LLM review summary bar
                    review_summary = gr.HTML()

                    # Action buttons
                    with gr.Row():
                        btn_approve = gr.Button("✅ 批准通过", elem_classes=["btn-approve"])
                        btn_revise_data = gr.Button("🔄 补充数据", elem_classes=["btn-revise"])
                        btn_revise_logic = gr.Button("📊 重新分析", elem_classes=["btn-revise"])
                        btn_revise_writing = gr.Button("✍️ 重写内容", elem_classes=["btn-revise"])

                    with gr.Row():
                        feedback_input = gr.Textbox(
                            label="💬 补充说明 (可选)",
                            placeholder="给Agent的额外修改指令...",
                            lines=2,
                        )

                    with gr.Row():
                        btn_stop = gr.Button("❌ 终止报告", elem_classes=["btn-danger"])

                    # Processing indicator
                    processing_html = gr.HTML(visible=False)

        # === Completion Page ===
        with gr.Group(visible=False) as completion_page:
            completion_html = gr.HTML()
            with gr.Row():
                btn_view_report = gr.Button("📄 查看报告")
                btn_download = gr.Button("📥 下载报告")
                btn_new_report = gr.Button("🆕 新建报告")

        # === Report Viewer Page ===
        with gr.Group(visible=False) as report_page:
            report_html = gr.HTML()
            btn_back = gr.Button("← 返回")

        # === History Page ===
        with gr.Group(visible=False) as history_page:
            history_list_html = gr.HTML()
            btn_back_from_history = gr.Button("← 返回")

        # === Event Handlers ===

        def start_generation(topic, tid):
            state = service.start_report(topic, tid)
            return _render_workbench(state)

        btn_start.click(
            fn=start_generation,
            inputs=[topic_input, thread_id],
            outputs=[welcome_page, workbench, completion_page, report_page, history_page,
                     progress_html, chapter_header, draft_html, docs_html,
                     analysis_html, review_html, history_html, review_summary,
                     blueprint_html, ui_state],
        )

        def submit_decision(decision, feedback, state, tid):
            new_state = service.submit_review(tid, decision, feedback)
            if new_state["status"] == "completed":
                return _show_completion(new_state)
            return _render_workbench(new_state)

        for btn, decision in [
            (btn_approve, "approve"),
            (btn_revise_data, "revise:data"),
            (btn_revise_logic, "revise:logic"),
            (btn_revise_writing, "revise:writing"),
            (btn_stop, "finished"),
        ]:
            btn.click(
                fn=lambda d=decision, f=feedback_input, s=ui_state, t=thread_id:
                    submit_decision(d, f, s, t),
                inputs=[feedback_input, ui_state, thread_id],
                outputs=[welcome_page, workbench, completion_page, report_page, history_page,
                         progress_html, chapter_header, draft_html, docs_html,
                         analysis_html, review_html, history_html, review_summary,
                         blueprint_html, ui_state],
            )

    return app


def _render_workbench(state: dict) -> dict:
    """Update all workbench components from current state."""
    # ... (wiring logic that maps state to each HTML component)
    pass


def _show_completion(state: dict) -> dict:
    """Switch to completion page."""
    pass
```

**Step 2: Commit**

```bash
git add rag_project/agent/frontend/app.py
git commit -m "feat: add main Gradio app with split-panel workbench layout"
```

---

## Task 8: Startup Script

**Files:**
- Create: `scripts/run_frontend.py`

**Step 1: Create launch script**

```python
"""Launch the Gradio frontend for report generation."""
from rag_project.agent.frontend.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
```

**Step 2: Commit**

```bash
git add scripts/run_frontend.py
git commit -m "feat: add Gradio frontend launch script"
```

---

## Task 9: Integration Testing

**Step 1: Write integration test for WorkflowService**

Create `tests/test_workflow_service.py`:

```python
"""Test WorkflowService backend adapter."""
import pytest
from rag_project.agent.workflow_service import WorkflowService


def test_service_init():
    service = WorkflowService()
    assert service.graph is not None


def test_start_report_creates_session():
    service = WorkflowService()
    state = service.start_report("测试报告", "test-thread-1")
    assert state["status"] == "waiting_review"
    assert "test-thread-1" in service._sessions


def test_get_current_state():
    service = WorkflowService()
    service.start_report("测试报告", "test-thread-2")
    state = service.get_current_state("test-thread-2")
    assert "current_chapter_index" in state


def test_submit_review():
    service = WorkflowService()
    service.start_report("测试报告", "test-thread-3")
    state = service.submit_review("test-thread-3", "approve", "")
    assert state["status"] in ("waiting_review", "completed")
```

**Step 2: Run tests**

```bash
pytest tests/test_workflow_service.py -v
```

**Step 3: Fix any issues, then commit**

```bash
git add tests/test_workflow_service.py
git commit -m "test: add integration tests for WorkflowService"
```

---

## Task 10: Polish & Final Integration

**Step 1: Test full workflow manually**

```bash
python scripts/run_frontend.py
```

Verify:
- Welcome page → input topic → workbench appears
- Left progress panel updates with chapter status
- Tabs unlock progressively
- Blueprint review page appears after Ch3
- Action buttons route correctly
- Completion page shows after Ch8
- Report viewer displays final report

**Step 2: Polish CSS animations and transitions**

Refine pulse animation timing, hover effects, card transitions. Add loading spinners for processing states.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Gradio frontend with Executive Intelligence Dashboard theme"
```
