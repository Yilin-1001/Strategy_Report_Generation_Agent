"""Right workspace: tab content renderers for scratchpad data."""
import re
from typing import Dict, Any, List, Optional


def _markdown_to_html(text: str) -> str:
    """Convert basic Markdown to HTML without external dependencies.

    Handles: headers (#, ##, ###), bold (**), italic (*), lists (-, *), paragraphs.
    """
    if not text:
        return ""

    lines = text.split("\n")
    html_parts = []
    in_list = False
    list_items = []

    for line in lines:
        stripped = line.strip()

        # Headers
        if stripped.startswith("### "):
            if in_list:
                html_parts.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
                in_list = False
            html_parts.append(f'<h3>{stripped[4:]}</h3>')
        elif stripped.startswith("## "):
            if in_list:
                html_parts.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
                in_list = False
            html_parts.append(f'<h2>{stripped[3:]}</h2>')
        elif stripped.startswith("# "):
            if in_list:
                html_parts.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
                in_list = False
            html_parts.append(f'<h1>{stripped[2:]}</h1>')
        # List items
        elif stripped.startswith("- ") or stripped.startswith("* "):
            in_list = True
            item_text = stripped[2:]
            # Bold and italic within list item
            item_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item_text)
            item_text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', item_text)
            list_items.append(f'<li>{item_text}</li>')
        # Empty line - paragraph break
        elif not stripped:
            if in_list:
                html_parts.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
                in_list = False
            html_parts.append("")  # Will be collapsed
        # Regular paragraph
        else:
            if in_list:
                html_parts.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
                in_list = False
            # Bold and italic
            para = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
            para = re.sub(r'\*(.+?)\*', r'<em>\1</em>', para)
            html_parts.append(f'<p>{para}</p>')

    # Close any remaining list
    if in_list:
        html_parts.append("<ul>" + "".join(list_items) + "</ul>")

    return "\n".join(html_parts)


def render_draft_tab(draft: str, metadata: Optional[Dict] = None) -> str:
    """Render chapter draft with metadata footer."""
    if not draft:
        return '<div class="text-muted">草稿尚未生成，请等待Writer完成</div>'

    # Convert markdown to HTML for proper rendering
    draft_html = _markdown_to_html(draft)

    meta_html = ""
    if metadata:
        word_count = metadata.get("word_count", 0)
        gen_time = metadata.get("generation_time", 0)
        citations = metadata.get("citation_count", 0)
        meta_html = f"""
        <div class="draft-meta">
            <span>字数: {word_count}</span>
            <span>生成耗时: {gen_time}s</span>
            <span>引用来源: {citations}处</span>
        </div>
        """
    return f"""
    <div class="draft-content">
        <div class="markdown-body markdown-light">{draft_html}</div>
        {meta_html}
    </div>
    """


def render_docs_tab(docs: List[Dict]) -> str:
    """Render retrieved document cards with relevance scores.

    Displays document title (name) and category (knowledge base folder source).
    """
    if not docs:
        return '<div class="text-muted">暂无检索文档，请等待Researcher完成</div>'

    cards = ""
    for i, doc in enumerate(docs):
        # Handle both dict and tuple/list formats
        if isinstance(doc, dict):
            score = doc.get("score", 0)
            # Access nested metadata (actual data structure from retriever)
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "") or doc.get("title", f"文档 #{i + 1}")
            # Use category as the source display (knowledge base folder name)
            category = metadata.get("category", "")
            # Fallback: if category is empty, use source (filename) as fallback
            source_display = category if category else metadata.get("source", "未知来源")
            content = doc.get("text", "") or doc.get("content", "")
        else:
            # Fallback for non-dict format
            score = 0.8
            title = f"文档 #{i + 1}"
            source_display = "知识库"
            content = str(doc)[:500]

        score_pct = int(score * 100) if isinstance(score, (int, float)) else 80
        score_color = "#4ade80" if score_pct > 85 else "#d4a574" if score_pct > 70 else "#6b7280"

        cards += f"""
        <div class="doc-card">
            <div class="doc-header">
                <span class="doc-title">&#128196; {title}</span>
                <span class="doc-score" style="color:{score_color}">相关度: {score_pct}%</span>
            </div>
            <div class="doc-source">来源: {source_display}</div>
            <div class="doc-content">{content[:300]}{'...' if len(content) > 300 else ''}</div>
        </div>
        """

    return f'<div class="docs-list">{cards}</div>'


def render_analysis_tab(scratchpad: Dict) -> str:
    """Render analyst reasoning process."""
    if not scratchpad:
        return '<div class="text-muted">分析师尚未开始工作，请等待Analyst完成</div>'

    # Data is at root level, not nested under "analyst"
    key_facts = scratchpad.get("key_facts", {})
    insights = scratchpad.get("insights", [])
    document_summary = scratchpad.get("document_summary", "")
    analysis_model = scratchpad.get("analysis_model_used", "未知模型")

    # Handle key_facts as dict (PEST/SWOT structured) or list
    if isinstance(key_facts, dict):
        # Structured facts like PEST: {"Political": [...], "Economic": [...]}
        facts_html = ""
        for category, facts_list in key_facts.items():
            if facts_list:
                facts_html += f'<div style="margin-bottom:12px;"><strong style="color:#d4a574;">{category}:</strong><br>'
                for fact in facts_list:
                    facts_html += f'<div class="fact-item">&#8226; {fact}</div>'
                facts_html += '</div>'
        if not facts_html:
            facts_html = '<div class="text-muted">待提取</div>'
    elif isinstance(key_facts, list):
        facts_html = "".join(
            f'<div class="fact-item">&#8226; {f.get("content", str(f)) if isinstance(f, dict) else f}</div>'
            for f in key_facts
        ) if key_facts else '<div class="text-muted">待提取</div>'
    else:
        facts_html = '<div class="text-muted">待提取</div>'

    insights_html = "".join(f"<li>{ins}</li>" for ins in insights) if insights else "<li>待生成</li>"

    summary_html = f'<div style="padding:12px;background:#1e2a3a;border-radius:6px;margin-bottom:16px;color:#e8e0d8;font-size:0.85rem;"><strong style="color:#d4a574;">分析模型:</strong> {analysis_model}<br><strong style="color:#d4a574;">文档摘要:</strong> {document_summary[:200]}{"..." if len(document_summary) > 200 else ""}</div>' if document_summary else ""

    return f"""
    <div class="analysis-panel">
        {summary_html}
        <div class="analysis-section">
            <h4>【关键事实提取】</h4>
            {facts_html}
        </div>
        <div class="analysis-section">
            <h4>&#128161; 洞察要点</h4>
            <ul>{insights_html}</ul>
        </div>
    </div>
    """


def render_review_tab(review: Dict) -> str:
    """Render LLM review with dimension scores and issues."""
    if not review:
        return '<div class="text-muted">LLM评审尚未完成，请等待审核节点</div>'

    total_score = review.get("score", 0)
    dimension_scores = review.get("dimension_scores", {})
    issues = review.get("issues", [])

    # Map English dimension keys to Chinese labels
    dim_labels = {
        "topic_relevance": "主题契合度",
        "analysis_depth": "分析深度",
        "writing_quality": "写作专业度",
        "citation_sufficiency": "引用充分性",
        "groundedness": "内容真实性",
        "context_coherence": "上下文连贯性",
    }

    dim_html = ""
    for dim_name, score in dimension_scores.items():
        label = dim_labels.get(dim_name, dim_name)
        # Normalize score to percentage for bar width
        # Each dimension has different max (15 or 20), calculate percentage
        max_scores = {
            "topic_relevance": 15, "analysis_depth": 20,
            "writing_quality": 15, "citation_sufficiency": 15,
            "groundedness": 20, "context_coherence": 15,
        }
        max_score = max_scores.get(dim_name, 20)
        bar_pct = int((score / max_score) * 100) if max_score else score
        warning = " &#9888;" if score < max_score * 0.7 else ""
        dim_html += f"""
        <div class="score-row">
            <span class="dim-name">{label}</span>
            <div class="score-bar">
                <div class="score-bar-fill" style="width:{bar_pct}%; background:{_score_color(bar_pct)};"></div>
            </div>
            <span class="dim-score">{score}/{max_score}{warning}</span>
        </div>
        """

    issues_html = ""
    for i, issue in enumerate(issues):
        # Issues can be strings or dicts
        if isinstance(issue, dict):
            title = issue.get("title", "")
            location = issue.get("location", "")
            desc = issue.get("description", "")
            suggestion = issue.get("suggestion", "")
            issues_html += f"""
            <div class="issue-card">
                <div class="issue-header">问题 #{i + 1}: {title}</div>
                {f'<div class="issue-location">位置: {location}</div>' if location else ''}
                <div class="issue-desc">{desc}</div>
                {f'<div class="issue-suggestion">建议: {suggestion}</div>' if suggestion else ''}
            </div>
            """
        else:
            issues_html += f"""
            <div class="issue-card">
                <div class="issue-header">问题 #{i + 1}</div>
                <div class="issue-desc">{issue}</div>
            </div>
            """

    # Show suggestion if available
    suggestion = review.get("suggestion", "")
    suggestion_html = ""
    if suggestion:
        sug_labels = {
            "approve": "通过",
            "revise:data": "需补充数据",
            "revise:logic": "需重新分析",
            "revise:writing": "需重写内容",
        }
        sug_text = sug_labels.get(suggestion, suggestion)
        sug_color = "#4ade80" if suggestion == "approve" else "#d4a574"
        suggestion_html = f"""
        <div style="margin-top:12px; padding:8px 12px; background:var(--bg-card); border-radius:6px; border-left:3px solid {sug_color}; color:#e8e0d8;">
            <strong style="color:#e8e0d8;">评审建议:</strong> <span style="color:{sug_color};">{sug_text}</span>
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
        {suggestion_html}
        <div class="issues-section">
            <h4>&#9888; 发现的问题</h4>
            {issues_html if issues_html else '<div class="text-muted">未发现严重问题</div>'}
        </div>
    </div>
    """


def render_history_tab(history: List[Dict]) -> str:
    """Render version history timeline."""
    if not history:
        return '<div class="text-muted">本章尚未发生修改</div>'

    entries = ""
    for i, entry in enumerate(history):
        is_current = entry.get("is_current", False)
        version_class = "history-current" if is_current else "history-past"
        marker = "&#9679;" if is_current else f"版本 #{i + 1}"

        entries += f"""
        <div class="history-entry {version_class}">
            <div class="history-header">{marker} &middot; {entry.get('label', '')}</div>
            <div class="history-meta">
                时间: {entry.get('timestamp', 'N/A')}<br>
                操作: {entry.get('action', 'N/A')}<br>
                评审: {entry.get('score', 'N/A')}分
            </div>
        </div>
        """

    return f'<div class="history-timeline">{entries}</div>'


def _score_color(score: int) -> str:
    if score >= 85:
        return "#4ade80"
    if score >= 70:
        return "#d4a574"
    return "#ef4444"