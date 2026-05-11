"""Right workspace: tab content renderers for scratchpad data."""
import re
from typing import Dict, Any, List, Optional


def _markdown_to_html(text: str, add_anchor_ids: bool = False) -> str:
    """Convert basic Markdown to HTML without external dependencies.

    Handles: headers (#, ##, ###), bold (**), italic (*), lists (-, *), paragraphs.

    When add_anchor_ids=True, <h1> headers get id="chapter-{index}" for TOC navigation.
    """
    if not text:
        return ""

    lines = text.split("\n")
    html_parts = []
    in_list = False
    list_items = []
    h1_counter = 0  # Track h1 index for anchor IDs

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
            header_text = stripped[2:]
            if add_anchor_ids:
                anchor_id = f"chapter-{h1_counter}"
                h1_counter += 1
                html_parts.append(f'<h1 id="{anchor_id}">{header_text}</h1>')
            else:
                html_parts.append(f'<h1>{header_text}</h1>')
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
    """Render LLM chapter review with 5-dimension scores.

    Supports both old dimension keys (for backward compatibility) and new chapter-level keys:
    - Old: methodology, strategic_alignment, logical_coherence, innovation_insight, organizational_governance
    - New: model_application, data_support, internal_logic, content_depth, writing_quality
    """
    if not review:
        return '<div class="text-muted">LLM评审尚未完成，请等待审核节点</div>'

    total_score = review.get("score", 0)
    dimension_scores = review.get("dimension_scores", {})
    issues = review.get("issues", [])

    # Check which dimension keys are present (new chapter-level or old full-report)
    has_new_dims = "model_application" in dimension_scores

    if has_new_dims:
        # Chapter-level evaluation (new keys)
        dim_config = {
            "model_application": ("模型运用与框架完整性", 20),
            "data_support": ("数据支撑与证据质量", 20),
            "internal_logic": ("内部逻辑与结构清晰度", 20),
            "content_depth": ("内容深度与专业水准", 20),
            "writing_quality": ("写作质量与规范表达", 20),
        }
    else:
        # Full-report evaluation (old keys, backward compatibility)
        dim_config = {
            "methodology": ("方法论运用与分析框架严谨度", 20),
            "strategic_alignment": ("战略一致性与外部环境契合度", 20),
            "logical_coherence": ("逻辑连贯性与战略闭环思维", 20),
            "innovation_insight": ("创新性与前瞻洞察力", 20),
            "organizational_governance": ("隐性约束洞察与组织治理深度", 20),
        }

    dim_html = ""
    for dim_key, (label, max_score) in dim_config.items():
        dim_data = dimension_scores.get(dim_key, {})
        # Handle both old format (int) and new format (dict with score/analysis)
        if isinstance(dim_data, dict):
            score = dim_data.get("score", 0)
            analysis = dim_data.get("analysis", "")
            display_label = dim_data.get("label", label)
        else:
            score = dim_data
            analysis = ""
            display_label = label

        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0

        bar_pct = int((score / max_score) * 100) if max_score else 0
        warning = " &#9888;" if score < max_score * 0.7 else ""
        analysis_html = ""
        if analysis:
            analysis_html = f'<div class="dim-analysis">{analysis}</div>'

        dim_html += f"""
        <div class="score-row">
            <span class="dim-name">{display_label}</span>
            <div class="score-bar">
                <div class="score-bar-fill" style="width:{bar_pct}%; background:{_score_color(bar_pct)};"></div>
            </div>
            <span class="dim-score">{score}/{max_score}{warning}</span>
        </div>
        {analysis_html}
        """

    issues_html = ""
    for i, issue in enumerate(issues):
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


def render_history_tab(history: List[Dict], current_draft: str = "") -> str:
    """Render version history timeline with expandable draft previews.

    Shows each revision as a timeline entry with:
    - Version number, timestamp, action type, review score
    - Top issues from the review
    - Expandable draft content preview
    - Current draft highlighted as latest version
    """
    # Build entries: combine revision_history with current draft
    entries_html = ""
    total_versions = len(history)

    # Render past revision entries
    for i, entry in enumerate(history):
        version = entry.get("version", i + 1)
        timestamp = entry.get("timestamp", "N/A")
        action = entry.get("action", entry.get("decision", "N/A"))
        score = entry.get("score", 0)
        issues = entry.get("issues", [])
        draft_preview = entry.get("draft_preview", "")
        draft_full = entry.get("draft_full", "")
        comments = entry.get("comments", "")
        decision = entry.get("decision", "")

        # Score color
        score_color = "#4ade80" if score >= 85 else "#d4a574" if score >= 70 else "#ef4444"

        # Action badge color
        action_colors = {
            "revise:data": "#60a5fa",
            "revise:logic": "#fbbf24",
            "revise:writing": "#c084fc",
        }
        badge_color = action_colors.get(decision, "#6b7280")

        # Issues summary (show top 2)
        issues_html = ""
        if issues:
            top_issues = issues[:2]
            issue_items = "".join(
                f'<div class="history-issue">&#8226; {issue if len(str(issue)) < 80 else str(issue)[:80] + "..."}</div>'
                for issue in top_issues
            )
            issues_html = f'<div class="history-issues">{issue_items}</div>'

        # Comments
        comments_html = ""
        if comments:
            display_comment = comments if len(comments) < 100 else comments[:100] + "..."
            comments_html = f'<div class="history-comment">反馈: {display_comment}</div>'

        # Draft preview (truncated)
        preview_html = ""
        if draft_preview:
            preview_text = draft_preview[:200] + ("..." if len(draft_preview) > 200 else "")
            preview_html = f'<div class="history-draft-preview">{preview_text}</div>'

        # Full draft (hidden by default, shown on click)
        full_draft_html = ""
        if draft_full:
            full_draft_html = _markdown_to_html(draft_full)

        entries_html += f"""
        <div class="history-entry history-past" onclick="this.querySelector('.history-draft-full').classList.toggle('expanded')">
            <div class="history-header-row">
                <span class="history-version-badge" style="background:{badge_color}20; color:{badge_color}; border: 1px solid {badge_color}40;">
                    V{version}
                </span>
                <span class="history-action-label">{action}</span>
                <span class="history-score-badge" style="color:{score_color}">{score}分</span>
            </div>
            <div class="history-meta-row">
                <span>{timestamp}</span>
            </div>
            {issues_html}
            {comments_html}
            {preview_html}
            <div class="history-draft-expand-hint">点击展开完整草稿</div>
            <div class="history-draft-full">
                <div class="markdown-body markdown-light">{full_draft_html}</div>
            </div>
        </div>
        """

    # Render current version entry (if there's a current draft)
    if current_draft:
        current_preview = current_draft[:200] + ("..." if len(current_draft) > 200 else "")
        current_full_html = _markdown_to_html(current_draft)

        entries_html += f"""
        <div class="history-entry history-current" onclick="this.querySelector('.history-draft-full').classList.toggle('expanded')">
            <div class="history-header-row">
                <span class="history-version-badge" style="background:rgba(212,165,116,0.2); color:#d4a574; border: 1px solid rgba(212,165,116,0.4);">
                    V{total_versions + 1}
                </span>
                <span class="history-action-label" style="color:#d4a574;">当前版本</span>
                <span class="history-current-marker">最新</span>
            </div>
            <div class="history-meta-row">
                <span>当前待审草稿</span>
            </div>
            <div class="history-draft-preview">{current_preview}</div>
            <div class="history-draft-expand-hint">点击展开完整草稿</div>
            <div class="history-draft-full">
                <div class="markdown-body markdown-light">{current_full_html}</div>
            </div>
        </div>
        """

    if not entries_html:
        return '<div class="text-muted">本章尚未发生修改</div>'

    return f'<div class="history-timeline">{entries_html}</div>'


def _score_color(score: int) -> str:
    if score >= 85:
        return "#4ade80"
    if score >= 70:
        return "#d4a574"
    return "#ef4444"