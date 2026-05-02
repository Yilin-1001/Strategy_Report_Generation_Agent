"""Left sidebar: milestone progress tracker + draft library."""
from typing import List, Dict, Any


CHAPTER_META = [
    {"index": 0, "title": "宏观政策环境与时代要求", "model": "PEST", "phase": "diagnosis"},
    {"index": 1, "title": "区域战略与交通强省建设剖析", "model": "区域分析", "phase": "diagnosis"},
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

    diagnosis_nodes = ""
    initiatives_nodes = ""
    for ch in CHAPTER_META:
        idx = ch["index"]
        if idx < len(context_pool):
            status = "done"
            icon = "&#10003;"
        elif idx == current_index and not is_blueprint_phase:
            status = "active"
            icon = "&#9679;"
        else:
            status = "pending"
            icon = "&#9675;"

        node = f"""
        <div class="milestone-node milestone-{status}">
            <div class="milestone-icon">{icon}</div>
            <div class="milestone-info">
                <div class="milestone-title">Ch{idx + 1} {ch['title']}</div>
                <div class="milestone-meta">{ch['model']} &middot; {_status_text(status)}</div>
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
        bp_icon = "&#9670;"
    elif blueprint_approved:
        bp_status = "done"
        bp_icon = "&#10003;"
    else:
        bp_status = "pending"
        bp_icon = "&#9671;"

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
        <div class="progress-label">总进度: {progress_pct}%</div>
        <div class="score-bar">
            <div class="score-bar-fill" style="width:{progress_pct}%; background:var(--accent-gold);"></div>
        </div>
        <div class="progress-count">{completed_count}/8 章</div>
    </div>
    """

    # Draft library
    draft_items = ""
    for i in range(min(len(context_pool), len(CHAPTER_META))):
        title = CHAPTER_META[i]["title"][:15]
        draft_items += f"""
        <div class="draft-item">
            <span style="color:#e8c9a0;">&#9656; {title}...</span>
            <span style="color:#4ade80;">&#10003;</span>
        </div>
        """

    return f"""
    <div class="progress-panel">
        <div class="panel-header">
            <span class="panel-icon">&#9654;</span>
            <span class="panel-title">战略报告生成</span>
        </div>

        <div class="phase-label">&mdash;&mdash; 诊断阶段 &mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;</div>
        {diagnosis_nodes}

        {blueprint_node}

        <div class="phase-label">&mdash;&mdash; 举措阶段 &mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;</div>
        {initiatives_nodes}

        {progress_bar}

        <div class="phase-label">&mdash;&mdash; 报告草稿库 &mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;</div>
        {draft_items if draft_items else '<div class="text-muted">暂无已完成章节</div>'}
    </div>
    """


def _status_text(status: str) -> str:
    return {"done": "已完成", "active": "生成中...", "pending": "待处理"}.get(status, "未知")
