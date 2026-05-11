"""Gradio frontend: Executive Intelligence Dashboard for report generation."""
import uuid
from pathlib import Path
import gradio as gr

from rag_project.agent.workflow_service import WorkflowService, NODE_LABELS, PIPELINE_NODES
from rag_project.agent.frontend.progress_panel import render_progress_panel, CHAPTER_META
from rag_project.agent.frontend.workspace_tabs import (
    render_draft_tab, render_docs_tab, render_analysis_tab,
    render_review_tab, render_history_tab,
)
from rag_project.agent.frontend.blueprint_panel import render_blueprint_review
from rag_project.agent.frontend.report_viewer import render_report_viewer, render_report_card

_CSS_PATH = Path(__file__).parent / "theme.css"


def _make_progress_html_with_node(
    current_node: str,
    chapter_index: int,
    context_pool: list,
    current_phase: str,
    is_blueprint_phase: bool = False,
    blueprint_approved: bool = False,
) -> str:
    """Build left sidebar HTML showing full milestone progress."""
    # Render the full milestone panel from progress_panel
    milestone_html = render_progress_panel(
        current_index=chapter_index,
        phase=current_phase,
        context_pool=context_pool,
        is_blueprint_phase=is_blueprint_phase,
        blueprint_approved=blueprint_approved,
    )
    return milestone_html


def _make_loading_right(current_node: str) -> str:
    """Build right side loading HTML — old design, sized to match chapter-header."""
    label = NODE_LABELS.get(current_node, current_node)
    current_idx = PIPELINE_NODES.index(current_node) if current_node in PIPELINE_NODES else -1

    pipeline_html = ""
    for i, node in enumerate(PIPELINE_NODES):
        if i < current_idx:
            dot_class = "dot-done"
            step_class = "step-done"
        elif i == current_idx:
            dot_class = "dot-active"
            step_class = "step-active"
        else:
            dot_class = "dot-pending"
            step_class = ""

        short_name = node.capitalize()
        if node == "prepare_chapter":
            short_name = "Prep"
        elif node == "coordinator":
            short_name = "Coord"

        connector = ""
        if i > 0:
            prev_idx = i - 1
            if prev_idx < current_idx:
                conn_class = "conn-done"
            elif prev_idx == current_idx:
                conn_class = "conn-active"
            else:
                conn_class = "conn-pending"
            connector = f'<div class="pipeline-connector {conn_class}"></div>'

        pipeline_html += f'''
        {connector}
        <div class="pipeline-step {step_class}">
            <div class="pipeline-dot {dot_class}"></div>
            <span class="pipeline-label">{short_name}</span>
        </div>
        '''

    return f'''
    <div class="processing-overlay">
        <div class="node-activity-card">
            <div class="node-activity-header">
                <div class="node-activity-ring">
                    <div class="node-activity-icon"></div>
                </div>
                <div class="node-activity-label">
                    <span class="node-activity-name">{label}</span>
                    <span class="node-activity-status">Processing</span>
                </div>
            </div>
            <div class="pipeline-track">
                {pipeline_html}
            </div>
        </div>
    </div>
    '''


def _get_draft_choices(context_pool):
    """Generate dropdown choices from completed chapters."""
    choices = []
    for i in range(min(len(context_pool), len(CHAPTER_META))):
        choices.append(f"Ch{i+1}: {CHAPTER_META[i]['title']}")
    return choices


def create_app():
    service = WorkflowService()

    with gr.Blocks(title="战略报告生成系统", css=_CSS_PATH.read_text(encoding="utf-8")) as app:
        tid = gr.State(value="t1")
        ui_state = gr.State(value={})

        gr.HTML("""
        <div class="top-nav">
            <div class="nav-logo">企业战略报告生成系统</div>
        </div>""")

        # === Welcome Page ===
        with gr.Group(visible=True, elem_classes=["welcome-page"]) as welcome_page:
            gr.HTML("""
            <div class="welcome-landing">
                <div class="welcome-bg-orb welcome-bg-orb-1"></div>
                <div class="welcome-bg-orb welcome-bg-orb-2"></div>
                <div class="welcome-hero">
                    <div class="welcome-badge">ENTERPRISE STRATEGY</div>
                    <h1 class="welcome-title">战略报告生成系统</h1>
                    <div class="welcome-divider">
                        <span class="welcome-divider-dot"></span>
                        <span class="welcome-divider-line"></span>
                        <span class="welcome-divider-dot"></span>
                    </div>
                    <p class="welcome-desc">
                        基于多Agent协作的智能战略分析平台<br>
                        全流程透明 · 人机协同 · 多维建模
                    </p>
                </div>
                <div class="welcome-features">
                    <div class="welcome-feature-card">
                        <div class="feature-card-icon">&#9670;</div>
                        <div class="feature-card-title">诊断建模</div>
                        <div class="feature-card-models">PEST · SWOT · 波特五力</div>
                        <div class="feature-card-desc">宏观环境与内部能力双维诊断</div>
                    </div>
                    <div class="welcome-feature-card">
                        <div class="feature-card-icon">&#9674;</div>
                        <div class="feature-card-title">战略推演</div>
                        <div class="feature-card-models">TOWS · BSC · BCG</div>
                        <div class="feature-card-desc">蓝图驱动的战略举措推演</div>
                    </div>
                    <div class="welcome-feature-card">
                        <div class="feature-card-icon">&#9733;</div>
                        <div class="feature-card-title">协同审核</div>
                        <div class="feature-card-models">LLM评审 · 人工决策</div>
                        <div class="feature-card-desc">人机协同的质量把控流程</div>
                    </div>
                </div>
            </div>""")
            with gr.Column(elem_classes=["welcome-input-area"]):
                topic_input = gr.Textbox(
                    label="报告主题",
                    placeholder="输入报告主题，例如：某集团2026年度战略规划报告",
                    lines=2,
                    elem_classes=["welcome-topic-input"],
                )
                btn_start = gr.Button("开始生成", variant="primary", elem_classes=["welcome-start-btn"])

        # === Main Workbench ===
        with gr.Group(visible=False) as workbench:
            with gr.Row(elem_classes=["workbench-row"]):
                # Left: Progress Sidebar
                with gr.Column(scale=1, min_width=380, elem_classes=["sidebar-left"]):
                    progress_html = gr.HTML()
                    draft_dropdown = gr.Dropdown(
                        label="报告草稿库",
                        choices=[],
                        interactive=True,
                        elem_classes=["draft-dropdown"],
                    )
                    btn_return_current = gr.Button(
                        "返回当前章节",
                        visible=False,
                        elem_classes=["btn-return-current"],
                    )
                # Middle: Main Content
                with gr.Column(scale=3, elem_classes=["content-main"]):
                    chapter_header = gr.HTML()
                    with gr.Tabs() as main_tabs:
                        with gr.Tab("检索文档", id="tab_docs") as tab_docs:
                            docs_html = gr.HTML()
                        with gr.Tab("分析过程", id="tab_analysis") as tab_analysis:
                            analysis_html = gr.HTML()
                        with gr.Tab("草稿", id="tab_draft") as tab_draft:
                            draft_html = gr.HTML()
                        with gr.Tab("评审详情", id="tab_review") as tab_review:
                            review_html = gr.HTML()
                        with gr.Tab("修改历史", id="tab_history") as tab_history:
                            history_html = gr.HTML()
                    blueprint_html = gr.HTML(visible=False)
                    review_summary = gr.HTML()
                # Right: Review Actions Panel
                with gr.Column(scale=1, min_width=220, elem_classes=["sidebar-right"]):
                    with gr.Column(elem_classes=["review-actions-panel"]):
                        gr.HTML("""<div class="review-panel-header"><span class="panel-icon">&#9998;</span>审核决策</div>""")
                        with gr.Row(elem_classes=["review-actions-primary"]):
                            btn_approve = gr.Button("✔ 批准通过", elem_classes=["btn-approve"])
                        gr.HTML("""<div class="review-section-label">修订指令</div>""")
                        with gr.Column(elem_classes=["review-actions-secondary"]):
                            btn_revise_blueprint = gr.Button("↻ 重新生成蓝图", elem_classes=["btn-revise"], visible=False)
                            btn_revise_data = gr.Button("↩ 补充数据", elem_classes=["btn-revise"])
                            btn_revise_logic = gr.Button("↺ 重新分析", elem_classes=["btn-revise"])
                            btn_revise_writing = gr.Button("↻ 重写内容", elem_classes=["btn-revise"])
                        gr.HTML("""<div class="review-section-label">反馈说明</div>""")
                        with gr.Row(elem_classes=["review-feedback-row"]):
                            feedback_input = gr.Textbox(show_label=False, placeholder="输入补充说明或修改建议...", lines=3, interactive=True, elem_classes=["feedback-input"])
                        gr.HTML("""<div class="review-divider-thin"></div>""")
                        with gr.Row(elem_classes=["review-actions-danger"]):
                            btn_stop = gr.Button("☠ 终止报告", elem_classes=["btn-danger"])

        # === Completion Page ===
        with gr.Group(visible=False) as completion_page:
            completion_content = gr.HTML()
            btn_view_report = gr.Button("查看报告")
            btn_new_report = gr.Button("新建报告")

        # === Report Viewer Page ===
        with gr.Group(visible=False) as report_page:
            report_content = gr.HTML()
            btn_back = gr.Button("<< 返回")

        # === History Page ===
        with gr.Group(visible=False) as history_page:
            history_list_html = gr.HTML()
            btn_back_from_history = gr.Button("<< 返回")

        # === Shared outputs (22) ===
        _N_OUTPUTS = 22
        _all_outputs = [
            welcome_page,          # 0
            workbench,             # 1
            completion_page,       # 2
            report_page,           # 3
            history_page,          # 4
            progress_html,         # 5
            chapter_header,        # 6
            docs_html,             # 7
            analysis_html,         # 8
            draft_html,            # 9
            review_html,           # 10
            history_html,          # 11
            review_summary,        # 12
            blueprint_html,        # 13
            completion_content,    # 14
            report_content,        # 15
            history_list_html,     # 16
            ui_state,              # 17
            btn_revise_blueprint,  # 18
            main_tabs,             # 19
            draft_dropdown,        # 20
            btn_return_current,    # 21
        ]

        # === Event Handlers ===

        def _make_tuple(items_21):
            assert len(items_21) == _N_OUTPUTS, f"Expected {_N_OUTPUTS}, got {len(items_21)}"
            return tuple(items_21)

        def _progress_tuple(current_node: str, chapter_index: int,
                            node_output: dict = None,
                            context_pool: list = None,
                            current_phase: str = "diagnosis",
                            is_blueprint_phase: bool = False,
                            blueprint_approved: bool = False):
            """Build UI state during workflow progress with partial results."""
            node_output = node_output or {}
            context_pool = context_pool or []

            # Render partial content if available
            docs_content = ""
            analysis_content = ""
            draft_content = ""
            selected_tab = "tab_docs"

            scratchpad = node_output.get("chapter_scratchpad", {}) or {}

            if current_node == "researcher":
                docs_data = scratchpad.get("retrieved_docs", [])
                if docs_data:
                    docs_content = render_docs_tab(docs_data)
                    selected_tab = "tab_docs"
            elif current_node == "analyst":
                if scratchpad:
                    analysis_content = render_analysis_tab(scratchpad)
                    docs_data = scratchpad.get("retrieved_docs", [])
                    docs_content = render_docs_tab(docs_data) if docs_data else ""
                selected_tab = "tab_analysis"
            elif current_node == "writer":
                draft_text = node_output.get("current_draft", "")
                if draft_text:
                    draft_content = render_draft_tab(draft_text)
                docs_data = scratchpad.get("retrieved_docs", [])
                docs_content = render_docs_tab(docs_data) if docs_data else ""
                if scratchpad:
                    analysis_content = render_analysis_tab(scratchpad)
                selected_tab = "tab_draft"
            else:
                docs_data = scratchpad.get("retrieved_docs", [])
                docs_content = render_docs_tab(docs_data) if docs_data else ""
                if scratchpad:
                    analysis_content = render_analysis_tab(scratchpad)
                draft_text = node_output.get("current_draft", "")
                draft_content = render_draft_tab(draft_text) if draft_text else ""

            progress_html = _make_progress_html_with_node(
                current_node, chapter_index, context_pool,
                current_phase, is_blueprint_phase, blueprint_approved,
            )

            return _make_tuple([
                gr.update(visible=False),                                    # 0  welcome
                gr.update(visible=True),                                     # 1  workbench
                gr.update(visible=False),                                    # 2  completion
                gr.update(visible=False),                                    # 3  report
                gr.update(visible=False),                                    # 4  history
                gr.update(value=progress_html),                              # 5  progress
                gr.update(value=_make_loading_right(current_node)),         # 6  chapter_header
                gr.update(value=docs_content),                               # 7  docs
                gr.update(value=analysis_content),                           # 8  analysis
                gr.update(value=draft_content),                              # 9  draft
                gr.update(value=""),                                         # 10 review
                gr.update(value=""),                                         # 11 history
                gr.update(value=""),                                         # 12 review_summary
                gr.update(value="", visible=False),                          # 13 blueprint
                gr.update(),                                                 # 14 completion_content
                gr.update(),                                                 # 15 report_content
                gr.update(),                                                 # 16 history_list_html
                gr.update(),                                                 # 17 ui_state
                gr.update(visible=False),                                    # 18 btn_revise_blueprint
                gr.update(selected=selected_tab),                            # 19 main_tabs
                gr.update(choices=_get_draft_choices(context_pool), value=None),  # 20 draft_dropdown
                gr.update(visible=True),                                         # 21 btn_return_current
            ])

        def start_generation(topic, tid_val):
            print(f"=== start_generation: topic={topic!r}")
            if not topic.strip():
                return _make_tuple([gr.update() for _ in range(_N_OUTPUTS)])

            yield _progress_tuple("coordinator", 0, node_output={},
                                  context_pool=[], current_phase="diagnosis")

            try:
                for update in service.start_report(topic, tid_val):
                    if update["type"] == "progress":
                        node = update["node"]
                        idx = update.get("chapter_index", 0)
                        node_output = update.get("node_output", {})
                        phase = node_output.get("current_phase", "diagnosis") if isinstance(node_output, dict) else "diagnosis"
                        ctx_pool = node_output.get("context_pool", []) if isinstance(node_output, dict) else []
                        yield _progress_tuple(node, idx, node_output,
                                              context_pool=ctx_pool, current_phase=phase)
                    elif update["type"] == "done":
                        print(f"=== start_report done, status={update.get('status')}")
                        yield _render_workbench(update)
            except Exception as e:
                import traceback
                traceback.print_exc()
                state = {"status": "error", "error": str(e)}
                yield _render_workbench(state)

        btn_start.click(
            fn=start_generation,
            inputs=[topic_input, tid],
            outputs=_all_outputs,
        )

        def _make_submit_fn(decision):
            def submit_handler(feedback, current_state, tid_val):
                effective = decision
                if decision == "approve" and current_state.get("is_blueprint_phase"):
                    effective = "approve_blueprint"

                idx = current_state.get("current_chapter_index", 0)
                ctx_pool = current_state.get("context_pool", [])
                phase = current_state.get("current_phase", "diagnosis")
                is_bp = current_state.get("is_blueprint_phase", False)
                bp = current_state.get("strategic_blueprint")
                bp_approved = bp.get("approved", False) if bp else False

                # Calculate expected display state after approval
                display_ctx = list(ctx_pool)
                next_idx = idx
                next_phase = phase
                next_is_bp = is_bp
                next_bp_approved = bp_approved

                if decision == "approve":
                    if is_bp:
                        # Blueprint approval -> enter initiatives phase
                        next_idx = idx + 1
                        next_phase = "initiatives"
                        next_is_bp = False
                        next_bp_approved = True
                    elif idx == 2:
                        # Ch3 approval -> strategist phase, index stays at 2
                        next_idx = 2
                        next_is_bp = True
                        display_ctx.append("_")  # Placeholder for just-approved Ch3
                    else:
                        # Normal chapter approval -> next chapter
                        next_idx = idx + 1
                        display_ctx.append("_")  # Placeholder for just-approved chapter

                display_node = "strategist" if next_is_bp and not is_bp else "prepare_chapter"

                yield _progress_tuple(display_node, next_idx, node_output={},
                                      context_pool=display_ctx, current_phase=next_phase,
                                      is_blueprint_phase=next_is_bp, blueprint_approved=next_bp_approved)

                try:
                    for update in service.submit_review(tid_val, effective, feedback):
                        if update["type"] == "progress":
                            node = update["node"]
                            ci = update.get("chapter_index", next_idx)
                            node_output = update.get("node_output", {})
                            out_phase = node_output.get("current_phase", next_phase) if isinstance(node_output, dict) else next_phase
                            yield _progress_tuple(node, ci, node_output,
                                                  context_pool=display_ctx, current_phase=out_phase,
                                                  is_blueprint_phase=next_is_bp, blueprint_approved=next_bp_approved)
                        elif update["type"] == "done":
                            new_state = update
                            if new_state["status"] == "completed":
                                yield _show_completion(new_state)
                            else:
                                yield _render_workbench(new_state)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    new_state = dict(current_state, error=str(e))
                    yield _render_workbench(new_state)
            return submit_handler

        for btn, decision in [
            (btn_approve, "approve"),
            (btn_revise_blueprint, "revise_blueprint"),
            (btn_revise_data, "revise:data"),
            (btn_revise_logic, "revise:logic"),
            (btn_revise_writing, "revise:writing"),
            (btn_stop, "finished"),
        ]:
            btn.click(
                fn=_make_submit_fn(decision),
                inputs=[feedback_input, ui_state, tid],
                outputs=_all_outputs,
            )

        def view_report(current_state, tid_val):
            report = service.get_report(tid_val) or ""
            chapters = current_state.get("context_pool", [])
            report_eval = current_state.get("report_evaluation")
            viewer = render_report_viewer(report, chapters, report_eval)
            return _nav_to_report(viewer)

        btn_view_report.click(fn=view_report, inputs=[ui_state, tid], outputs=_all_outputs)
        btn_new_report.click(fn=lambda: _nav_to_welcome(), outputs=_all_outputs)
        btn_back.click(fn=lambda s: _show_completion(s), inputs=[ui_state], outputs=_all_outputs)
        btn_back_from_history.click(fn=lambda: _nav_to_welcome(), outputs=_all_outputs)

        # === Draft library selection handler ===
        def _on_draft_click(selected, current_state):
            """Handle draft library selection: show the selected chapter's draft.

            When viewing a historical chapter (not the current one being reviewed),
            shows the 'return to current' button in the sidebar.
            """
            if not selected:
                return _make_tuple([gr.update() for _ in range(_N_OUTPUTS)])

            try:
                idx = int(selected.split(":")[0].replace("Ch", "")) - 1
            except (ValueError, IndexError):
                return _make_tuple([gr.update() for _ in range(_N_OUTPUTS)])

            context_pool = current_state.get("context_pool", [])
            if idx < 0 or idx >= len(context_pool):
                return _make_tuple([gr.update() for _ in range(_N_OUTPUTS)])

            chapter_content = context_pool[idx]
            draft_display = render_draft_tab(chapter_content)

            # Check if this is the current chapter being reviewed
            current_idx = current_state.get("current_chapter_index", 0)
            is_current_chapter = (idx == current_idx)

            # Show return button only when viewing a different chapter
            show_return = not is_current_chapter

            items = [gr.update() for _ in range(_N_OUTPUTS)]
            items[1] = gr.update(visible=True)                          # workbench
            items[9] = gr.update(value=draft_display)                   # draft content
            items[19] = gr.update(selected="tab_draft")                 # switch to draft tab
            items[21] = gr.update(visible=show_return)                  # btn_return_current
            return _make_tuple(items)

        draft_dropdown.change(
            fn=_on_draft_click,
            inputs=[draft_dropdown, ui_state],
            outputs=_all_outputs,
        )

        # === Return to current chapter handler ===
        def _return_to_current_chapter(current_state):
            """Return to viewing the current chapter being reviewed.

            Restores the full workbench state with current draft, review, history, etc.
            """
            return _render_workbench(current_state)

        btn_return_current.click(
            fn=_return_to_current_chapter,
            inputs=[ui_state],
            outputs=_all_outputs,
        )

    return app


# ========================================
# Render helpers
# ========================================

def _render_workbench(state: dict) -> tuple:
    if state.get("status") == "error":
        error_msg = state.get("error", "未知错误")
        error_html = f"""
        <div class="chapter-header" style="border-color: var(--danger);">
            <div class="chapter-title" style="color: var(--danger);">生成出错</div>
            <div class="chapter-question" style="color: var(--text-muted);">{error_msg}</div>
        </div>
        """
        items = [
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=error_html),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            gr.update(value="", visible=False),
            gr.update(), gr.update(), gr.update(),
            gr.update(value=state), gr.update(visible=False),
            gr.update(),  # main_tabs
            gr.update(),  # draft_dropdown
            gr.update(),  # btn_return_current
        ]
        return tuple(items)

    is_blueprint = state.get("is_blueprint_phase", False)
    blueprint = state.get("strategic_blueprint")
    blueprint_approved = blueprint.get("approved", False) if blueprint else False

    progress = render_progress_panel(
        current_index=state.get("current_chapter_index", 0),
        phase=state.get("current_phase", "diagnosis"),
        context_pool=state.get("context_pool", []),
        is_blueprint_phase=is_blueprint,
        blueprint_approved=blueprint_approved,
    )

    chapter_title = state.get("chapter_title", "")
    chapter_question = state.get("chapter_question", "")
    phase = state.get("current_phase", "diagnosis")
    phase_label = "诊断阶段" if phase == "diagnosis" else "举措阶段"

    if is_blueprint:
        header_html = f"""
        <div class="blueprint-banner">
            <div class="blueprint-banner-title">战略蓝图审核阶段</div>
            <div class="blueprint-banner-desc">基于前三章诊断结果，系统已自动生成战略蓝图。请审核后批准进入举措阶段，或要求重新生成。</div>
        </div>
        <div class="chapter-header">
            <div class="chapter-title">战略蓝图生成</div>
            <div class="chapter-question">基于前三章诊断结果，生成战略蓝图供审核</div>
            <div class="chapter-phase-badge">{phase_label} &middot; 蓝图审核</div>
        </div>
        """
    elif chapter_title:
        idx = state.get("current_chapter_index", 0)
        header_html = f"""
        <div class="chapter-header">
            <div class="chapter-title">Ch{idx + 1}: {chapter_title}</div>
            <div class="chapter-question">{chapter_question}</div>
            <div class="chapter-phase-badge">{phase_label}</div>
        </div>
        """
    else:
        header_html = '<div class="chapter-header"><div class="chapter-title">准备中...</div></div>'

    scratchpad = state.get("chapter_scratchpad", {})
    docs_data = scratchpad.get("retrieved_docs", [])
    docs = render_docs_tab(docs_data)
    analysis = render_analysis_tab(scratchpad)
    draft = render_draft_tab(state.get("current_draft", ""))
    review = render_review_tab(state.get("llm_review_result"))
    history_data = scratchpad.get("revision_history", [])
    history = render_history_tab(history_data, current_draft=state.get("current_draft", ""))

    # Default to draft tab when workflow finishes (human review)
    selected_tab = "tab_draft"

    review_result = state.get("llm_review_result")
    if review_result:
        score = review_result.get("score", 0)
        score_color = "#4ade80" if score >= 85 else "#d4a574" if score >= 70 else "#ef4444"
        summary_html = f"""
        <div style="display:flex; align-items:center; gap:16px; padding:12px 16px;
                     background:var(--bg-card); border-radius:8px; border:1px solid var(--border); margin-top:16px;">
            <span style="font-family:var(--font-display); font-size:1.5rem; font-weight:700; color:{score_color};">{score}</span>
            <span style="color:var(--text-muted);">/100</span>
            <span style="color:var(--text-secondary); font-size:0.9rem;">
                {len(review_result.get("issues", []))} 个问题需要关注
            </span>
        </div>
        """
        # Show review tab when review is done
        selected_tab = "tab_review"
    else:
        summary_html = ""

    bp_html = ""
    bp_visible = False
    bp_btn_visible = False
    if is_blueprint and blueprint:
        bp_html = render_blueprint_review(blueprint)
        bp_visible = True
        bp_btn_visible = True

    items = [
        gr.update(visible=False),        # 0  welcome
        gr.update(visible=True),         # 1  workbench
        gr.update(visible=False),        # 2  completion
        gr.update(visible=False),        # 3  report
        gr.update(visible=False),        # 4  history
        gr.update(value=progress),       # 5  progress
        gr.update(value=header_html),    # 6  chapter_header
        gr.update(value=docs),           # 7  docs
        gr.update(value=analysis),       # 8  analysis
        gr.update(value=draft),          # 9  draft
        gr.update(value=review),         # 10 review
        gr.update(value=history),        # 11 history
        gr.update(value=summary_html),   # 12 review_summary
        gr.update(value=bp_html, visible=bp_visible),  # 13 blueprint
        gr.update(),                     # 14 completion_content
        gr.update(),                     # 15 report_content
        gr.update(),                     # 16 history_list_html
        gr.update(value=state),          # 17 ui_state
        gr.update(visible=bp_btn_visible),  # 18 btn_revise_blueprint
        gr.update(selected=selected_tab),   # 19 main_tabs
        gr.update(choices=_get_draft_choices(state.get("context_pool", [])), value=None),  # 20 draft_dropdown
        gr.update(visible=True),                                              # 21 btn_return_current
    ]
    return tuple(items)


def _show_completion(state: dict) -> tuple:
    report = state.get("final_report", "")
    context_pool = state.get("context_pool", [])
    report_eval = state.get("report_evaluation")
    word_count = len(report) if report else 0
    chapter_count = len(context_pool)

    # Build evaluation display if available
    eval_display = ""
    if report_eval and report_eval.get("total_score", 0) > 0:
        eval_total = report_eval.get("total_score", 0)
        eval_color = "#4ade80" if eval_total >= 75 else "#d4a574" if eval_total >= 60 else "#ef4444"
        eval_display = f"""
        <div style="margin-top:24px; padding:20px; background:var(--bg-card); border-radius:12px; border:1px solid var(--border);">
            <div style="font-family:var(--font-display); font-size:1.1rem; color:var(--accent-gold); margin-bottom:12px;">全文质量评估</div>
            <div style="display:flex; align-items:center; gap:16px;">
                <span style="font-family:var(--font-display); font-size:2.5rem; font-weight:700; color:{eval_color};">{eval_total}</span>
                <span style="color:var(--text-muted); font-size:1rem;">/100</span>
                <span style="color:var(--text-secondary); font-size:0.85rem;">方法论·战略一致·逻辑闭环·创新前瞻·组织治理</span>
            </div>
        </div>
        """

    content = f"""
    <div class="completion-page">
        <div class="completion-icon">&#10003;</div>
        <div class="completion-title">报告生成完成</div>
        <div class="completion-subtitle">
            共完成 {chapter_count}/8 个章节<br>
            总字数: {word_count:,}
        </div>
        {eval_display}
    </div>
    """
    items = [
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(value=content), gr.update(), gr.update(),
        gr.update(value=state), gr.update(visible=False),
        gr.update(),  # main_tabs
        gr.update(),  # draft_dropdown
        gr.update(),  # btn_return_current
    ]
    return tuple(items)


def _nav_to_report(viewer_html: str) -> tuple:
    items = [
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=True), gr.update(visible=False),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(value=viewer_html), gr.update(),
        gr.update(), gr.update(),
        gr.update(),  # main_tabs
        gr.update(),  # draft_dropdown
        gr.update(),  # btn_return_current
    ]
    return tuple(items)


def _nav_to_welcome() -> tuple:
    items = [
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(),
        gr.update(),  # main_tabs
        gr.update(),  # draft_dropdown
        gr.update(),  # btn_return_current
    ]
    return tuple(items)
