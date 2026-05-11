"""Blueprint review page: structured display of strategic blueprint."""
from typing import Dict, Any, Optional, List


def _normalize_swot_keys(swot: Dict) -> Dict:
    """Normalize SWOT keys to lowercase for consistent access."""
    if not swot:
        return {}

    # Map possible key variations to lowercase
    key_map = {
        "Strengths": "strengths",
        "strengths": "strengths",
        "Weaknesses": "weaknesses",
        "weaknesses": "weaknesses",
        "Opportunities": "opportunities",
        "opportunities": "opportunities",
        "Threats": "threats",
        "threats": "threats",
    }

    normalized = {}
    for key, value in swot.items():
        lower_key = key_map.get(key, key.lower())
        normalized[lower_key] = value

    return normalized


def _normalize_pillars(pillars: List) -> List[Dict]:
    """Convert pillar strings to dict format if needed."""
    if not pillars:
        return []

    result = []
    for p in pillars:
        if isinstance(p, dict):
            result.append(p)
        elif isinstance(p, str):
            # Parse string format like "战略支柱1：名称 - 描述"
            if "：" in p:
                parts = p.split("：", 1)
                name = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
            elif ":" in p:
                parts = p.split(":", 1)
                name = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
            elif "-" in p:
                parts = p.split("-", 1)
                name = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
            else:
                name = p
                desc = ""
            result.append({"name": name, "description": desc})
        else:
            result.append({"name": str(p), "description": ""})

    return result


def render_blueprint_review(blueprint: Dict) -> str:
    """Render the full strategic blueprint for review."""
    if not blueprint:
        return '<div class="text-muted">蓝图数据不可用</div>'

    mission = blueprint.get("mission", "待定义")
    swot = _normalize_swot_keys(blueprint.get("swot_analysis", blueprint.get("swot", {})))
    tows = blueprint.get("tows_strategies", blueprint.get("tows", {}))
    pillars = _normalize_pillars(blueprint.get("strategic_pillars", blueprint.get("pillars", [])))
    kpis = blueprint.get("kpis", {})

    mission_html = f"""
    <div class="blueprint-section">
        <div class="mission-title">&#127919; 企业使命</div>
        <div class="mission-statement">{mission}</div>
    </div>
    """

    swot_html = f"""
    <div class="blueprint-section">
        <h3>SWOT 分析矩阵</h3>
        <div class="swot-grid">
            <div class="swot-cell swot-strength">
                <h4>优势 (S)</h4>
                {_render_list(swot.get("strengths", []))}
            </div>
            <div class="swot-cell swot-opportunity">
                <h4>机会 (O)</h4>
                {_render_list(swot.get("opportunities", []))}
            </div>
            <div class="swot-cell swot-weakness">
                <h4>劣势 (W)</h4>
                {_render_list(swot.get("weaknesses", []))}
            </div>
            <div class="swot-cell swot-threat">
                <h4>威胁 (T)</h4>
                {_render_list(swot.get("threats", []))}
            </div>
        </div>
    </div>
    """

    tows_html = ""
    if tows:
        tows_rows = ""
        for strategy_name, details in tows.items():
            if isinstance(details, list):
                items = "".join(f"<div class='list-item'>&#8226; {d}</div>" for d in details)
            else:
                items = f"<div class='list-item'>&#8226; {details}</div>"
            tows_rows += f"""
            <div class="swot-cell" style="border-left: 4px solid var(--accent-gold);">
                <h4 style="color: var(--accent-gold);">{strategy_name}</h4>
                {items}
            </div>
            """
        if tows_rows:
            tows_html = f"""
            <div class="blueprint-section">
                <h3>TOWS 战略矩阵</h3>
                <div class="swot-grid">{tows_rows}</div>
            </div>
            """

    pillar_cards = "".join(
        f'<div class="pillar-card"><div class="pillar-name">{p.get("name", "")}</div>'
        f'<div class="pillar-desc">{p.get("description", "")}</div></div>'
        for p in pillars
    )
    pillars_html = f"""
    <div class="blueprint-section">
        <h3>战略支柱</h3>
        <div class="pillar-grid">{pillar_cards}</div>
    </div>
    """

    kpi_rows = ""
    for dim, indicators in kpis.items():
        if isinstance(indicators, list):
            kpi_rows += f"<tr><td>{dim}</td><td>{'、'.join(str(ind) for ind in indicators)}</td></tr>"
        else:
            kpi_rows += f"<tr><td>{dim}</td><td>{indicators}</td></tr>"

    kpis_html = ""
    if kpi_rows:
        kpis_html = f"""
        <div class="blueprint-section">
            <h3>关键指标 (BSC框架)</h3>
            <table class="kpi-table">
                <thead><tr><th>维度</th><th>指标</th></tr></thead>
                <tbody>{kpi_rows}</tbody>
            </table>
        </div>
        """

    return f"""
    <div class="blueprint-review">
        <div class="blueprint-header">
            <h2>战略蓝图审核</h2>
            <p>基于前三章诊断结果自动生成，审核通过后将为Ch4-8提供战略约束</p>
        </div>
        {mission_html}
        {swot_html}
        {tows_html}
        {pillars_html}
        {kpis_html}
    </div>
    """


def _render_list(items: list) -> str:
    if not items:
        return '<div class="text-muted">暂无数据</div>'
    return "".join(f"<div class='list-item'>&#8226; {item}</div>" for item in items)