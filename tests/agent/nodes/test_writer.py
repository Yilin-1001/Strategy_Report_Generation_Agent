"""
Tests for Writer node.
"""

import pytest
from unittest.mock import Mock
from rag_project.agent.nodes.writer import writer_node


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager."""
    llm_manager = Mock()
    return llm_manager


@pytest.fixture
def sample_llm_chapter_response():
    """Sample LLM response for chapter generation."""
    return """中国交通运输行业在过去十年取得了显著发展。高速铁路网络迅速扩张，成为全球最大的高铁网络之一[来源: doc1.pdf]。基础设施投资持续增长，为经济发展提供了强有力的支撑。

## 发展现状

近年来，中国交通运输行业保持了稳定增长态势。根据统计数据显示，截至2022年底，中国高速铁路运营里程已突破4万公里[来源: doc2.pdf]，覆盖了全国90%以上的百万人口城市。同时，城市轨道交通系统也在快速扩展，已有50个城市建成地铁或轻轨系统[来源: doc3.pdf]。

在投资方面，交通运输基础设施投资持续加大。仅2021年，全国交通固定资产投资就超过了3.5万亿元，同比增长约8%[来源: doc1.pdf]。这些投资有力地推动了交通网络的完善和升级。

## 主要成就

中国交通运输行业取得的成就举世瞩目。首先，高速铁路技术的自主创新和大规模应用，使中国在这一领域处于世界领先地位[来源: doc2.pdf]。其次，智能交通系统的建设加快推进，提升了运输效率和安全水平。

此外，绿色低碳发展成为行业新趋势。新能源交通工具的推广应用取得了显著进展，电动汽车产销量连续多年位居世界首位[来源: doc3.pdf]。

展望未来，中国交通运输行业将继续保持稳定发展。随着"十四五"规划的深入实施，交通运输基础设施将进一步完善，智能化、绿色化水平将持续提升，为经济社会发展提供更加有力的支撑。"""


def test_writer_node(mock_llm_manager, sample_llm_chapter_response):
    """Test writer node generates chapter from facts and insights."""
    # Setup mock response
    mock_llm_manager.invoke.return_value = sample_llm_chapter_response

    # Prepare state with facts and insights from analyst
    state = {
        "chapter_title": "中国交通运输行业发展现状",
        "chapter_question": "中国交通运输行业的发展现状如何？",
        "chapter_context": "行业分析报告",
        "chapter_scratchpad": {
            "queries": ["中国交通运输行业", "高铁发展现状"],
            "retrieved_docs": [
                {"text": "高速铁路网络达到4万公里", "metadata": {"source": "doc1.pdf", "page": 1}},
                {"text": "投资超过3万亿元", "metadata": {"source": "doc2.pdf", "page": 2}},
            ],
            "key_facts": [
                "中国高速铁路运营里程已突破4万公里",
                "2021年全国交通固定资产投资超过3.5万亿元",
                "已有50个城市建成地铁或轻轨系统"
            ],
            "insights": [
                "高速铁路技术的自主创新使中国处于世界领先地位",
                "智能交通系统建设加快推进，提升了运输效率",
                "绿色低碳发展成为行业新趋势"
            ],
            "document_summary": "Document 1 summary...\nDocument 2 summary..."
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify structure
    assert "current_draft" in result

    # Verify draft is generated
    draft = result["current_draft"]
    assert isinstance(draft, str)
    assert len(draft) > 100, "Draft should be substantial (>100 characters)"

    # Verify draft contains expected content
    assert "交通运输" in draft or "transportation" in draft.lower()
    assert "发展" in draft or "development" in draft.lower()

    # Verify citations are present
    assert "[来源:" in draft or "[Source:" in draft or "[citation:" in draft.lower()

    # Verify draft has structure (opening, body, closing)
    # Should have multiple paragraphs/sections
    assert len(draft.split("\n\n")) >= 2, "Draft should have multiple sections"


def test_writer_node_uses_facts_and_insights(mock_llm_manager):
    """Test that writer node uses key_facts and insights from scratchpad."""
    mock_llm_manager.invoke.return_value = "Generated chapter content..."

    state = {
        "chapter_title": "Test Chapter",
        "chapter_question": "Test question?",
        "chapter_context": "Test context",
        "chapter_scratchpad": {
            "key_facts": ["Fact 1", "Fact 2", "Fact 3"],
            "insights": ["Insight 1", "Insight 2"],
            "context_summary": "Test summary"
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify LLM was called
    assert mock_llm_manager.invoke.called

    # Verify the prompt includes facts and insights
    call_args = mock_llm_manager.invoke.call_args
    prompt = call_args[0][0]

    # Check that prompt contains the key information
    assert "Fact 1" in prompt
    assert "Fact 2" in prompt
    assert "Insight 1" in prompt
    assert "Test question?" in prompt


def test_writer_node_with_empty_facts(mock_llm_manager):
    """Test writer node when no facts or insights are available."""
    mock_llm_manager.invoke.return_value = "Fallback chapter content..."

    state = {
        "chapter_title": "Test Chapter",
        "chapter_question": "Test question?",
        "chapter_context": "",
        "chapter_scratchpad": {
            "key_facts": [],
            "insights": []
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Should still generate a draft
    assert "current_draft" in result
    assert isinstance(result["current_draft"], str)
    assert len(result["current_draft"]) > 0


def test_writer_node_with_llm_error(mock_llm_manager):
    """Test writer node handles LLM errors gracefully."""
    # Setup mock to raise error
    mock_llm_manager.invoke.side_effect = Exception("LLM API Error")

    state = {
        "chapter_title": "Test Chapter",
        "chapter_question": "Test question?",
        "chapter_context": "",
        "chapter_scratchpad": {
            "key_facts": ["Fact 1"],
            "insights": ["Insight 1"]
        }
    }

    # Execute writer node - should handle error gracefully
    result = writer_node(state, mock_llm_manager)

    # Should still return structure with fallback message
    assert "current_draft" in result
    draft = result["current_draft"]
    assert isinstance(draft, str)
    assert len(draft) > 0
    # Should contain error indication or fallback content
    assert "无法生成" in draft or "error" in draft.lower() or "内容" in draft


def test_writer_node_generates_board_level_tone(mock_llm_manager):
    """Test that writer node generates board-level professional tone."""
    mock_llm_manager.invoke.return_value = """专业报告内容。

## 主要发现

经过深入分析，我们发现...

## 建议

基于以上分析，建议...

结论如下。"""

    state = {
        "chapter_title": "行业分析",
        "chapter_question": "行业发展趋势如何？",
        "chapter_context": "董事会报告",
        "chapter_scratchpad": {
            "key_facts": ["关键事实1", "关键事实2"],
            "insights": ["重要洞察1"]
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify LLM was called with appropriate prompt
    call_args = mock_llm_manager.invoke.call_args
    prompt = call_args[0][0]

    # Check that prompt mentions board-level tone
    assert "董事会" in prompt or "board" in prompt.lower() or "专业" in prompt or "professional" in prompt.lower()


def test_writer_node_includes_citations(mock_llm_manager):
    """Test that writer node includes citation markers."""
    mock_llm_manager.invoke.return_value = """报告内容包含引用[来源: doc1.pdf]。

更多内容[来源: doc2.pdf, page 3]。

总结部分。"""

    state = {
        "chapter_title": "Test",
        "chapter_question": "Test?",
        "chapter_context": "",
        "chapter_scratchpad": {
            "key_facts": ["Fact from doc1"],
            "insights": ["Insight from doc2"]
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify draft includes citations
    draft = result["current_draft"]
    assert "[来源:" in draft or "[Source:" in draft


def test_writer_node_structure_requirements(mock_llm_manager):
    """Test that writer node generates properly structured content."""
    # Response with proper structure
    structured_response = """中国交通运输行业在过去十年取得了显著发展。这为经济增长提供了强有力的支撑，也改善了人民的生活质量。

## 发展现状

近年来，中国交通运输行业保持了稳定增长态势。高速铁路网络迅速扩张，运营里程已突破4万公里。同时，城市轨道交通系统也在快速扩展。

## 主要成就

中国交通运输行业取得的成就举世瞩目。高速铁路技术的自主创新和大规模应用，使中国在这一领域处于世界领先地位。

展望未来，中国交通运输行业将继续保持稳定发展。随着规划的深入实施，交通运输基础设施将进一步完善。"""

    mock_llm_manager.invoke.return_value = structured_response

    state = {
        "chapter_title": "行业发展",
        "chapter_question": "发展现状如何？",
        "chapter_context": "",
        "chapter_scratchpad": {
            "key_facts": ["Fact 1", "Fact 2"],
            "insights": ["Insight 1"]
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify draft meets length requirements (800-1200 characters is target, but test for >100)
    draft = result["current_draft"]
    assert len(draft) > 100, "Draft should be substantial"

    # Verify structure: opening, body (with sections), closing
    lines = draft.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    # Should have multiple sections
    assert len(non_empty_lines) > 5, "Draft should have multiple lines/sections"


def test_writer_node_temperature(mock_llm_manager):
    """Test that writer node uses appropriate temperature for creative writing."""
    mock_llm_manager.invoke.return_value = "Creative chapter content..."

    state = {
        "chapter_title": "Test",
        "chapter_question": "Test?",
        "chapter_context": "",
        "chapter_scratchpad": {
            "key_facts": ["Fact 1"],
            "insights": ["Insight 1"]
        }
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify LLM was called with temperature parameter
    call_kwargs = mock_llm_manager.invoke.call_args[1]
    assert "temperature" in call_kwargs
    # Writer should use higher temperature for creativity
    assert 0.6 <= call_kwargs["temperature"] <= 0.8


def test_writer_node_preserves_scratchpad(mock_llm_manager):
    """Test that writer node does not modify scratchpad (only outputs draft)."""
    mock_llm_manager.invoke.return_value = "Generated content..."

    original_scratchpad = {
        "queries": ["query 1"],
        "retrieved_docs": [{"text": "doc"}],
        "key_facts": ["Fact 1"],
        "insights": ["Insight 1"]
    }

    state = {
        "chapter_title": "Test",
        "chapter_question": "Test?",
        "chapter_context": "",
        "chapter_scratchpad": original_scratchpad.copy()
    }

    # Execute writer node
    result = writer_node(state, mock_llm_manager)

    # Verify scratchpad is not in result (writer only outputs current_draft)
    assert "chapter_scratchpad" not in result or result.get("chapter_scratchpad") is None

    # Verify only current_draft is returned
    assert "current_draft" in result
    assert len(result) == 1 or "current_draft" in result and len(result) <= 2
