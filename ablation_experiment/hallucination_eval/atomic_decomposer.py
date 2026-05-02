"""
Atomic Fact Decomposer Module

Decompose chapter text into atomic facts using GLM-5.1.
Uses verified V2 prompt with broadened FACT definition.
Aligned with FActScore methodology (Min et al., 2023).
"""

import re
import time
from typing import List
from dataclasses import dataclass
from openai import OpenAI

from .config import EVAL_API_KEY, EVAL_BASE_URL, DECOMPOSE_MODEL, DECOMPOSE_TIMEOUT, DECOMPOSE_MAX_TOKENS, LLM_CALL_INTERVAL


@dataclass
class AtomicFact:
    """An atomic fact extracted from text."""
    text: str           # The fact text
    fact_type: str      # "FACT" or "ANALYSIS"
    citation: str       # Citation text if any (e.g., "中国交通运输2021_merged"), empty if none
    source_index: int   # Index in the original decomposition list


# V2 Prompt (verified through testing)
ATOMIC_DECOMPOSE_SYSTEM = "你是一个严谨的事实核查专家，擅长将复杂文本分解为可验证的原子事实。严格按照分类规则标注每个条目。"

ATOMIC_DECOMPOSE_PROMPT = """你是一个事实核查专家。请将以下文本分解为独立的原子事实。

【原子事实定义】
原子事实是一个最简短的、包含单一信息的断言。只要涉及以下任何一类信息，就应标注为 [FACT]：
- 具体实体：人名、机构名、文档名、法规名称、项目名称等
- 具体数据：数字、比例、排名、时间等
- 具体事件：某人说某话、某政策出台、某会议召开等
- 具体状态/性质：某事物被描述为某种状态（如"是基础性产业"）
- 因果/逻辑断言：A导致B、A要求B等可验证的因果关系

只有以下情况才标注为 [ANALYSIS]：
- 纯主观评价（"展现了强大能力"、"具有重要意义"）
- 未来预测/建议（"应加大力度"、"需要重点关注"）
- 方法论描述（"本章将运用XX模型"）
- 纯过渡/连接语句

【引用规则】
- 仅当原文中该句明确包含 [来源:xxx] 标注时，才在原子事实中保留该引用
- 不要为没有引用的原子事实添加引用

【示例】
原文："2024年，集团在省厅考核中排名第二，获评突出单位，展现了强大的战略执行能力。"
分解：
1. [FACT] 集团在2024年省厅考核中排名第二
2. [FACT] 集团获评2024年度全省交通运输工作表现突出单位
3. [ANALYSIS] 集团展现了强大的战略执行能力

原文："交通运输作为国民经济的基础性、先导性产业，其发展环境正经历深刻变革。"
分解：
1. [FACT] 交通运输是国民经济的基础性产业
2. [FACT] 交通运输是国民经济的先导性产业
3. [FACT] 交通运输发展环境正经历深刻变革

原文："本章将运用PEST模型，重点从政策与经济维度剖析宏观环境。"
分解：
1. [ANALYSIS] 本章将运用PEST模型分析宏观环境

【要求】
1. 每个原子事实只包含一个可验证的信息点
2. 保留原文中的关键实体和数据，不要概括或改写
3. 不要遗漏任何事实性断言
4. 格式：序号. [FACT/ANALYSIS] 内容 [来源:xxx]（如有引用）

待分解文本：
{text}

原子事实列表："""


class AtomicDecomposer:
    """Decompose text into atomic facts using LLM."""

    def __init__(self):
        self.client = OpenAI(
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL
        )
        self._last_call_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < LLM_CALL_INTERVAL:
            time.sleep(LLM_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    def decompose(self, text: str) -> List[AtomicFact]:
        """
        Decompose text into atomic facts.

        Args:
            text: Chapter or section text

        Returns:
            List of AtomicFact objects
        """
        self._rate_limit()

        prompt = ATOMIC_DECOMPOSE_PROMPT.format(text=text)

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=DECOMPOSE_MODEL,
                    messages=[
                        {"role": "system", "content": ATOMIC_DECOMPOSE_SYSTEM},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=DECOMPOSE_MAX_TOKENS,
                    timeout=DECOMPOSE_TIMEOUT
                )

                result = response.choices[0].message.content
                return self._parse_result(result)
            except Exception as e:
                if attempt < 2:
                    wait = 10 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise

    def _parse_result(self, result: str) -> List[AtomicFact]:
        """Parse LLM response into AtomicFact objects."""
        facts = []

        for line in result.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Determine type
            if '[FACT]' in line:
                fact_type = "FACT"
            elif '[ANALYSIS]' in line:
                fact_type = "ANALYSIS"
            else:
                # Skip lines without type markers
                continue

            # Extract citation if present
            citation = ""
            cite_match = re.search(r'\[来源[：:]\s*([^\]]+)\]', line)
            if cite_match:
                citation = cite_match.group(1).strip()

            # Clean the fact text: remove number prefix, type marker, and citation
            text = line
            # Remove leading number and dot
            text = re.sub(r'^\d+\.\s*', '', text)
            # Remove type marker
            text = text.replace('[FACT]', '').replace('[ANALYSIS]', '').strip()
            # Remove citation from text (we store it separately)
            text = re.sub(r'\s*\[来源[：:]\s*[^\]]+\]', '', text).strip()

            if text:
                facts.append(AtomicFact(
                    text=text,
                    fact_type=fact_type,
                    citation=citation,
                    source_index=len(facts)
                ))

        return facts

    def decompose_chapter(self, chapter_text: str, max_chunk_chars: int = 3000) -> List[AtomicFact]:
        """
        Decompose a full chapter, splitting into chunks if too long.

        Args:
            chapter_text: Full chapter text
            max_chunk_chars: Maximum characters per decomposition call

        Returns:
            Combined list of AtomicFact objects
        """
        # If text is short enough, decompose directly
        if len(chapter_text) <= max_chunk_chars:
            return self.decompose(chapter_text)

        # Split into paragraphs and batch
        paragraphs = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]

        all_facts = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > max_chunk_chars and current_chunk:
                # Process current chunk
                chunk_facts = self.decompose(current_chunk)
                all_facts.extend(chunk_facts)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # Process remaining
        if current_chunk:
            chunk_facts = self.decompose(current_chunk)
            all_facts.extend(chunk_facts)

        return all_facts