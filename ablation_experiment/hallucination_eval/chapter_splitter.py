"""
Chapter Splitter Module

Split markdown report into 8 chapters for per-chapter evaluation.
Uses regex to detect chapter boundaries (# 第X章).
"""

import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chapter:
    """A chapter from the report."""
    index: int           # 0-7
    title: str           # Chapter title
    phase: str           # "diagnosis" or "initiatives"
    content: str         # Full chapter text


def split_report(markdown_text: str) -> List[Chapter]:
    """
    Split a markdown report into 8 chapters.

    Args:
        markdown_text: Full report markdown content

    Returns:
        List of 8 Chapter objects
    """
    # Remove cover page and table of contents (before first chapter)
    # Pattern: # 第一章 or # 第1章
    chapter_pattern = r'^#\s*第([一二三四五六七八]|1|2|3|4|5|6|7|8)章'

    # Find all chapter start positions
    lines = markdown_text.split('\n')
    chapter_starts = []

    for i, line in enumerate(lines):
        match = re.match(chapter_pattern, line)
        if match:
            # Convert Chinese numeral to index
            numeral = match.group(1)
            index_map = {
                '一': 0, '二': 1, '三': 2, '四': 3,
                '五': 4, '六': 5, '七': 6, '八': 7,
                '1': 0, '2': 1, '3': 2, '4': 3,
                '5': 4, '6': 5, '7': 6, '8': 7
            }
            index = index_map.get(numeral, -1)
            if index >= 0:
                chapter_starts.append((index, i, line))

    # Sort by index to ensure correct order
    chapter_starts.sort(key=lambda x: x[0])

    # Extract chapter content
    chapters = []
    phase_map = {
        0: "diagnosis", 1: "diagnosis", 2: "diagnosis",
        3: "initiatives", 4: "initiatives", 5: "initiatives",
        6: "initiatives", 7: "initiatives"
    }

    for idx, (chapter_idx, start_line, title_line) in enumerate(chapter_starts):
        # Find end: next chapter start or end of document
        if idx < len(chapter_starts) - 1:
            end_line = chapter_starts[idx + 1][1]
        else:
            end_line = len(lines)

        # Extract content
        content_lines = lines[start_line:end_line]
        # Remove trailing --- separators
        while content_lines and content_lines[-1].strip() == "---":
            content_lines.pop()

        content = '\n'.join(content_lines)

        chapters.append(Chapter(
            index=chapter_idx,
            title=title_line.strip(),
            phase=phase_map[chapter_idx],
            content=content
        ))

    return chapters


def validate_chapters(chapters: List[Chapter]) -> bool:
    """Validate that we have exactly 8 chapters."""
    if len(chapters) != 8:
        return False
    for i, ch in enumerate(chapters):
        if ch.index != i:
            return False
    return True


def get_report_metadata(markdown_text: str) -> Dict[str, str]:
    """
    Extract metadata from report header.

    Returns:
        Dict with 'generation_time' and 'topic' keys
    """
    lines = markdown_text.split('\n')

    metadata = {
        "generation_time": "",
        "topic": ""
    }

    # Look for metadata in first 20 lines
    for i, line in enumerate(lines[:20]):
        # 生成时间: 2026年04月12日
        if '**生成时间**' in line or '生成时间' in line:
            # Extract date
            match = re.search(r'(\d{4}年\d{2}月\d{2}日)', line)
            if match:
                metadata["generation_time"] = match.group(1)

        # 主题: xxx
        if '**主题**' in line or '主题' in line:
            # Extract topic after colon
            match = re.search(r'主题[：:\s]*(.+)', line)
            if match:
                metadata["topic"] = match.group(1).strip()

    return metadata