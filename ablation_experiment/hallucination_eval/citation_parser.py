"""
Citation Parser Module

Parse [来源:xxx] citations from text and verify authenticity against KB.
"""

import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

from .config import KB_CHUNKS_PATH


@dataclass
class Citation:
    """A citation found in text."""
    raw_text: str        # Original citation text like "中国交通运输2021_merged"
    position: int        # Character position in original text
    is_valid: bool       # Whether the source exists in KB
    source_file: str     # Matching source file in KB (if found)


class CitationParser:
    """Parse and verify citations from text."""

    def __init__(self, kb_chunks_path: Path = None):
        """
        Initialize parser with KB chunks index.

        Args:
            kb_chunks_path: Path to all_chunks.json
        """
        self.kb_chunks_path = kb_chunks_path or KB_CHUNKS_PATH
        self.source_index = self._build_source_index()

    def _build_source_index(self) -> Dict[str, List[dict]]:
        """Build index of source files to their chunks."""
        if not self.kb_chunks_path.exists():
            return {}

        chunks = json.load(open(self.kb_chunks_path, 'r', encoding='utf-8'))

        index = {}
        for chunk in chunks:
            source = chunk.get('metadata', {}).get('source', '')
            if source:
                if source not in index:
                    index[source] = []
                index[source].append(chunk)

        return index

    def parse_citations(self, text: str) -> List[Citation]:
        """
        Extract all [来源:xxx] citations from text.

        Args:
            text: Text to parse

        Returns:
            List of Citation objects
        """
        # Pattern: [来源: xxx] or [来源：xxx]
        pattern = r'\[来源[：:]\s*([^\]]+)\]'

        citations = []
        for match in re.finditer(pattern, text):
            raw_text = match.group(1).strip()
            position = match.start()

            # Verify citation
            is_valid, source_file = self.verify_citation(raw_text)

            citations.append(Citation(
                raw_text=raw_text,
                position=position,
                is_valid=is_valid,
                source_file=source_file
            ))

        return citations

    def verify_citation(self, citation_text: str) -> Tuple[bool, str]:
        """
        Verify if a citation source exists in KB.

        Args:
            citation_text: Citation name like "中国交通运输2021_merged"

        Returns:
            (is_valid, source_file): Whether valid, and matching source if found
        """
        citation_clean = citation_text.strip()

        # Try direct match
        for source_file in self.source_index:
            # Check if citation matches or is substring of source
            source_base = source_file.replace('.txt', '').replace('.pdf', '')

            if citation_clean in source_file or citation_clean in source_base:
                return True, source_file

            # Also check reverse: source in citation
            if source_base in citation_clean:
                return True, source_file

        # No match found
        return False, ""

    def get_cited_chunks(self, citation_text: str, top_k: int = 20) -> List[dict]:
        """
        Get chunks from a cited source document.

        Args:
            citation_text: Citation name
            top_k: Maximum chunks to return

        Returns:
            List of chunks from the cited source
        """
        is_valid, source_file = self.verify_citation(citation_text)

        if not is_valid:
            return []

        chunks = self.source_index.get(source_file, [])
        return chunks[:top_k]

    def get_stats(self) -> Dict:
        """Get statistics about the KB index."""
        return {
            "total_sources": len(self.source_index),
            "total_chunks": sum(len(chunks) for chunks in self.source_index.values()),
            "sources": sorted(self.source_index.keys())
        }