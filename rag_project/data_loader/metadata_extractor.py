import uuid
import re
from typing import Dict
from langchain_core.documents import Document
from rag_project.utils.logger import logger

class MetadataExtractor:
    """Metadata extractor for documents (MVP - core metadata only)"""

    @staticmethod
    def extract_core_metadata(
        doc: Document,
        doc_type: str,
        source: str
    ) -> Dict[str, any]:
        """
        Extract core metadata from document

        Args:
            doc: LangChain Document object
            doc_type: Document type (news, pdf, regulation)
            source: Source file path or name

        Returns:
            Dictionary with core metadata
        """
        metadata = {
            'doc_id': str(uuid.uuid4()),
            'doc_type': doc_type,
            'source': source,
            'title': doc.metadata.get('title', source),
        }

        # Add publish_date if available
        if 'publish_date' in doc.metadata:
            metadata['publish_date'] = doc.metadata['publish_date']

        # Add page_number if available (PDF)
        if 'page_number' in doc.metadata:
            metadata['page_number'] = doc.metadata['page_number']

        return MetadataExtractor.remove_none_values(metadata)

    @staticmethod
    def extract_from_filename(filename: str) -> Dict[str, any]:
        """
        Extract metadata from filename

        Args:
            filename: Filename (can include path)

        Returns:
            Dictionary with extracted metadata
        """
        import os
        basename = os.path.basename(filename)

        metadata = {}

        # Extract title (part before first underscore or extension)
        title_part = basename.split('_')[0] if '_' in basename else basename.rsplit('.', 1)[0]
        metadata['title'] = title_part

        # Extract date (pattern: YYYY-MM-DD)
        date_pattern = r'(\d{4})-(\d{2})-(\d{2})'
        date_match = re.search(date_pattern, basename)

        if date_match:
            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            metadata['publish_date'] = date_str

        return metadata

    @staticmethod
    def add_doc_type_metadata(doc: Document, doc_type: str) -> Document:
        """
        Add document type to document metadata

        Args:
            doc: LangChain Document
            doc_type: Document type to add

        Returns:
            Document with updated metadata
        """
        doc.metadata['doc_type'] = doc_type
        return doc

    @staticmethod
    def remove_none_values(metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Remove None values from metadata dictionary

        Args:
            metadata: Metadata dictionary

        Returns:
            Cleaned metadata dictionary
        """
        return {k: v for k, v in metadata.items() if v is not None}

# Convenience functions for backward compatibility
def extract_core_metadata(doc: Document, doc_type: str, source: str) -> Dict[str, any]:
    """Convenience function for extracting core metadata"""
    return MetadataExtractor.extract_core_metadata(doc, doc_type, source)
