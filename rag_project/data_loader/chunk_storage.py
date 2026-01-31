import json
import uuid
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from rag_project.utils.logger import logger

class ChunkStorage:
    """Manage storage of document chunks before embedding"""

    def save_chunks_to_json(
        self,
        documents: List[Document],
        output_path: str
    ) -> None:
        """
        Save chunks to JSON file

        Args:
            documents: List of LangChain Document objects
            output_path: Path to output JSON file
        """
        chunks_data = []

        for doc in documents:
            chunk_data = {
                'id': doc.metadata.get('doc_id', str(uuid.uuid4())),
                'text': doc.page_content,
                'metadata': doc.metadata,
                'char_count': len(doc.page_content),
                'created_at': datetime.now().isoformat(),
            }
            chunks_data.append(chunk_data)

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks_data)} chunks to {output_path}")

    def load_chunks_from_json(self, json_path: str) -> List[Document]:
        """
        Load chunks from JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            List of LangChain Document objects
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        documents = []
        for chunk in chunks_data:
            doc = Document(
                page_content=chunk['text'],
                metadata=chunk['metadata']
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks from {json_path}")

        return documents

    def get_chunks_summary(self, json_path: str) -> Dict[str, any]:
        """
        Get summary information about chunks file

        Args:
            json_path: Path to chunks JSON file

        Returns:
            Summary dictionary
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        total_chunks = len(chunks_data)
        total_chars = sum(c['char_count'] for c in chunks_data)
        avg_chunk_length = total_chars / total_chunks if total_chunks > 0 else 0

        doc_types = {}
        for chunk in chunks_data:
            doc_type = chunk['metadata'].get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'avg_chunk_length': avg_chunk_length,
            'doc_types_distribution': doc_types,
        }
