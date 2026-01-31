from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger

class ConfigurableChunker:
    """Configurable document chunker with YAML-based configuration"""

    def __init__(self, config_path: str = "config/chunking_config.yaml"):
        """
        Initialize configurable chunker

        Args:
            config_path: Path to chunking configuration YAML file
        """
        self.config_path = config_path
        self.splitters = self._create_splitters()
        logger.info(f"ConfigurableChunker initialized with {len(self.splitters)} splitters")

    def _create_splitters(self) -> Dict[str, RecursiveCharacterTextSplitter]:
        """Create text splitters based on configuration"""
        config = load_config(self.config_path)
        chunking_config = config.get('chunking', {})

        splitters = {}

        for doc_type, params in chunking_config.items():
            if doc_type == 'advanced':
                continue

            try:
                splitters[doc_type] = RecursiveCharacterTextSplitter(
                    chunk_size=params['chunk_size'],
                    chunk_overlap=params['chunk_overlap'],
                    separators=params['separators'],
                    length_function=len,
                )
                logger.debug(f"Created splitter for {doc_type}: chunk_size={params['chunk_size']}")
            except Exception as e:
                logger.warning(f"Failed to create splitter for {doc_type}: {e}")

        return splitters

    def split_documents(
        self,
        documents: List[Document],
        doc_type: str = 'default'
    ) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split
            doc_type: Document type (news, pdf, regulation, default)

        Returns:
            List of chunked documents
        """
        splitter = self.splitters.get(doc_type, self.splitters.get('default'))

        if not splitter:
            logger.warning(f"No splitter found for {doc_type}, using default")
            splitter = self.splitters.get('default')

        if not splitter:
            raise ValueError("No default splitter configured")

        chunks = splitter.split_documents(documents)

        # Add doc_type to metadata
        for chunk in chunks:
            chunk.metadata['doc_type'] = doc_type

        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks (type={doc_type})")

        return chunks

    def reload_config(self, config_path: str = None):
        """
        Reload configuration from file

        Args:
            config_path: Optional path to new configuration file
        """
        config_path = config_path or self.config_path
        self.config_path = config_path
        self.splitters = self._create_splitters()
        logger.info(f"Configuration reloaded from {config_path}")
