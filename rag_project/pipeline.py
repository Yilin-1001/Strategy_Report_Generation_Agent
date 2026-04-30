import uuid
from typing import List, Dict, Optional
from pathlib import Path
from langchain_core.documents import Document

from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from rag_project.data_loader.metadata_extractor import MetadataExtractor
from rag_project.data_loader.chunk_storage import ChunkStorage
from rag_project.embeddings.embedding_model import EmbeddingModel
from rag_project.embeddings.embedding_client import EmbeddingClient
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.logger import logger

class RAGPipeline:
    """Complete RAG processing pipeline"""

    def __init__(
        self,
        chunking_config_path: str = "config/chunking_config.yaml",
        milvus_config_path: str = "config/milvus_config.yaml",
        chunks_storage_path: Optional[str] = None,
        knowledge_base_path: str = "知识库/知识库"
    ):
        """
        Initialize RAG pipeline

        Args:
            chunking_config_path: Path to chunking configuration
            milvus_config_path: Path to Milvus configuration
            chunks_storage_path: Optional path to save chunks before embedding
            knowledge_base_path: Path to knowledge base directory (for tags extraction)
        """
        # 将knowledge_base_path转换为绝对路径
        kb_path = Path(knowledge_base_path)
        base_dir = str(kb_path.resolve()) if kb_path.exists() else knowledge_base_path

        self.chunker = ConfigurableChunker(
            config_path=chunking_config_path,
            base_dir=base_dir
        )

        # 根据配置选择embedding模式
        from rag_project.utils.config_loader import load_config
        config = load_config(milvus_config_path)
        embedding_mode = config.get('embedding', {}).get('mode', 'local')

        if embedding_mode == 'api':
            logger.info("Using Embedding API client (faster startup)")
            self.embedding_model = EmbeddingClient(milvus_config_path)
        else:
            logger.info("Using local Embedding model")
            self.embedding_model = EmbeddingModel(milvus_config_path, load_on_init=False)

        self.milvus_manager = MilvusManager(milvus_config_path)

        # Reranker (optional, disabled by default)
        self.reranker = None
        self.reranker_config = {}
        try:
            reranker_config = load_config("config/reranker_config.yaml").get('reranker', {})
            if reranker_config.get('enabled', False):
                from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
                self.reranker = SiliconFlowReranker("config/reranker_config.yaml")
                self.reranker_config = reranker_config
                logger.info("Reranker enabled")
            else:
                logger.info("Reranker disabled")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")

        self.chunk_storage = ChunkStorage()
        self.chunks_storage_path = chunks_storage_path
        self.knowledge_base_path = base_dir

        logger.info(f"RAG Pipeline initialized with knowledge_base: {self.knowledge_base_path}")

    def index_documents(self, file_paths: List[str]) -> int:
        """
        Index documents: load -> chunk -> embed -> store

        Args:
            file_paths: List of file paths to index

        Returns:
            Number of chunks indexed
        """
        all_chunks = []

        # Step 1: Load and chunk documents
        logger.info(f"Loading {len(file_paths)} documents...")

        for file_path in file_paths:
            chunks = self._load_and_chunk_file(file_path)
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} chunks total")

        if not all_chunks:
            logger.warning("No chunks generated. Check input files.")
            return 0

        # Step 2: Save chunks (optional)
        if self.chunks_storage_path:
            self.chunk_storage.save_chunks_to_json(all_chunks, self.chunks_storage_path)
            logger.info(f"Chunks saved to {self.chunks_storage_path}")

        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.embed_documents(all_chunks)
        logger.info(f"Generated embeddings: {embeddings.shape}")

        # Step 4: Prepare and insert into Milvus
        logger.info("Inserting into Milvus...")
        milvus_data = self._prepare_milvus_data(all_chunks, embeddings)
        self.milvus_manager.insert_data(milvus_data)

        logger.info(f"Indexing complete: {len(all_chunks)} chunks")

        return len(all_chunks)

    def _load_and_chunk_file(self, file_path: str) -> List[Document]:
        """Load single file and split into chunks"""
        # Display file being processed
        file_name = Path(file_path).name
        logger.info(f"[Processing] {file_name}")

        # Skip DOCX and DOC files
        if file_path.lower().endswith(('.docx', '.doc')):
            logger.info(f"[Skipped] {file_name} -> DOCX/DOC files skipped")
            return []

        # Skip PDF files (assumed already converted to TXT)
        if file_path.lower().endswith('.pdf'):
            logger.info(f"[Skipped] {file_name} -> PDF files skipped (use converted TXT instead)")
            return []

        try:
            # Detect document type
            doc_type = detect_doc_type(file_path)

            # Load document
            loader = get_loader_for_file(file_path)
            documents = loader.load()

            # Add source to metadata - 保存完整路径用于tags提取
            source = Path(file_path).name  # 文件名（用于显示）
            full_path = str(Path(file_path).resolve())  # 完整路径（用于tags提取）

            for doc in documents:
                doc.metadata['source'] = source
                doc.metadata['full_path'] = full_path  # 添加完整路径
                doc.metadata['doc_type'] = doc_type

            # Extract and add metadata
            for doc in documents:
                core_metadata = MetadataExtractor.extract_core_metadata(doc, doc_type, source)
                doc.metadata.update(core_metadata)

            # Split into chunks
            chunks = self.chunker.split_documents(documents, doc_type)

            logger.info(f"[Completed] {file_name} -> {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"[ERROR] {file_name}: {e}")
            return []

    def _prepare_milvus_data(
        self,
        chunks: List[Document],
        embeddings
    ) -> List[Dict]:
        """Prepare data for Milvus insertion"""
        milvus_data = []

        for chunk, embedding in zip(chunks, embeddings):
            milvus_data.append({
                "id": chunk.metadata.get('doc_id', str(uuid.uuid4())),
                "vector": embedding.tolist(),
                "text": chunk.page_content,
                "doc_type": chunk.metadata.get("doc_type", "unknown"),
                "source": chunk.metadata.get("source", ""),
                "publish_date": chunk.metadata.get("publish_date"),
                "page_number": chunk.metadata.get("page_number", 0),
                "title": chunk.metadata.get("title", ""),
            })

        return milvus_data

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_reranker: bool = False
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"doc_type": ["news"]})
            use_reranker: Whether to apply reranking (requires reranker enabled)

        Returns:
            List of search results
        """
        # Generate query embedding
        query_vector = self.embedding_model.embed_text(query)

        # Determine retrieval count
        if use_reranker and self.reranker:
            expansion = self.reranker_config.get('expansion_factor', 3)
            retrieve_k = top_k * expansion
            logger.info(f"Rerank mode: retrieving {retrieve_k} -> reranking to {top_k}")
        else:
            retrieve_k = top_k

        # Search Milvus
        results = self.milvus_manager.search(
            query_vector.tolist(),
            top_k=retrieve_k,
            filters=filters
        )

        # Apply reranking if requested and available
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results

    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        milvus_stats = self.milvus_manager.get_collection_stats()

        return {
            "milvus_collection": milvus_stats,
            "embedding_model": self.embedding_model.get_model_info(),
        }
