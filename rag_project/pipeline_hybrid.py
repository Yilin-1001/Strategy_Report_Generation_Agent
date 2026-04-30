"""
Hybrid RAG Pipeline — Dense Vector + BM25 Full-Text Search.

Drop-in replacement for RAGPipeline that uses MilvusHybridManager
for hybrid retrieval. Reuses existing chunking and embedding logic.
"""

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
from rag_project.storage.milvus_hybrid_manager import MilvusHybridManager
from rag_project.utils.logger import logger


class HybridRAGPipeline:
    """RAG Pipeline with hybrid (dense + BM25) retrieval."""

    def __init__(
        self,
        chunking_config_path: str = "config/chunking_config.yaml",
        milvus_config_path: str = "config/milvus_config.yaml",
        knowledge_base_path: str = "知识库/知识库",
    ):
        kb_path = Path(knowledge_base_path)
        base_dir = str(kb_path.resolve()) if kb_path.exists() else knowledge_base_path

        self.chunker = ConfigurableChunker(
            config_path=chunking_config_path,
            base_dir=base_dir,
        )

        # Embedding model (same as original)
        from rag_project.utils.config_loader import load_config
        config = load_config(milvus_config_path)
        embedding_mode = config.get("embedding", {}).get("mode", "local")
        if embedding_mode == "api":
            self.embedding_model = EmbeddingClient(milvus_config_path)
        else:
            self.embedding_model = EmbeddingModel(milvus_config_path, load_on_init=False)

        # Hybrid Milvus manager (dense + BM25)
        self.milvus_manager = MilvusHybridManager(
            config_path=milvus_config_path,
        )

        # Reranker (optional)
        self.reranker = None
        self.reranker_config = {}
        try:
            reranker_cfg = load_config("config/reranker_config.yaml").get("reranker", {})
            if reranker_cfg.get("enabled", False):
                from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
                self.reranker = SiliconFlowReranker("config/reranker_config.yaml")
                self.reranker_config = reranker_cfg
                logger.info("Reranker enabled")
            else:
                logger.info("Reranker disabled")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")

        self.chunk_storage = ChunkStorage()
        self.chunks_storage_path = None
        self.knowledge_base_path = base_dir

        logger.info(f"HybridRAGPipeline initialized (dense + BM25)")

    # ── Indexing (reuses original chunking/embedding) ──────────────────

    def index_documents(self, file_paths: List[str]) -> int:
        all_chunks = []
        for file_path in file_paths:
            chunks = self._load_and_chunk_file(file_path)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks generated.")
            return 0

        embeddings = self.embedding_model.embed_documents(all_chunks)

        milvus_data = self._prepare_milvus_data(all_chunks, embeddings)
        self.milvus_manager.insert_data(milvus_data)

        logger.info(f"Hybrid index complete: {len(all_chunks)} chunks")
        return len(all_chunks)

    def _load_and_chunk_file(self, file_path: str) -> List[Document]:
        file_name = Path(file_path).name
        if file_path.lower().endswith((".docx", ".doc", ".pdf")):
            return []
        try:
            doc_type = detect_doc_type(file_path)
            loader = get_loader_for_file(file_path)
            documents = loader.load()
            source = Path(file_path).name
            full_path = str(Path(file_path).resolve())
            for doc in documents:
                doc.metadata["source"] = source
                doc.metadata["full_path"] = full_path
                doc.metadata["doc_type"] = doc_type
            for doc in documents:
                core_meta = MetadataExtractor.extract_core_metadata(doc, doc_type, source)
                doc.metadata.update(core_meta)
            chunks = self.chunker.split_documents(documents, doc_type)
            logger.info(f"[Completed] {file_name} -> {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"[ERROR] {file_name}: {e}")
            return []

    def _prepare_milvus_data(self, chunks: List[Document], embeddings) -> List[Dict]:
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            data.append({
                "id": chunk.metadata.get("doc_id", str(uuid.uuid4())),
                "vector": embedding.tolist(),
                "text": chunk.page_content,
                "doc_type": chunk.metadata.get("doc_type", "unknown"),
                "source": chunk.metadata.get("source", ""),
                "publish_date": chunk.metadata.get("publish_date") or 0,
                "page_number": chunk.metadata.get("page_number", 0),
                "title": chunk.metadata.get("title", ""),
            })
        return data

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_reranker: bool = False,
    ) -> List[Dict]:
        # Embed query for dense search
        query_vector = self.embedding_model.embed_text(query)

        # Determine retrieval count for reranker
        if use_reranker and self.reranker:
            expansion = self.reranker_config.get("expansion_factor", 3)
            retrieve_k = top_k * expansion
        else:
            retrieve_k = top_k

        # Hybrid search (dense + BM25)
        results = self.milvus_manager.search(
            query_vector=query_vector.tolist(),
            query_text=query,
            top_k=retrieve_k,
            filters=filters,
        )

        # Apply reranking if requested
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results

    def get_pipeline_stats(self) -> Dict:
        return {
            "milvus_collection": self.milvus_manager.get_collection_stats(),
            "embedding_model": self.embedding_model.get_model_info(),
            "search_type": "hybrid_dense_bm25",
        }
