"""
RAG Pipeline Test (without Milvus dependency)
Tests pipeline logic without requiring Milvus server
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

def test_pipeline_without_milvus():
    """Test pipeline logic without Milvus connection"""
    print("="*80)
    print("RAG Pipeline Logic Test (Mock Milvus)")
    print("="*80)

    from rag_project.pipeline import RAGPipeline
    from langchain_core.documents import Document

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test documents
        news_file = Path(tmp_dir) / "news.txt"
        news_file.write_text(
            "江西省交通运输厅发布最新政策。"
            "高速公路建设将加速推进。",
            encoding='utf-8'
        )

        chunks_output = Path(tmp_dir) / "chunks.json"

        # Test: Loading and chunking
        print("\n[Test 1] Document Loading and Chunking")
        from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
        from rag_project.data_loader.configurable_splitter import ConfigurableChunker

        doc_type = detect_doc_type(str(news_file))
        loader = get_loader_for_file(str(news_file))
        docs = loader.load()
        print(f"  [OK] Loaded {len(docs)} documents")
        print(f"  [OK] Document type: {doc_type}")

        splitter = ConfigurableChunker()
        chunks = splitter.split_documents(docs, doc_type)
        print(f"  [OK] Split into {len(chunks)} chunks")

        # Test: Metadata extraction
        print("\n[Test 2] Metadata Extraction")
        from rag_project.data_loader.metadata_extractor import MetadataExtractor

        for doc in docs:
            metadata = MetadataExtractor.extract_core_metadata(doc, doc_type, str(news_file.name))
            doc.metadata.update(metadata)

        print(f"  [OK] Metadata added to documents")

        # Test: Chunk storage
        print("\n[Test 3] Chunk Storage")
        from rag_project.data_loader.chunk_storage import ChunkStorage

        storage = ChunkStorage()
        storage.save_chunks_to_json(chunks, str(chunks_output))
        print(f"  [OK] Saved {len(chunks)} chunks to {chunks_output.name}")

        loaded_chunks = storage.load_chunks_from_json(str(chunks_output))
        assert len(loaded_chunks) == len(chunks)
        print(f"  [OK] Loaded {len(loaded_chunks)} chunks")

        # Test: Embedding generation
        print("\n[Test 4] Embedding Generation")
        from rag_project.embeddings.embedding_model import EmbeddingModel
        import numpy as np

        embedding_model = EmbeddingModel(load_on_init=False)
        print(f"  [INFO] Loading BGE-M3 model (may take time on first run)...")

        # Test with sample text
        sample_texts = ["测试文本1", "测试文本2"]
        embeddings = embedding_model.embed_texts(sample_texts)
        assert embeddings.shape == (2, 1024)
        print(f"  [OK] Generated embeddings: {embeddings.shape}")

        # Test normalization
        norms = [np.linalg.norm(emb) for emb in embeddings for emb in [emb]]
        assert all(abs(n - 1.0) < 0.001 for n in norms)
        print(f"  [OK] Embeddings are normalized")

        # Test: Milvus data preparation
        print("\n[Test 5] Milvus Data Preparation")
        import uuid

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

        print(f"  [OK] Prepared {len(milvus_data)} records for Milvus")
        print(f"  - Sample record:")
        print(f"    - id: {milvus_data[0]['id'][:8]}...")
        print(f"    - text: {milvus_data[0]['text'][:30]}...")
        print(f"    - vector dimension: {len(milvus_data[0]['vector'])}")

        # Verify data structure
        assert all('id' in record for record in milvus_data)
        assert all('vector' in record for record in milvus_data)
        assert all('text' in record for record in milvus_data)
        assert all(len(record['vector']) == 1024 for record in milvus_data)
        print(f"  [OK] All records have required fields")

    print("\n" + "="*80)
    print("[SUCCESS] Pipeline logic test passed!")
    print("="*80)
    return True

def main():
    print("\n")
    print("="*80)
    print("RAG PROJECT - PIPELINE LOGIC TEST (NO MILVUS REQUIRED)")
    print("="*80)
    print("\n")

    result = test_pipeline_without_milvus()

    if result:
        print("\n*** All pipeline logic tests passed! ***")
        print("\nNote: Full end-to-end test requires Milvus server to be running.")
        print("You can start Milvus with Docker:")
        print("  docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest")
        return 0
    else:
        print("\n*** Some tests failed ***")
        return 1

if __name__ == "__main__":
    sys.exit(main())
