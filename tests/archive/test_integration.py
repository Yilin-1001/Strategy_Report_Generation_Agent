"""
Complete RAG Pipeline Integration Test
Tests the entire pipeline from document loading to search
"""
import sys
import tempfile
import os
from pathlib import Path

# Add rag_project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_complete_pipeline():
    """Test complete RAG pipeline end-to-end"""
    print("="*80)
    print("Complete RAG Pipeline Integration Test")
    print("="*80)

    from rag_project.pipeline import RAGPipeline
    from rag_project.data_loader.chunk_storage import ChunkStorage
    from langchain_core.documents import Document

    # Step 1: Create test documents
    print("\n[Step 1] Creating test documents...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test files
        news_file = Path(tmp_dir) / "news.txt"
        news_file.write_text(
            "江西省交通运输厅发布最新政策。"
            "高速公路建设将加速推进。"
            "预计投资100亿元用于基础设施建设。"
            "这是江西省交通发展的重要里程碑。"
            "项目将在2025年正式启动。",
            encoding='utf-8'
        )

        report_file = Path(tmp_dir) / "report.txt"
        report_file.write_text(
            "2024年江西省交通发展报告\n\n"
            "一、总体情况\n"
            "2024年，江西省交通运输业持续快速发展。\n\n"
            "二、高速公路建设\n"
            "全省新增高速公路里程500公里。\n\n"
            "三、未来展望\n"
            "预计2025年投资将增长20%。",
            encoding='utf-8'
        )

        chunks_output = Path(tmp_dir) / "chunks.json"

        file_paths = [str(news_file), str(report_file)]
        print(f"[OK] Created {len(file_paths)} test documents")

        # Step 2: Initialize pipeline
        print("\n[Step 2] Initializing RAG Pipeline...")
        pipeline = RAGPipeline(chunks_storage_path=str(chunks_output))
        print("[OK] Pipeline initialized")

        # Step 3: Index documents
        print("\n[Step 3] Indexing documents...")
        print(f"  Files to index: {len(file_paths)}")

        try:
            chunk_count = pipeline.index_documents(file_paths)
            print(f"[OK] Indexed {chunk_count} chunks")
        except Exception as e:
            print(f"[FAIL] Indexing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 4: Verify chunks were saved
        print("\n[Step 4] Verifying chunk storage...")
        if os.path.exists(chunks_output):
            storage = ChunkStorage()
            summary = storage.get_chunks_summary(str(chunks_output))
            print(f"[OK] Chunks saved:")
            print(f"  - Total chunks: {summary['total_chunks']}")
            print(f"  - Avg length: {summary['avg_chunk_length']:.1f} characters")
            print(f"  - Distribution: {summary['doc_types_distribution']}")
        else:
            print(f"[WARN] Chunks file not found at {chunks_output}")

        # Step 5: Get pipeline stats
        print("\n[Step 5] Getting pipeline statistics...")
        try:
            stats = pipeline.get_pipeline_stats()
            print(f"[OK] Pipeline stats:")
            print(f"  - Collection: {stats['milvus_collection']['name']}")
            print(f"  - Documents: {stats['milvus_collection']['num_entities']}")
            print(f"  - Model: {stats['embedding_model']['model_name']}")
            print(f"  - Dimension: {stats['embedding_model']['dimension']}")
        except Exception as e:
            print(f"[WARN] Could not get stats (Milvus may not be running): {e}")

        # Step 6: Test search functionality
        print("\n[Step 6] Testing search functionality...")
        try:
            # Test 1: Basic search
            print("  Test 1: Basic search for '高速公路'")
            results = pipeline.search("高速公路", top_k=3)
            print(f"  [OK] Found {len(results)} results")

            for i, result in enumerate(results[:2], 1):
                print(f"    Result {i}:")
                print(f"      - Score: {result['score']:.4f}")
                print(f"      - Type: {result['metadata']['doc_type']}")
                print(f"      - Source: {result['metadata']['source']}")
                print(f"      - Text: {result['text'][:50]}...")

            # Test 2: Search with filters
            if len(results) > 0:
                print("\n  Test 2: Search with doc_type filter")
                results_filtered = pipeline.search(
                    "交通",
                    top_k=3,
                    filters={"doc_type": ["news"]}
                )
                print(f"  [OK] Found {len(results_filtered)} filtered results")

        except Exception as e:
            print(f"[FAIL] Search failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*80)
    print("[SUCCESS] Complete pipeline test passed!")
    print("="*80)
    return True

def test_component_integration():
    """Test integration between components"""
    print("\n" + "="*80)
    print("Component Integration Test")
    print("="*80)

    from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
    from rag_project.data_loader.configurable_splitter import ConfigurableChunker
    from rag_project.data_loader.metadata_extractor import MetadataExtractor
    from langchain_core.documents import Document

    # Test 1: Document type detection -> loader
    print("\n[Test 1] Document Type Detection -> Loader")
    file_path = "test_news.txt"
    doc_type = detect_doc_type(file_path)
    loader = get_loader_for_file(file_path)
    print(f"  [OK] File: {file_path} -> Type: {doc_type} -> Loader: {type(loader).__name__}")

    # Test 2: Metadata extractor
    print("\n[Test 2] Metadata Extractor")
    doc = Document(page_content="Test", metadata={})
    metadata = MetadataExtractor.extract_core_metadata(doc, "news", "test.txt")
    assert 'doc_id' in metadata
    assert metadata['doc_type'] == 'news'
    print(f"  [OK] Metadata extracted: doc_id={metadata['doc_id'][:8]}..., doc_type={metadata['doc_type']}")

    # Test 3: Configurable splitter
    print("\n[Test 3] Configurable Splitter")
    splitter = ConfigurableChunker()
    doc = Document(page_content="这是一段测试文本。" * 10, metadata={})
    chunks = splitter.split_documents([doc], doc_type='news')
    print(f"  [OK] Split into {len(chunks)} chunks")
    print(f"  - First chunk length: {len(chunks[0].page_content)} characters")
    print(f"  - Metadata: doc_type={chunks[0].metadata.get('doc_type')}")

    # Test 4: Chunk storage roundtrip
    print("\n[Test 4] Chunk Storage Roundtrip")
    from rag_project.data_loader.chunk_storage import ChunkStorage
    import tempfile

    storage = ChunkStorage()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test.json"

        # Save
        storage.save_chunks_to_json(chunks, str(test_file))
        print(f"  [OK] Saved {len(chunks)} chunks")

        # Load
        loaded_chunks = storage.load_chunks_from_json(str(test_file))
        assert len(loaded_chunks) == len(chunks)
        print(f"  [OK] Loaded {len(loaded_chunks)} chunks")

        # Verify
        assert loaded_chunks[0].page_content == chunks[0].page_content
        print(f"  [OK] Content preserved after roundtrip")

    print("\n" + "="*80)
    print("[SUCCESS] Component integration test passed!")
    print("="*80)
    return True

def main():
    """Run all integration tests"""
    print("\n")
    print("="*80)
    print("RAG PROJECT - INTEGRATION TEST SUITE")
    print("="*80)
    print("\n")

    results = []

    # Test component integration
    results.append(("Component Integration", test_component_integration()))

    # Test complete pipeline (may fail if Milvus not running)
    results.append(("Complete Pipeline", test_complete_pipeline()))

    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n*** All integration tests passed! ***")
        return 0
    else:
        print(f"\n*** {total - passed} test(s) failed ***")
        return 1

if __name__ == "__main__":
    sys.exit(main())
