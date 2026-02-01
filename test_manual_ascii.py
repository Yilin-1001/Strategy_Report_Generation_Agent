"""
Manual test script to verify RAG project implementation (ASCII only)
"""
import sys
from pathlib import Path

# Add rag_project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from rag_project.utils.config_loader import load_config, get_chunking_config
        print("[OK] config_loader imported")
    except Exception as e:
        print(f"[FAIL] config_loader failed: {e}")
        return False

    try:
        from rag_project.utils.logger import logger, setup_logger
        print("[OK] logger imported")
    except Exception as e:
        print(f"[FAIL] logger failed: {e}")
        return False

    try:
        from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
        print("[OK] document_type_detector imported")
    except Exception as e:
        print(f"[FAIL] document_type_detector failed: {e}")
        return False

    try:
        from rag_project.data_loader.configurable_splitter import ConfigurableChunker
        print("[OK] configurable_splitter imported")
    except Exception as e:
        print(f"[FAIL] configurable_splitter failed: {e}")
        return False

    try:
        from rag_project.data_loader.metadata_extractor import MetadataExtractor, extract_core_metadata
        print("[OK] metadata_extractor imported")
    except Exception as e:
        print(f"[FAIL] metadata_extractor failed: {e}")
        return False

    try:
        from rag_project.data_loader.chunk_storage import ChunkStorage
        print("[OK] chunk_storage imported")
    except Exception as e:
        print(f"[FAIL] chunk_storage failed: {e}")
        return False

    return True

def test_config_loader():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("Testing config_loader...")
    print("="*60)

    from rag_project.utils.config_loader import load_config, get_chunking_config

    try:
        # Test chunking config
        config = load_config("config/chunking_config.yaml")
        assert "chunking" in config
        assert config["chunking"]["news"]["chunk_size"] == 512
        print("[OK] load_chunking_config() works")

        # Test milvus config
        config = load_config("config/milvus_config.yaml")
        assert config["milvus"]["collection"]["dimension"] == 1024
        print("[OK] load_milvus_config() works")

        # Test get_chunking_config
        news_config = get_chunking_config('news')
        assert news_config['chunk_size'] == 512
        print("[OK] get_chunking_config() works")

        return True
    except Exception as e:
        print(f"[FAIL] Config loader tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_type_detector():
    """Test document type detection"""
    print("\n" + "="*60)
    print("Testing document_type_detector...")
    print("="*60)

    from rag_project.data_loader.document_type_detector import detect_doc_type

    try:
        assert detect_doc_type("news.txt") == "news"
        assert detect_doc_type("policy.pdf") == "pdf"
        assert detect_doc_type("regulation.docx") == "regulation"
        assert detect_doc_type("unknown.xyz") == "default"
        print("[OK] detect_doc_type() works")

        return True
    except Exception as e:
        print(f"[FAIL] Document type detector tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configurable_splitter():
    """Test configurable splitter"""
    print("\n" + "="*60)
    print("Testing configurable_splitter...")
    print("="*60)

    from rag_project.data_loader.configurable_splitter import ConfigurableChunker
    from langchain_core.documents import Document

    try:
        splitter = ConfigurableChunker()

        # Test with news document
        doc = Document(page_content="这是一段测试文本。用来测试新闻文档的分块功能。")
        chunks = splitter.split_documents([doc], doc_type='news')
        assert len(chunks) >= 1
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
        assert all(chunk.metadata.get('doc_type') == 'news' for chunk in chunks)
        print(f"[OK] Split document into {len(chunks)} chunks")

        # Test config usage
        from rag_project.utils.config_loader import get_chunking_config
        news_config = get_chunking_config('news')
        assert splitter.splitters['news']._chunk_size == news_config['chunk_size']
        print("[OK] Config loaded correctly")

        return True
    except Exception as e:
        print(f"[FAIL] Configurable splitter tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_extractor():
    """Test metadata extractor"""
    print("\n" + "="*60)
    print("Testing metadata_extractor...")
    print("="*60)

    from rag_project.data_loader.metadata_extractor import MetadataExtractor, extract_core_metadata
    from langchain_core.documents import Document

    try:
        # Test extract_core_metadata
        doc = Document(page_content="Test content", metadata={})
        metadata = extract_core_metadata(doc, "news", "test.txt")

        assert 'doc_id' in metadata
        assert metadata['doc_type'] == 'news'
        assert metadata['source'] == 'test.txt'
        assert isinstance(metadata['doc_id'], str)
        print("[OK] extract_core_metadata() works")

        # Test extract_from_filename
        filename = "全省网约车平台上线_2025-02-20 16_21.txt"
        file_metadata = MetadataExtractor.extract_from_filename(filename)
        assert 'title' in file_metadata
        assert 'publish_date' in file_metadata
        print("[OK] extract_from_filename() works")

        # Test remove_none_values
        metadata = {'doc_id': '123', 'title': 'Test', 'publish_date': None}
        cleaned = MetadataExtractor.remove_none_values(metadata)
        assert 'doc_id' in cleaned
        assert 'title' in cleaned
        assert 'publish_date' not in cleaned
        print("[OK] remove_none_values() works")

        return True
    except Exception as e:
        print(f"[FAIL] Metadata extractor tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunk_storage():
    """Test chunk storage"""
    print("\n" + "="*60)
    print("Testing chunk_storage...")
    print("="*60)

    from rag_project.data_loader.chunk_storage import ChunkStorage
    from langchain_core.documents import Document
    import tempfile
    import os

    try:
        storage = ChunkStorage()

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "test_chunks.json")

            # Test save
            documents = [
                Document(page_content="First chunk", metadata={"doc_type": "news"}),
                Document(page_content="Second chunk", metadata={"doc_type": "news"}),
            ]
            storage.save_chunks_to_json(documents, test_file)
            assert os.path.exists(test_file)
            print("[OK] save_chunks_to_json() works")

            # Test load
            loaded_docs = storage.load_chunks_from_json(test_file)
            assert len(loaded_docs) == 2
            assert loaded_docs[0].page_content == "First chunk"
            print("[OK] load_chunks_from_json() works")

            # Test get_chunks_summary
            summary = storage.get_chunks_summary(test_file)
            assert summary['total_chunks'] == 2
            print(f"[OK] get_chunks_summary() works: {summary}")

        return True
    except Exception as e:
        print(f"[FAIL] Chunk storage tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("RAG Project Manual Test Suite")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    if results[-1][1]:  # Only continue if imports work
        results.append(("Config Loader", test_config_loader()))
        results.append(("Document Type Detector", test_document_type_detector()))
        results.append(("Configurable Splitter", test_configurable_splitter()))
        results.append(("Metadata Extractor", test_metadata_extractor()))
        results.append(("Chunk Storage", test_chunk_storage()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n*** All tests passed! ***")
        return 0
    else:
        print(f"\n*** {total - passed} test(s) failed ***")
        return 1

if __name__ == "__main__":
    sys.exit(main())
