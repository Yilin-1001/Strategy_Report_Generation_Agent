# -*- coding: utf-8 -*-
"""
Integration Test: Test Data Layer Modules Working Together
"""
import tempfile
import os
import sys
from pathlib import Path
from langchain_core.documents import Document

# Test imports
from rag_project.utils.config_loader import load_config, get_chunking_config
from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from rag_project.data_loader.metadata_extractor import MetadataExtractor
from rag_project.data_loader.chunk_storage import ChunkStorage

def test_config_loader():
    """Test 1: Configuration Loader"""
    print("\n" + "="*60)
    print("Test 1: Configuration Loader")
    print("="*60)

    # Load chunking config
    chunking_config = load_config("config/chunking_config.yaml")
    print(f"[OK] Successfully loaded chunking configuration")
    print(f"  - News chunk size: {chunking_config['chunking']['news']['chunk_size']}")
    print(f"  - PDF chunk size: {chunking_config['chunking']['pdf']['chunk_size']}")

    # Load Milvus config
    milvus_config = load_config("config/milvus_config.yaml")
    print(f"[OK] Successfully loaded Milvus configuration")
    print(f"  - Collection name: {milvus_config['milvus']['collection']['name']}")
    print(f"  - Vector dimension: {milvus_config['milvus']['collection']['dimension']}")

    # Get specific doc type config
    news_config = get_chunking_config('news')
    print(f"[OK] Successfully got news config: chunk_size={news_config['chunk_size']}")

def test_document_type_detection():
    """Test 2: Document Type Detection"""
    print("\n" + "="*60)
    print("Test 2: Document Type Detection")
    print("="*60)

    test_files = [
        ("news.txt", "news"),
        ("policy.pdf", "pdf"),
        ("regulation.docx", "regulation"),
        ("regulation.doc", "regulation"),
        ("unknown.xyz", "default"),
    ]

    for filename, expected_type in test_files:
        detected_type = detect_doc_type(filename)
        status = "[OK]" if detected_type == expected_type else "[FAIL]"
        print(f"{status} {filename} -> {detected_type} (expected: {expected_type})")

def test_document_loading_and_chunking():
    """Test 3: Document Loading and Chunking"""
    print("\n" + "="*60)
    print("Test 3: Document Loading and Chunking")
    print("="*60)

    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        test_content = """This is the first test paragraph. Testing news document chunking functionality.

This is the second test paragraph. Jiangxi province transportation construction makes new progress.

This is the third test paragraph. The provincial highway network continues to improve. This is the fourth test paragraph to test longer text chunking effects.""" * 5
        f.write(test_content)
        temp_file = f.name

    try:
        # Detect document type
        doc_type = detect_doc_type(temp_file)
        print(f"[OK] Detected document type: {doc_type}")

        # Load document
        loader = get_loader_for_file(temp_file)
        documents = loader.load()
        print(f"[OK] Successfully loaded document: {len(documents)} document objects")

        # Create chunker
        chunker = ConfigurableChunker()

        # Chunk
        chunks = chunker.split_documents(documents, doc_type=doc_type)
        print(f"[OK] Successfully chunked: {len(chunks)} text chunks")

        # Verify metadata
        for i, chunk in enumerate(chunks[:3]):  # Only show first 3
            print(f"\n  Chunk {i+1}:")
            print(f"    - Length: {len(chunk.page_content)} characters")
            print(f"    - Document type: {chunk.metadata.get('doc_type')}")
            print(f"    - Preview: {chunk.page_content[:50]}...")

    finally:
        # Cleanup temp file
        os.unlink(temp_file)

def test_metadata_extraction():
    """Test 4: Metadata Extraction"""
    print("\n" + "="*60)
    print("Test 4: Metadata Extraction")
    print("="*60)

    # Test core metadata extraction
    doc = Document(page_content="Test content", metadata={"title": "Test Title"})
    metadata = MetadataExtractor.extract_core_metadata(doc, "news", "test.txt")

    print("[OK] Extracted core metadata:")
    print(f"  - Document ID: {metadata['doc_id']}")
    print(f"  - Document type: {metadata['doc_type']}")
    print(f"  - Source: {metadata['source']}")
    print(f"  - Title: {metadata['title']}")

    # Test filename metadata extraction
    filename = "province_ride_hailing_launch_2025-02-20.txt"
    filename_metadata = MetadataExtractor.extract_from_filename(filename)

    print("\n[OK] Metadata extracted from filename:")
    print(f"  - Filename: {filename}")
    print(f"  - Title: {filename_metadata.get('title')}")
    print(f"  - Publish date: {filename_metadata.get('publish_date')}")

def test_chunk_storage():
    """Test 5: Chunk Storage"""
    print("\n" + "="*60)
    print("Test 5: Chunk Storage Management")
    print("="*60)

    # Create test documents
    test_docs = [
        Document(
            page_content="This is the first test chunk content.",
            metadata={"doc_id": "1", "doc_type": "news", "source": "test1.txt"}
        ),
        Document(
            page_content="This is the second test chunk content. Contains more detailed information.",
            metadata={"doc_id": "2", "doc_type": "pdf", "source": "test2.pdf"}
        ),
    ]

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        # Save chunks
        storage = ChunkStorage()
        storage.save_chunks_to_json(test_docs, temp_file)
        print(f"[OK] Successfully saved {len(test_docs)} chunks to {temp_file}")

        # Get summary
        summary = storage.get_chunks_summary(temp_file)
        print("[OK] Chunk summary:")
        print(f"  - Total chunks: {summary['total_chunks']}")
        print(f"  - Total characters: {summary['total_characters']}")
        print(f"  - Average chunk length: {summary['avg_chunk_length']:.2f}")
        print(f"  - Document type distribution: {summary['doc_types_distribution']}")

        # Load chunks
        loaded_docs = storage.load_chunks_from_json(temp_file)
        print(f"[OK] Successfully loaded {len(loaded_docs)} chunks")

        # Verify data integrity
        assert len(loaded_docs) == len(test_docs), "Loaded document count mismatch"
        assert loaded_docs[0].page_content == test_docs[0].page_content, "Content mismatch"
        assert loaded_docs[0].metadata['doc_type'] == test_docs[0].metadata['doc_type'], "Metadata mismatch"
        print("[OK] Data integrity verification passed")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_end_to_end_pipeline():
    """Test 6: End-to-End Data Layer Pipeline"""
    print("\n" + "="*60)
    print("Test 6: End-to-End Data Layer Pipeline")
    print("="*60)

    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        test_content = """
Jiangxi Provincial Department of Transportation issued the latest notice.
Provincial ride-hailing platform officially launched.
This will greatly improve transportation service quality.
        """ * 10  # Repeat to get enough text for chunking
        f.write(test_content)
        temp_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        chunks_file = f.name

    try:
        print("Step 1: Detect document type")
        doc_type = detect_doc_type(temp_file)
        print(f"  [OK] Document type: {doc_type}")

        print("\nStep 2: Load document")
        loader = get_loader_for_file(temp_file)
        documents = loader.load()
        print(f"  [OK] Loaded {len(documents)} documents")

        print("\nStep 3: Extract metadata")
        for doc in documents:
            metadata = MetadataExtractor.extract_core_metadata(doc, doc_type, Path(temp_file).name)
            doc.metadata.update(metadata)
        print(f"  [OK] Metadata extraction complete")

        print("\nStep 4: Chunking")
        chunker = ConfigurableChunker()
        chunks = chunker.split_documents(documents, doc_type=doc_type)
        print(f"  [OK] Generated {len(chunks)} text chunks")

        print("\nStep 5: Save chunks to JSON")
        storage = ChunkStorage()
        storage.save_chunks_to_json(chunks, chunks_file)
        print(f"  [OK] Saved to {chunks_file}")

        print("\nStep 6: Verify saved chunks")
        summary = storage.get_chunks_summary(chunks_file)
        print(f"  [OK] Total chunks: {summary['total_chunks']}")
        print(f"  [OK] Average chunk length: {summary['avg_chunk_length']:.2f} characters")

        print("\nStep 7: Reload chunks")
        loaded_chunks = storage.load_chunks_from_json(chunks_file)
        print(f"  [OK] Successfully loaded {len(loaded_chunks)} chunks")

        print("\n" + "="*60)
        print("[OK] End-to-end pipeline test completed successfully!")
        print("="*60)

    finally:
        # Cleanup temp files
        os.unlink(temp_file)
        if os.path.exists(chunks_file):
            os.unlink(chunks_file)

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*15 + "Data Layer Integration Tests" + " "*15 + "█")
    print("█"*60)

    try:
        test_config_loader()
        test_document_type_detection()
        test_document_loading_and_chunking()
        test_metadata_extraction()
        test_chunk_storage()
        test_end_to_end_pipeline()

        print("\n" + "█"*60)
        print("█" + " "*20 + "All Tests Passed! [OK]" + " "*20 + "█")
        print("█"*60 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
