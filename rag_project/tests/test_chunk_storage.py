import pytest
import json
import os
from langchain_core.documents import Document
from rag_project.data_loader.chunk_storage import ChunkStorage

def test_save_chunks_to_json(tmp_path):
    """Test saving chunks to JSON file"""
    storage = ChunkStorage()

    documents = [
        Document(page_content="First chunk", metadata={"id": "1", "doc_type": "news"}),
        Document(page_content="Second chunk", metadata={"id": "2", "doc_type": "news"}),
    ]

    output_path = tmp_path / "chunks.json"
    storage.save_chunks_to_json(documents, str(output_path))

    assert os.path.exists(output_path)

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]['text'] == "First chunk"
    assert data[0]['metadata']['doc_type'] == "news"

def test_load_chunks_from_json(tmp_path):
    """Test loading chunks from JSON file"""
    storage = ChunkStorage()

    # First save some chunks
    chunks_data = [
        {"id": "1", "text": "Test 1", "metadata": {"doc_type": "news"}, "char_count": 6},
        {"id": "2", "text": "Test 2", "metadata": {"doc_type": "pdf"}, "char_count": 6},
    ]

    json_path = tmp_path / "test_chunks.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False)

    # Now load them
    documents = storage.load_chunks_from_json(str(json_path))

    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].page_content == "Test 1"
    assert documents[0].metadata['doc_type'] == "news"

def test_save_and_load_roundtrip(tmp_path):
    """Test roundtrip: save -> load -> save"""
    storage = ChunkStorage()

    original_docs = [
        Document(page_content="Original text", metadata={"id": "1", "source": "test.txt"}),
    ]

    # Save
    json_path = tmp_path / "roundtrip.json"
    storage.save_chunks_to_json(original_docs, str(json_path))

    # Load
    loaded_docs = storage.load_chunks_from_json(str(json_path))

    # Verify
    assert len(loaded_docs) == len(original_docs)
    assert loaded_docs[0].page_content == original_docs[0].page_content
    assert loaded_docs[0].metadata['source'] == original_docs[0].metadata['source']
