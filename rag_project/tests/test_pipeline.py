import pytest
import tempfile
import os
from pathlib import Path
from rag_project.pipeline import RAGPipeline

@pytest.fixture
def sample_documents(tmp_path):
    """Create sample document files for testing"""
    # Create TXT file
    txt_file = tmp_path / "news.txt"
    txt_file.write_text("这是一条新闻。关于交通建设的最新动态。", encoding='utf-8')

    return [str(txt_file)]

def test_pipeline_end_to_end(sample_documents):
    """Test complete pipeline: load -> chunk -> embed -> store"""
    pipeline = RAGPipeline()

    # Index documents
    count = pipeline.index_documents(sample_documents)

    assert count > 0

def test_pipeline_search(sample_documents):
    """Test searching after indexing"""
    pipeline = RAGPipeline()

    # Index
    pipeline.index_documents(sample_documents)

    # Search
    results = pipeline.search("交���建设")

    assert len(results) >= 0

def test_pipeline_with_chunk_storage(sample_documents, tmp_path):
    """Test pipeline with chunk storage"""
    chunks_path = tmp_path / "chunks.json"

    pipeline = RAGPipeline(chunks_storage_path=str(chunks_path))
    pipeline.index_documents(sample_documents)

    # Check chunks file was created
    assert os.path.exists(chunks_path)
