import pytest
from langchain_core.documents import Document
from rag_project.data_loader.configurable_splitter import ConfigurableChunker

def test_chunk_news_document():
    """Test chunking news documents with news configuration"""
    splitter = ConfigurableChunker()

    doc = Document(page_content="这是一段测试文本。用来测试新闻文档的分块功能。")
    chunks = splitter.split_documents([doc], doc_type='news')

    assert len(chunks) >= 1
    assert all(isinstance(chunk, Document) for chunk in chunks)

def test_chunk_pdf_document():
    """Test chunking PDF documents with PDF configuration"""
    splitter = ConfigurableChunker()

    doc = Document(page_content="这是一段较长的测试文本。用来测试PDF文档的分块功能。PDF文档通常包含更多的内容。" * 10)
    chunks = splitter.split_documents([doc], doc_type='pdf')

    assert len(chunks) >= 1
    assert all(isinstance(chunk, Document) for chunk in chunks)

def test_splitter_uses_correct_config():
    """Test that splitter uses correct configuration for document type"""
    from rag_project.utils.config_loader import get_chunking_config

    splitter = ConfigurableChunker()

    # Test news config
    news_config = get_chunking_config('news')
    assert splitter.splitters['news'].chunk_size == news_config['chunk_size']
    assert splitter.splitters['news'].chunk_overlap == news_config['chunk_overlap']

def test_splitter_adds_doc_type_metadata():
    """Test that splitter adds doc_type to metadata"""
    splitter = ConfigurableChunker()

    doc = Document(page_content="测试文本", metadata={})
    chunks = splitter.split_documents([doc], doc_type='news')

    assert all('doc_type' in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata['doc_type'] == 'news' for chunk in chunks)

def test_reload_config():
    """Test reloading configuration"""
    splitter = ConfigurableChunker()
    old_chunk_size = splitter.splitters['news'].chunk_size

    # Reload config (should work even with same file)
    splitter.reload_config()

    # Splitter should be re-created with same config
    assert splitter.splitters['news'].chunk_size == old_chunk_size
