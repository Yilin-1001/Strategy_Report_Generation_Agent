import pytest
from langchain_core.documents import Document
from rag_project.data_loader.metadata_extractor import MetadataExtractor, extract_core_metadata

def test_extract_core_metadata_minimal():
    """Test extracting minimal core metadata"""
    doc = Document(page_content="Test content", metadata={})

    metadata = extract_core_metadata(doc, "news", "test.txt")

    assert 'doc_id' in metadata
    assert metadata['doc_type'] == 'news'
    assert metadata['source'] == 'test.txt'
    assert isinstance(metadata['doc_id'], str)

def test_extract_from_filename_with_date():
    """Test extracting date from filename"""
    extractor = MetadataExtractor()

    filename = "全省网约车平台上线_2025-02-20 16_21.txt"
    metadata = extractor.extract_from_filename(filename)

    assert 'title' in metadata
    assert 'publish_date' in metadata

def test_extract_from_filename_no_date():
    """Test extracting from filename without date"""
    extractor = MetadataExtractor()

    filename = "simple_news.txt"
    metadata = extractor.extract_from_filename(filename)

    assert 'title' in metadata

def test_add_doc_type_metadata():
    """Test adding document type to existing metadata"""
    doc = Document(
        page_content="Test",
        metadata={"existing": "value"}
    )

    enhanced = MetadataExtractor.add_doc_type_metadata(doc, "news")

    assert enhanced.metadata['doc_type'] == 'news'
    assert enhanced.metadata['existing'] == 'value'

def test_remove_none_values():
    """Test removing None values from metadata"""
    metadata = {
        'doc_id': '123',
        'title': 'Test',
        'publish_date': None,
        'page_number': None,
    }

    cleaned = MetadataExtractor.remove_none_values(metadata)

    assert 'doc_id' in cleaned
    assert 'title' in cleaned
    assert 'publish_date' not in cleaned
    assert 'page_number' not in cleaned
