import pytest
import numpy as np
from pymilvus import connections, Collection
from rag_project.storage.milvus_manager import MilvusManager

@pytest.fixture(scope="module")
def milvus_manager():
    """Create Milvus manager for testing"""
    # Note: This test requires Milvus to be running
    try:
        manager = MilvusManager(collection_name="test_collection")
        yield manager
        # Cleanup
        manager.drop_collection()
    except Exception as e:
        pytest.skip(f"Milvus not available: {e}")

def test_create_collection(milvus_manager):
    """Test creating Milvus collection"""
    assert milvus_manager.collection is not None
    assert milvus_manager.collection.name == "test_collection"

def test_insert_data(milvus_manager):
    """Test inserting data into collection"""
    test_data = [
        {
            "id": "test_1",
            "vector": np.random.rand(1024).tolist(),
            "text": "Test document 1",
            "doc_type": "news",
            "source": "test.txt",
            "publish_date": 1708454400,
            "page_number": 1,
            "title": "Test Title 1",
        },
        {
            "id": "test_2",
            "vector": np.random.rand(1024).tolist(),
            "text": "Test document 2",
            "doc_type": "pdf",
            "source": "test.pdf",
            "publish_date": 1708540800,
            "page_number": 2,
            "title": "Test Title 2",
        },
    ]

    count = milvus_manager.insert_data(test_data)
    assert count == 2

def test_search_without_filters(milvus_manager):
    """Test searching without filters"""
    query_vector = np.random.rand(1024).tolist()
    results = milvus_manager.search(query_vector, top_k=2)

    assert len(results) <= 2
    assert all('score' in r for r in results)

def test_collection_stats(milvus_manager):
    """Test getting collection statistics"""
    stats = milvus_manager.get_collection_stats()

    assert 'num_entities' in stats
    assert stats['num_entities'] >= 0
