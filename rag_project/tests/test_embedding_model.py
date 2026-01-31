import pytest
import numpy as np
from rag_project.embeddings.embedding_model import EmbeddingModel

def test_embed_single_text():
    """Test embedding a single text"""
    model = EmbeddingModel()

    text = "这是一段测试文本"
    embedding = model.embed_text(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1024,)  # BGE-M3 dimension
    assert np.allclose(np.linalg.norm(embedding), 1.0)  # Normalized

def test_embed_batch_texts():
    """Test embedding a batch of texts"""
    model = EmbeddingModel()

    texts = ["文本1", "文本2", "文本3"]
    embeddings = model.embed_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 1024)

def test_embed_documents():
    """Test embedding LangChain documents"""
    model = EmbeddingModel()

    from langchain_core.documents import Document
    documents = [
        Document(page_content="文档1"),
        Document(page_content="文档2"),
    ]

    embeddings = model.embed_documents(documents)

    assert embeddings.shape == (2, 1024)

def test_model_lazy_loading():
    """Test that model is loaded on first use"""
    model = EmbeddingModel(load_on_init=False)

    assert model.model is None

    # First use triggers loading
    model.embed_text("测试")

    assert model.model is not None

def test_get_model_info():
    """Test getting model information"""
    model = EmbeddingModel()

    info = model.get_model_info()

    assert 'model_name' in info
    assert 'dimension' in info
    assert info['dimension'] == 1024
