import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger

class EmbeddingModel:
    """Wrapper for BGE-M3 embedding model"""

    def __init__(
        self,
        config_path: str = "config/milvus_config.yaml",
        load_on_init: bool = True
    ):
        """
        Initialize embedding model

        Args:
            config_path: Path to configuration file
            load_on_init: Whether to load model on initialization
        """
        self.config = load_config(config_path)
        self.embedding_config = self.config.get('embedding', {})
        self.model = None

        if load_on_init:
            self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        if self.model is not None:
            return

        model_name = self.embedding_config.get('model_name', 'BAAI/bge-m3')
        device = self.embedding_config.get('device', 'cpu')
        cache_dir = self.embedding_config.get('cache_dir', './data/models')

        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir
        )

        logger.info(f"Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            self._load_model()

        normalize = self.embedding_config.get('normalize_embeddings', True)

        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts

        Args:
            texts: List of input texts

        Returns:
            Embedding vectors as numpy array (N, D)
        """
        if self.model is None:
            self._load_model()

        batch_size = self.embedding_config.get('batch_size', 32)
        normalize = self.embedding_config.get('normalize_embeddings', True)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True
        )

        return embeddings

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Embed LangChain documents

        Args:
            documents: List of LangChain Document objects

        Returns:
            Embedding vectors as numpy array (N, D)
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            self._load_model()

        return {
            'model_name': self.embedding_config.get('model_name'),
            'dimension': self.model.get_sentence_embedding_dimension(),
            'device': self.embedding_config.get('device'),
            'max_length': self.embedding_config.get('max_length'),
            'normalize_embeddings': self.embedding_config.get('normalize_embeddings'),
        }
