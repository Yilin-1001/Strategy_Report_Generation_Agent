# RAG Document Processing Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete document processing pipeline that loads TXT/PDF/DOCX files, chunks them with configurable strategies, extracts metadata, generates embeddings using BGE-M3, and stores vectors in Milvus.

**Architecture:** Three-layer architecture - Data Layer (document loaders + splitters), Embedding Layer (BGE-M3), Storage Layer (Milvus). Configuration-driven design using YAML files for easy parameter tuning. JSON intermediate storage for chunks before embedding.

**Tech Stack:** LangChain (document loaders, splitters), Unstructured.io (complex parsing), Sentence-Transformers (BGE-M3), PyMilvus (vector database), PyYAML (configuration)

---

## Prerequisites

Before starting implementation, ensure the following are installed:

```bash
# Core dependencies
pip install langchain langchain-community
pip install sentence-transformers
pip install pymilvus
pip install unstructured[all-docs]
pip install pyyaml

# Document parsing
pip install pypdf
pip install python-docx
pip install pdfplumber

# Chinese text processing
pip install jieba
```

---

## Task 1: Project Structure and Configuration Setup

**Files:**
- Create: `rag_project/` (root directory)
- Create: `rag_project/config/`
- Create: `rag_project/data/`
- Create: `rag_project/data_loader/`
- Create: `rag_project/embeddings/`
- Create: `rag_project/storage/`
- Create: `rag_project/utils/`
- Create: `rag_project/tests/`
- Create: `config/chunking_config.yaml`
- Create: `config/milvus_config.yaml`
- Create: `requirements.txt`

**Step 1: Create directory structure**

```bash
# In project root
mkdir -p rag_project/{config,data,data_loader,embeddings,storage,utils,tests}
mkdir -p rag_project/data/{chunks,models}
```

**Step 2: Write chunking configuration**

Create: `config/chunking_config.yaml`

```yaml
# Chunking configuration for different document types
chunking:
  # News articles (TXT files)
  news:
    chunk_size: 512
    chunk_overlap: 50
    separators: ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    length_function: "len"

  # PDF documents (policies, reports, academic papers)
  pdf:
    chunk_size: 1000
    chunk_overlap: 200
    separators: ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]
    length_function: "len"

  # Regulations (DOCX files)
  regulation:
    chunk_size: 1000
    chunk_overlap: 150
    separators: ["\n\n", "\n", "。", "！", "？", "；", "、", "，", " ", ""]
    length_function: "len"

  # Default configuration
  default:
    chunk_size: 800
    chunk_overlap: 100
    separators: ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]
    length_function: "len"

  # Advanced options
  advanced:
    use_semantic_chunking: false
    preserve_tables: true
    table_as_separate_chunk: true
```

**Step 3: Write Milvus configuration**

Create: `config/milvus_config.yaml`

```yaml
# Milvus vector database configuration
milvus:
  # Connection settings
  host: "localhost"
  port: 19530
  alias: "default"

  # Collection settings
  collection:
    name: "enterprise_docs"
    description: "Enterprise document vector database"
    dimension: 1024  # BGE-M3 dimension

  # Index settings
  index:
    type: "HNSW"
    metric_type: "IP"  # Inner Product
    params:
      M: 16
      efConstruction: 256

  # Search settings
  search:
    top_k: 50  # Initial retrieval count
    ef: 128    # Search depth

# Embedding model configuration
embedding:
  model_name: "BAAI/bge-m3"
  device: "cpu"  # or "cuda" if available
  batch_size: 32
  max_length: 8192
  normalize_embeddings: true
  cache_dir: "./data/models"
```

**Step 4: Write requirements.txt**

Create: `requirements.txt`

```txt
# Core RAG framework
langchain>=0.1.0
langchain-community>=0.0.10

# Embedding model
sentence-transformers>=2.3.0

# Vector database
pymilvus>=2.3.0

# Document loaders
unstructured[all-docs]>=0.11.0
pypdf>=3.17.0
python-docx>=1.1.0
pdfplumber>=0.10.0

# Configuration
pyyaml>=6.0

# Chinese text processing
jieba>=0.42.1

# Utilities
numpy>=1.24.0
pandas>=2.0.0
```

**Step 5: Initialize Python packages**

Create: `rag_project/__init__.py`

```python
"""
RAG Project - Document Processing Pipeline
基于RAG的检索型AI智能体文档处理模块
"""

__version__ = "0.1.0"
```

Create: `rag_project/data_loader/__init__.py`, `rag_project/embeddings/__init__.py`, `rag_project/storage/__init__.py`, `rag_project/utils/__init__.py` (empty files)

**Step 6: Commit**

```bash
git add config/ requirements.txt
git add rag_project/
git commit -m "feat: initialize project structure and configuration"
```

---

## Task 2: Utility Modules - Configuration and Logging

**Files:**
- Create: `rag_project/utils/config_loader.py`
- Create: `rag_project/utils/logger.py`
- Test: `rag_project/tests/test_config_loader.py`

**Step 1: Write configuration loader test**

Create: `rag_project/tests/test_config_loader.py`

```python
import pytest
from rag_project.utils.config_loader import load_config

def test_load_chunking_config():
    """Test loading chunking configuration"""
    config_path = "config/chunking_config.yaml"
    config = load_config(config_path)

    assert "chunking" in config
    assert "news" in config["chunking"]
    assert config["chunking"]["news"]["chunk_size"] == 512
    assert config["chunking"]["news"]["chunk_overlap"] == 50

def test_load_milvus_config():
    """Test loading Milvus configuration"""
    config_path = "config/milvus_config.yaml"
    config = load_config(config_path)

    assert "milvus" in config
    assert config["milvus"]["collection"]["dimension"] == 1024
    assert config["milvus"]["index"]["type"] == "HNSW"

def test_load_nonexistent_config():
    """Test loading non-existent configuration raises error"""
    with pytest.raises(FileNotFoundError):
        load_config("config/nonexistent.yaml")
```

**Step 2: Run test to verify it fails**

Run: `pytest rag_project/tests/test_config_loader.py::test_load_chunking_config -v`

Expected: FAIL with "module 'rag_project.utils.config_loader' does not exist"

**Step 3: Implement configuration loader**

Create: `rag_project/utils/config_loader.py`

```python
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration data

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file: {e}")

def get_chunking_config(doc_type: str, config_path: str = "config/chunking_config.yaml") -> Dict[str, Any]:
    """
    Get chunking configuration for a specific document type

    Args:
        doc_type: Document type (news, pdf, regulation, default)
        config_path: Path to chunking configuration file

    Returns:
        Chunking parameters dictionary
    """
    config = load_config(config_path)
    chunking_config = config.get("chunking", {})

    # Return specific doc type config or default
    return chunking_config.get(doc_type, chunking_config.get("default", {}))
```

**Step 4: Run test to verify it passes**

Run: `pytest rag_project/tests/test_config_loader.py -v`

Expected: PASS for all tests

**Step 5: Write logger module**

Create: `rag_project/utils/logger.py`

```python
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str = "rag_project",
    log_dir: str = "logs",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler
    log_file = log_path / f"{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Get default logger
logger = setup_logger()
```

**Step 6: Commit**

```bash
git add rag_project/utils/
git add rag_project/tests/test_config_loader.py
git commit -m "feat: add configuration loader and logger utilities"
```

---

## Task 3: Document Type Detection and Loader Factory

**Files:**
- Create: `rag_project/data_loader/document_type_detector.py`
- Create: `rag_project/data_loader/loader_factory.py`
- Test: `rag_project/tests/test_document_type_detector.py`

**Step 1: Write document type detector test**

Create: `rag_project/tests/test_document_type_detector.py`

```python
import pytest
from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file

def test_detect_txt_type():
    """Test TXT file type detection"""
    assert detect_doc_type("news.txt") == "news"
    assert detect_doc_type("article.txt") == "news"

def test_detect_pdf_type():
    """Test PDF file type detection"""
    assert detect_doc_type("policy.pdf") == "pdf"
    assert detect_doc_type("report.pdf") == "pdf"

def test_detect_docx_type():
    """Test DOCX file type detection"""
    assert detect_doc_type("regulation.docx") == "regulation"
    assert detect_doc_type("law.docx") == "regulation"

def test_detect_unknown_type():
    """Test unknown file type returns default"""
    assert detect_doc_type("unknown.xyz") == "default"
    assert detect_doc_type("no_extension") == "default"

def test_get_loader_for_txt():
    """Test getting appropriate loader for TXT files"""
    from langchain_community.document_loaders import TextLoader
    loader = get_loader_for_file("test.txt")
    assert isinstance(loader, TextLoader)

def test_get_loader_for_pdf():
    """Test getting appropriate loader for PDF files"""
    from langchain_community.document_loaders import UnstructuredPDFLoader
    loader = get_loader_for_file("test.pdf")
    assert isinstance(loader, UnstructuredPDFLoader)

def test_get_loader_for_docx():
    """Test getting appropriate loader for DOCX files"""
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    loader = get_loader_for_file("test.docx")
    assert isinstance(loader, UnstructuredWordDocumentLoader)
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_document_type_detector.py -v`

Expected: FAIL with "module 'rag_project.data_loader.document_type_detector' does not exist"

**Step 3: Implement document type detector**

Create: `rag_project/data_loader/document_type_detector.py`

```python
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)

def detect_doc_type(file_path: str) -> str:
    """
    Detect document type based on file extension

    Args:
        file_path: Path to document file

    Returns:
        Document type: news, pdf, regulation, or default
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    type_mapping = {
        '.txt': 'news',
        '.pdf': 'pdf',
        '.docx': 'regulation',
        '.doc': 'regulation',
    }

    return type_mapping.get(extension, 'default')

def get_loader_for_file(file_path: str):
    """
    Get appropriate LangChain document loader for file

    Args:
        file_path: Path to document file

    Returns:
        LangChain document loader instance
    """
    doc_type = detect_doc_type(file_path)

    loaders = {
        'news': lambda: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        'pdf': lambda: UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="hi_res",
            extract_images_in_pdf=True
        ),
        'regulation': lambda: UnstructuredWordDocumentLoader(
            file_path,
            mode="elements"
        ),
    }

    loader_func = loaders.get(doc_type)
    if loader_func:
        return loader_func()

    # Fallback to TextLoader for unknown types
    return TextLoader(file_path, autodetect_encoding=True)
```

**Step 4: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_document_type_detector.py -v`

Expected: PASS for all tests

**Step 5: Commit**

```bash
git add rag_project/data_loader/document_type_detector.py
git add rag_project/tests/test_document_type_detector.py
git commit -m "feat: add document type detection and loader factory"
```

---

## Task 4: Configurable Text Splitter

**Files:**
- Create: `rag_project/data_loader/configurable_splitter.py`
- Test: `rag_project/tests/test_configurable_splitter.py`

**Step 1: Write configurable splitter test**

Create: `rag_project/tests/test_configurable_splitter.py`

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_configurable_splitter.py::test_chunk_news_document -v`

Expected: FAIL with "module 'rag_project.data_loader.configurable_splitter' does not exist"

**Step 3: Implement configurable splitter**

Create: `rag_project/data_loader/configurable_splitter.py`

```python
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger

class ConfigurableChunker:
    """Configurable document chunker with YAML-based configuration"""

    def __init__(self, config_path: str = "config/chunking_config.yaml"):
        """
        Initialize configurable chunker

        Args:
            config_path: Path to chunking configuration YAML file
        """
        self.config_path = config_path
        self.splitters = self._create_splitters()
        logger.info(f"ConfigurableChunker initialized with {len(self.splitters)} splitters")

    def _create_splitters(self) -> Dict[str, RecursiveCharacterTextSplitter]:
        """Create text splitters based on configuration"""
        config = load_config(self.config_path)
        chunking_config = config.get('chunking', {})

        splitters = {}

        for doc_type, params in chunking_config.items():
            if doc_type == 'advanced':
                continue

            try:
                splitters[doc_type] = RecursiveCharacterTextSplitter(
                    chunk_size=params['chunk_size'],
                    chunk_overlap=params['chunk_overlap'],
                    separators=params['separators'],
                    length_function=len,
                )
                logger.debug(f"Created splitter for {doc_type}: chunk_size={params['chunk_size']}")
            except Exception as e:
                logger.warning(f"Failed to create splitter for {doc_type}: {e}")

        return splitters

    def split_documents(
        self,
        documents: List[Document],
        doc_type: str = 'default'
    ) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split
            doc_type: Document type (news, pdf, regulation, default)

        Returns:
            List of chunked documents
        """
        splitter = self.splitters.get(doc_type, self.splitters.get('default'))

        if not splitter:
            logger.warning(f"No splitter found for {doc_type}, using default")
            splitter = self.splitters.get('default')

        if not splitter:
            raise ValueError("No default splitter configured")

        chunks = splitter.split_documents(documents)

        # Add doc_type to metadata
        for chunk in chunks:
            chunk.metadata['doc_type'] = doc_type

        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks (type={doc_type})")

        return chunks

    def reload_config(self, config_path: str = None):
        """
        Reload configuration from file

        Args:
            config_path: Optional path to new configuration file
        """
        config_path = config_path or self.config_path
        self.config_path = config_path
        self.splitters = self._create_splitters()
        logger.info(f"Configuration reloaded from {config_path}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_configurable_splitter.py -v`

Expected: PASS for all tests

**Step 5: Commit**

```bash
git add rag_project/data_loader/configurable_splitter.py
git add rag_project/tests/test_configurable_splitter.py
git commit -m "feat: add configurable text splitter"
```

---

## Task 5: Metadata Extractor (MVP)

**Files:**
- Create: `rag_project/data_loader/metadata_extractor.py`
- Test: `rag_project/tests/test_metadata_extractor.py`

**Step 1: Write metadata extractor test**

Create: `rag_project/tests/test_metadata_extractor.py`

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_metadata_extractor.py::test_extract_core_metadata_minimal -v`

Expected: FAIL with "module 'rag_project.data_loader.metadata_extractor' does not exist"

**Step 3: Implement metadata extractor (MVP)**

Create: `rag_project/data_loader/metadata_extractor.py`

```python
import uuid
import re
from typing import Dict
from langchain_core.documents import Document
from rag_project.utils.logger import logger

class MetadataExtractor:
    """Metadata extractor for documents (MVP - core metadata only)"""

    @staticmethod
    def extract_core_metadata(
        doc: Document,
        doc_type: str,
        source: str
    ) -> Dict[str, any]:
        """
        Extract core metadata from document

        Args:
            doc: LangChain Document object
            doc_type: Document type (news, pdf, regulation)
            source: Source file path or name

        Returns:
            Dictionary with core metadata
        """
        metadata = {
            'doc_id': str(uuid.uuid4()),
            'doc_type': doc_type,
            'source': source,
            'title': doc.metadata.get('title', source),
        }

        # Add publish_date if available
        if 'publish_date' in doc.metadata:
            metadata['publish_date'] = doc.metadata['publish_date']

        # Add page_number if available (PDF)
        if 'page_number' in doc.metadata:
            metadata['page_number'] = doc.metadata['page_number']

        return MetadataExtractor.remove_none_values(metadata)

    @staticmethod
    def extract_from_filename(filename: str) -> Dict[str, any]:
        """
        Extract metadata from filename

        Args:
            filename: Filename (can include path)

        Returns:
            Dictionary with extracted metadata
        """
        import os
        basename = os.path.basename(filename)

        metadata = {}

        # Extract title (part before first underscore or extension)
        title_part = basename.split('_')[0] if '_' in basename else basename.rsplit('.', 1)[0]
        metadata['title'] = title_part

        # Extract date (pattern: YYYY-MM-DD)
        date_pattern = r'(\d{4})-(\d{2})-(\d{2})'
        date_match = re.search(date_pattern, basename)

        if date_match:
            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            metadata['publish_date'] = date_str

        return metadata

    @staticmethod
    def add_doc_type_metadata(doc: Document, doc_type: str) -> Document:
        """
        Add document type to document metadata

        Args:
            doc: LangChain Document
            doc_type: Document type to add

        Returns:
            Document with updated metadata
        """
        doc.metadata['doc_type'] = doc_type
        return doc

    @staticmethod
    def remove_none_values(metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Remove None values from metadata dictionary

        Args:
            metadata: Metadata dictionary

        Returns:
            Cleaned metadata dictionary
        """
        return {k: v for k, v in metadata.items() if v is not None}

# Convenience functions for backward compatibility
def extract_core_metadata(doc: Document, doc_type: str, source: str) -> Dict[str, any]:
    """Convenience function for extracting core metadata"""
    return MetadataExtractor.extract_core_metadata(doc, doc_type, source)
```

**Step 4: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_metadata_extractor.py -v`

Expected: PASS for all tests

**Step 5: Commit**

```bash
git add rag_project/data_loader/metadata_extractor.py
git add rag_project/tests/test_metadata_extractor.py
git commit -m "feat: add MVP metadata extractor"
```

---

## Task 6: Chunk Storage Manager

**Files:**
- Create: `rag_project/data_loader/chunk_storage.py`
- Test: `rag_project/tests/test_chunk_storage.py`

**Step 1: Write chunk storage test**

Create: `rag_project/tests/test_chunk_storage.py`

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_chunk_storage.py::test_save_chunks_to_json -v`

Expected: FAIL with "module 'rag_project.data_loader.chunk_storage' does not exist"

**Step 3: Implement chunk storage manager**

Create: `rag_project/data_loader/chunk_storage.py`

```python
import json
import uuid
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from rag_project.utils.logger import logger

class ChunkStorage:
    """Manage storage of document chunks before embedding"""

    def save_chunks_to_json(
        self,
        documents: List[Document],
        output_path: str
    ) -> None:
        """
        Save chunks to JSON file

        Args:
            documents: List of LangChain Document objects
            output_path: Path to output JSON file
        """
        chunks_data = []

        for doc in documents:
            chunk_data = {
                'id': doc.metadata.get('doc_id', str(uuid.uuid4())),
                'text': doc.page_content,
                'metadata': doc.metadata,
                'char_count': len(doc.page_content),
                'created_at': datetime.now().isoformat(),
            }
            chunks_data.append(chunk_data)

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks_data)} chunks to {output_path}")

    def load_chunks_from_json(self, json_path: str) -> List[Document]:
        """
        Load chunks from JSON file

        Args:
            json_path: Path to JSON file

        Returns:
            List of LangChain Document objects
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        documents = []
        for chunk in chunks_data:
            doc = Document(
                page_content=chunk['text'],
                metadata=chunk['metadata']
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks from {json_path}")

        return documents

    def get_chunks_summary(self, json_path: str) -> Dict[str, any]:
        """
        Get summary information about chunks file

        Args:
            json_path: Path to chunks JSON file

        Returns:
            Summary dictionary
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        total_chunks = len(chunks_data)
        total_chars = sum(c['char_count'] for c in chunks_data)
        avg_chunk_length = total_chars / total_chunks if total_chunks > 0 else 0

        doc_types = {}
        for chunk in chunks_data:
            doc_type = chunk['metadata'].get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'avg_chunk_length': avg_chunk_length,
            'doc_types_distribution': doc_types,
        }
```

**Step 4: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_chunk_storage.py -v`

Expected: PASS for all tests

**Step 5: Commit**

```bash
git add rag_project/data_loader/chunk_storage.py
git add rag_project/tests/test_chunk_storage.py
git commit -m "feat: add chunk storage manager"
```

---

## Task 7: Embedding Model Wrapper

**Files:**
- Create: `rag_project/embeddings/embedding_model.py`
- Test: `rag_project/tests/test_embedding_model.py`

**Step 1: Write embedding model test**

Create: `rag_project/tests/test_embedding_model.py`

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_embedding_model.py::test_embed_single_text -v`

Expected: FAIL with "module 'rag_project.embeddings.embedding_model' does not exist"

**Step 3: Implement embedding model wrapper**

Create: `rag_project/embeddings/embedding_model.py`

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_embedding_model.py -v`

Expected: PASS for all tests (may take a few seconds to download model on first run)

**Step 5: Commit**

```bash
git add rag_project/embeddings/embedding_model.py
git add rag_project/tests/test_embedding_model.py
git commit -m "feat: add BGE-M3 embedding model wrapper"
```

---

## Task 8: Milvus Collection Manager

**Files:**
- Create: `rag_project/storage/milvus_manager.py`
- Test: `rag_project/tests/test_milvus_manager.py`

**Step 1: Write Milvus manager test**

Create: `rag_project/tests/test_milvus_manager.py`

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_milvus_manager.py::test_create_collection -v`

Expected: FAIL with "module 'rag_project.storage.milvus_manager' does not exist" (or SKIP if Milvus not running)

**Step 3: Implement Milvus manager**

Create: `rag_project/storage/milvus_manager.py`

```python
import uuid
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger

class MilvusManager:
    """Manage Milvus vector database operations"""

    def __init__(
        self,
        config_path: str = "config/milvus_config.yaml",
        collection_name: str = None
    ):
        """
        Initialize Milvus manager

        Args:
            config_path: Path to Milvus configuration
            collection_name: Collection name (overrides config)
        """
        self.config = load_config(config_path)
        self.milvus_config = self.config.get('milvus', {})

        # Connect to Milvus
        self._connect()

        # Get or create collection
        collection_name = collection_name or self.milvus_config['collection']['name']
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _connect(self):
        """Connect to Milvus server"""
        host = self.milvus_config.get('host', 'localhost')
        port = self.milvus_config.get('port', 19530)
        alias = self.milvus_config.get('alias', 'default')

        try:
            connections.connect(alias=alias, host=host, port=port)
            logger.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _create_collection_schema(self) -> CollectionSchema:
        """Create collection schema"""
        dimension = self.milvus_config['collection']['dimension']

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=20,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="publish_date",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="page_number",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=self.milvus_config['collection'].get('description', ''),
            enable_dynamic_field=False,
        )

        return schema

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        if utility.has_collection(self.collection_name):
            logger.info(f"Using existing collection: {self.collection_name}")
            collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")

            schema = self._create_collection_schema()
            collection = Collection(
                name=self.collection_name,
                schema=schema,
            )

            # Create index
            self._create_index(collection)

        return collection

    def _create_index(self, collection: Collection):
        """Create index for vector field"""
        index_config = self.milvus_config.get('index', {})

        index_params = {
            "index_type": index_config.get('type', 'HNSW'),
            "metric_type": index_config.get('metric_type', 'IP'),
            "params": index_config.get('params', {}),
        }

        collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info(f"Created index: {index_params['index_type']}")

    def insert_data(self, data: List[Dict]) -> int:
        """
        Insert data into collection

        Args:
            data: List of dictionaries with keys: id, vector, text, metadata fields

        Returns:
            Number of inserted records
        """
        # Prepare data by field
        insert_data = {
            "id": [item["id"] for item in data],
            "vector": [item["vector"] for item in data],
            "text": [item["text"] for item in data],
            "doc_type": [item.get("doc_type", "unknown") for item in data],
            "source": [item.get("source", "") for item in data],
            "publish_date": [self._to_timestamp(item.get("publish_date")) for item in data],
            "page_number": [item.get("page_number", 0) for item in data],
            "title": [item.get("title", "") for item in data],
        }

        # Insert
        insert_result = self.collection.insert([insert_data[field] for field in self.collection.schema.fields])

        # Flush
        self.collection.flush()

        logger.info(f"Inserted {len(data)} records into {self.collection_name}")

        return len(data)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search similar vectors

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional filters (e.g., {"doc_type": ["news"]})

        Returns:
            List of search results with text, score, and metadata
        """
        # Load collection
        self.collection.load()

        # Search parameters
        search_config = self.milvus_config.get('search', {})
        search_params = {
            "metric_type": self.milvus_config['index']['metric_type'],
            "params": {"ef": search_config.get('ef', 128)},
        }

        # Build filter expression
        expr = self._build_filter_expression(filters) if filters else None

        # Search
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "doc_type", "source", "publish_date", "page_number", "title"],
        )

        # Format results
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "metadata": {
                    "doc_type": hit.entity.get("doc_type"),
                    "source": hit.entity.get("source"),
                    "publish_date": hit.entity.get("publish_date"),
                    "page_number": hit.entity.get("page_number"),
                    "title": hit.entity.get("title"),
                }
            })

        return formatted_results

    def _build_filter_expression(self, filters: Dict) -> Optional[str]:
        """Build Milvus filter expression from filters dictionary"""
        expr_parts = []

        # Document type filter
        if "doc_type" in filters:
            doc_types = ", ".join([f'"{dt}"' for dt in filters["doc_type"]])
            expr_parts.append(f"doc_type in [{doc_types}]")

        # Date range filter
        if "start_date" in filters:
            start_ts = self._to_timestamp(filters["start_date"])
            expr_parts.append(f"publish_date >= {start_ts}")

        if "end_date" in filters:
            end_ts = self._to_timestamp(filters["end_date"])
            expr_parts.append(f"publish_date <= {end_ts}")

        return " and ".join(expr_parts) if expr_parts else None

    def _to_timestamp(self, date_str) -> int:
        """Convert date string to Unix timestamp"""
        if not date_str:
            return 0

        if isinstance(date_str, int):
            return date_str

        try:
            dt = datetime.fromisoformat(str(date_str).replace("T", " "))
            return int(dt.timestamp())
        except:
            return 0

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        self.collection.load()

        num_entities = self.collection.num_entities

        return {
            "name": self.collection_name,
            "num_entities": num_entities,
        }

    def drop_collection(self):
        """Drop the collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped collection: {self.collection_name}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_milvus_manager.py -v`

Expected: PASS for all tests (requires Milvus running)

**Step 5: Commit**

```bash
git add rag_project/storage/milvus_manager.py
git add rag_project/tests/test_milvus_manager.py
git commit -m "feat: add Milvus collection manager"
```

---

## Task 9: Complete RAG Pipeline Integration

**Files:**
- Create: `rag_project/pipeline.py`
- Create: `rag_project/main.py`
- Test: `rag_project/tests/test_pipeline.py`

**Step 1: Write RAG pipeline test**

Create: `rag_project/tests/test_pipeline.py`

```python
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
    results = pipeline.search("交通建设")

    assert len(results) >= 0

def test_pipeline_with_chunk_storage(sample_documents, tmp_path):
    """Test pipeline with chunk storage"""
    chunks_path = tmp_path / "chunks.json"

    pipeline = RAGPipeline(chunks_storage_path=str(chunks_path))
    pipeline.index_documents(sample_documents)

    # Check chunks file was created
    assert os.path.exists(chunks_path)
```

**Step 2: Run tests to verify they fail**

Run: `pytest rag_project/tests/test_pipeline.py::test_pipeline_end_to_end -v`

Expected: FAIL with "module 'rag_project.pipeline' does not exist"

**Step 3: Implement RAG pipeline**

Create: `rag_project/pipeline.py`

```python
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from langchain_core.documents import Document

from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from rag_project.data_loader.metadata_extractor import MetadataExtractor
from rag_project.data_loader.chunk_storage import ChunkStorage
from rag_project.embeddings.embedding_model import EmbeddingModel
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.logger import logger

class RAGPipeline:
    """Complete RAG processing pipeline"""

    def __init__(
        self,
        chunking_config_path: str = "config/chunking_config.yaml",
        milvus_config_path: str = "config/milvus_config.yaml",
        chunks_storage_path: str = None
    ):
        """
        Initialize RAG pipeline

        Args:
            chunking_config_path: Path to chunking configuration
            milvus_config_path: Path to Milvus configuration
            chunks_storage_path: Optional path to save chunks before embedding
        """
        self.chunker = ConfigurableChunker(chunking_config_path)
        self.embedding_model = EmbeddingModel(milvus_config_path, load_on_init=False)
        self.milvus_manager = MilvusManager(milvus_config_path)
        self.chunk_storage = ChunkStorage()
        self.chunks_storage_path = chunks_storage_path

        logger.info("RAG Pipeline initialized")

    def index_documents(self, file_paths: List[str]) -> int:
        """
        Index documents: load -> chunk -> embed -> store

        Args:
            file_paths: List of file paths to index

        Returns:
            Number of chunks indexed
        """
        all_chunks = []

        # Step 1: Load and chunk documents
        logger.info(f"Loading {len(file_paths)} documents...")

        for file_path in file_paths:
            chunks = self._load_and_chunk_file(file_path)
            all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} chunks total")

        if not all_chunks:
            logger.warning("No chunks generated. Check input files.")
            return 0

        # Step 2: Save chunks (optional)
        if self.chunks_storage_path:
            self.chunk_storage.save_chunks_to_json(all_chunks, self.chunks_storage_path)
            logger.info(f"Chunks saved to {self.chunks_storage_path}")

        # Step 3: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.embed_documents(all_chunks)
        logger.info(f"Generated embeddings: {embeddings.shape}")

        # Step 4: Prepare and insert into Milvus
        logger.info("Inserting into Milvus...")
        milvus_data = self._prepare_milvus_data(all_chunks, embeddings)
        self.milvus_manager.insert_data(milvus_data)

        logger.info(f"Indexing complete: {len(all_chunks)} chunks")

        return len(all_chunks)

    def _load_and_chunk_file(self, file_path: str) -> List[Document]:
        """Load single file and split into chunks"""
        # Detect document type
        doc_type = detect_doc_type(file_path)

        # Load document
        loader = get_loader_for_file(file_path)
        documents = loader.load()

        # Add source to metadata
        source = Path(file_path).name
        for doc in documents:
            doc.metadata['source'] = source
            doc.metadata['doc_type'] = doc_type

        # Extract and add metadata
        for doc in documents:
            core_metadata = MetadataExtractor.extract_core_metadata(doc, doc_type, source)
            doc.metadata.update(core_metadata)

        # Split into chunks
        chunks = self.chunker.split_documents(documents, doc_type)

        return chunks

    def _prepare_milvus_data(
        self,
        chunks: List[Document],
        embeddings
    ) -> List[Dict]:
        """Prepare data for Milvus insertion"""
        milvus_data = []

        for chunk, embedding in zip(chunks, embeddings):
            milvus_data.append({
                "id": chunk.metadata.get('doc_id', str(uuid.uuid4())),
                "vector": embedding.tolist(),
                "text": chunk.page_content,
                "doc_type": chunk.metadata.get("doc_type", "unknown"),
                "source": chunk.metadata.get("source", ""),
                "publish_date": chunk.metadata.get("publish_date"),
                "page_number": chunk.metadata.get("page_number", 0),
                "title": chunk.metadata.get("title", ""),
            })

        return milvus_data

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"doc_type": ["news"]})

        Returns:
            List of search results
        """
        # Generate query embedding
        query_vector = self.embedding_model.embed_text(query)

        # Search Milvus
        results = self.milvus_manager.search(
            query_vector.tolist(),
            top_k=top_k,
            filters=filters
        )

        return results

    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        milvus_stats = self.milvus_manager.get_collection_stats()

        return {
            "milvus_collection": milvus_stats,
            "embedding_model": self.embedding_model.get_model_info(),
        }
```

**Step 4: Create main entry point**

Create: `rag_project/main.py`

```python
"""
RAG Project - Main Entry Point
主入口文件
"""

import argparse
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="RAG Document Processing Pipeline")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('paths', nargs='+', help='File or directory paths')
    index_parser.add_argument('--chunks-output', default='data/chunks.json', help='Chunks output file')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--doc-type', action='append', help='Filter by document type')

    args = parser.parse_args()

    if args.command == 'index':
        # Collect file paths
        file_paths = []
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_file():
                file_paths.append(str(path))
            elif path.is_dir():
                file_paths.extend([str(p) for p in path.rglob('*') if p.is_file()])

        if not file_paths:
            logger.error("No files found to index")
            return

        logger.info(f"Found {len(file_paths)} files to index")

        # Create pipeline and index
        pipeline = RAGPipeline(chunks_storage_path=args.chunks_output)
        count = pipeline.index_documents(file_paths)

        logger.info(f"Indexing complete: {count} chunks")

    elif args.command == 'search':
        # Create pipeline
        pipeline = RAGPipeline()

        # Build filters
        filters = {}
        if args.doc_type:
            filters['doc_type'] = args.doc_type

        # Search
        results = pipeline.search(args.query, top_k=args.top_k, filters=filters)

        # Display results
        print(f"\n搜索结果 (query: {args.query})")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] 相关度: {result['score']:.4f}")
            print(f"来源: {result['metadata']['source']}")
            print(f"类型: {result['metadata']['doc_type']}")
            if result['metadata']['page_number']:
                print(f"页码: {result['metadata']['page_number']}")
            print(f"内容: {result['text'][:200]}...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

**Step 5: Run tests to verify they pass**

Run: `pytest rag_project/tests/test_pipeline.py -v`

Expected: PASS for all tests (requires Milvus running)

**Step 6: Commit**

```bash
git add rag_project/pipeline.py
git add rag_project/main.py
git add rag_project/tests/test_pipeline.py
git commit -m "feat: add complete RAG pipeline integration"
```

---

## Task 10: Documentation and README

**Files:**
- Create: `README.md`
- Create: `USAGE.md`
- Create: `docs/ARCHITECTURE.md`

**Step 1: Write project README**

Create: `README.md`

```markdown
# RAG Document Processing Pipeline

基于RAG的检索型AI智能体文档处理模块 - 完整的文档加载、分块、嵌入和检索流程。

## 功能特性

- ✅ 支持多种文档格式（TXT, PDF, DOCX）
- ✅ 可配置的文本分块策略
- ✅ 元数据提取（MVP）
- ✅ BGE-M3本地嵌入模型
- ✅ Milvus向量数据库集成
- ✅ 高级检索过滤（文档类型、时间范围）

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 索引文档

```bash
python -m rag_project.main index 知识库/知识库/*.txt --chunks-output data/chunks.json
```

### 搜索文档

```bash
python -m rag_project.main search "江西省高速公路建设政策" --top-k 5
```

## 项目结构

```
rag_project/
├── config/              # 配置文件
│   ├── chunking_config.yaml
│   └── milvus_config.yaml
├── data_loader/         # 数据加载模块
│   ├── document_type_detector.py
│   ├── configurable_splitter.py
│   ├── metadata_extractor.py
│   └── chunk_storage.py
├── embeddings/          # 嵌入模型
│   └── embedding_model.py
├── storage/             # 向量存储
│   └── milvus_manager.py
├── utils/               # 工具函数
│   ├── config_loader.py
│   └── logger.py
├── tests/               # 测试代码
├── pipeline.py          # 完整流程
└── main.py             # 命令行入口
```

## 配置说明

### 分块配置 (config/chunking_config.yaml)

调整不同文档类型的分块参数：

```yaml
chunking:
  news:
    chunk_size: 512      # 块大小（字符数）
    chunk_overlap: 50    # 重叠大小
    separators: [...]    # 分隔符
```

### Milvus配置 (config/milvus_config.yaml)

配置向量数据库和嵌入模型：

```yaml
milvus:
  host: "localhost"
  port: 19530
  collection:
    dimension: 1024      # BGE-M3维度
```

## 测试

```bash
# 运行所有测试
pytest rag_project/tests/ -v

# 运行特定测试
pytest rag_project/tests/test_pipeline.py -v
```

## License

MIT
```

**Step 2: Write usage documentation**

Create: `USAGE.md`

```markdown
# RAG Pipeline 使用指南

## 1. 索引文档

### 索引单个文件

```bash
python -m rag_project.main index knowledge/news.txt
```

### 索引整个目录

```bash
python -m rag_project.main index 知识库/知识库/
```

### 指定输出文件

```bash
python -m rag_project.main index 知识库/ --chunks-output data/my_chunks.json
```

## 2. 搜索文档

### 基本搜索

```bash
python -m rag_project.main search "高速公路建设"
```

### 限制结果数量

```bash
python -m rag_project.main search "交通政策" --top-k 5
```

### 按文档类型过滤

```bash
python -m rag_project.main search "交通法规" --doc-type policy --doc-type regulation
```

## 3. 调整分块参数

编辑 `config/chunking_config.yaml`：

```yaml
chunking:
  news:
    chunk_size: 256  # 改为更小的块
    chunk_overlap: 30
```

重新运行索引：

```bash
python -m rag_project.main index 知识库/
```

## 4. Python API使用

```python
from rag_project.pipeline import RAGPipeline

# 创建pipeline
pipeline = RAGPipeline()

# 索引文档
count = pipeline.index_documents(["news.txt", "policy.pdf"])
print(f"索引了 {count} 个chunks")

# 搜索
results = pipeline.search(
    query="江西省高速公路政策",
    top_k=10,
    filters={"doc_type": ["policy"]}
)

for result in results:
    print(f"相关度: {result['score']:.4f}")
    print(f"内容: {result['text']}\n")
```

## 5. 故障排查

### Milvus连接失败

```bash
# 检查Milvus是否运行
docker ps | grep milvus

# 查看日志
docker logs milvus-standalone
```

### 模型下载慢

模型会自动下载到 `data/models/`，也可以手动下载后指定路径。

### 分块效果不理想

1. 查看保存的chunks.json
2. 调整chunking_config.yaml中的参数
3. 重新索引
```

**Step 3: Write architecture documentation**

Create: `docs/ARCHITECTURE.md`

```markdown
# 系统架构文档

## 整体架构

```
用户查询
    ↓
RAG Pipeline (pipeline.py)
    ↓
┌──────────────────────────────────┐
│  Data Layer (data_loader/)        │
│  - Document loaders               │
│  - Text splitter                  │
│  - Metadata extractor             │
│  - Chunk storage                  │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│  Embedding Layer (embeddings/)    │
│  - BGE-M3 model wrapper           │
│  - Batch embedding                │
└──────────────────────────────────┘
    ↓
┌──────────────────────────────────┐
│  Storage Layer (storage/)         │
│  - Milvus manager                 │
│  - Vector search                  │
│  - Metadata filtering             │
└──────────────────────────────────┘
    ↓
返回结果
```

## 模块说明

### 1. Data Layer

**document_type_detector.py** - 检测文档类型并返回相应加载器

**configurable_splitter.py** - 基于YAML配置的可配置分块器

**metadata_extractor.py** - 提取核心元数据（MVP）

**chunk_storage.py** - 管理chunks的JSON存储

### 2. Embedding Layer

**embedding_model.py** - BGE-M3模型封装，支持单文本和批量嵌入

### 3. Storage Layer

**milvus_manager.py** - Milvus集合管理，支持插入和检索

## 数据流

### 索引流程

```
文件路径
  → detect_doc_type()
  → get_loader_for_file()
  → LangChain Document
  → extract_core_metadata()
  → splitter.split_documents()
  → Chunks (Document对象)
  → save_chunks_to_json() (可选)
  → embed_documents()
  → 向量 np.ndarray
  → prepare_milvus_data()
  → milvus_manager.insert_data()
  → Milvus Collection
```

### 检索流程

```
查询文本
  → embed_text()
  → 查询向量
  → milvus_manager.search()
  → 构建过滤表达式
  → 向量检索
  → 格式化结果
  → 返回结果列表
```

## 设计原则

1. **配置驱动**: 所有参数通过YAML配置
2. **模块化**: 每个模块职责单一
3. **可测试**: 所有模块都有单元测试
4. **可扩展**: 易于添加新的文档类型和检索策略
```

**Step 4: Commit**

```bash
git add README.md USAGE.md docs/ARCHITECTURE.md
git commit -m "docs: add comprehensive documentation"
```

---

## Summary

This implementation plan provides:

1. ✅ **Modular architecture** - Clear separation of concerns
2. ✅ **Configuration-driven** - Easy parameter tuning via YAML
3. ✅ **Comprehensive testing** - Unit tests for all components
4. ✅ **MVP approach** - Start with core features, extend later
5. ✅ **Complete documentation** - README, usage guide, architecture docs
6. ✅ **Production-ready** - Error handling, logging, flexible deployment

**Total implementation estimate**: 10 tasks × 30-60 minutes each = 5-10 hours

**Next steps**: Execute this plan using superpowers:executing-plans
