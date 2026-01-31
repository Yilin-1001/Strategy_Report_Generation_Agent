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
