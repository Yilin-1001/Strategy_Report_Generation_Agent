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
