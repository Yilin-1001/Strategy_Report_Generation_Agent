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
        # Skip DOCX and DOC files for now
        # '.docx': 'regulation',
        # '.doc': 'regulation',
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
            mode="single",  # Single text mode
            strategy="fast",  # Fast strategy (no OCR)
            extract_images_in_pdf=False  # Skip images
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
