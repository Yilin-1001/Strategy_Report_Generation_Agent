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
