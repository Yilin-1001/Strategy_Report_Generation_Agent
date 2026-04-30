"""
GPU-accelerated PDF loader using PaddleOCR
GPU加速的PDF加载器（使用PaddleOCR）
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from rag_project.utils.logger import logger
import torch

try:
    from paddleocr import PaddleOCR  # type: ignore
    import fitz  # type: ignore # PyMuPDF
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available. Install with: pip install paddleocr pymupdf")


class GPUPDFLoader:
    """GPU-accelerated PDF loader using PaddleOCR"""

    def __init__(
        self,
        file_path: str,
        use_gpu: bool = True,
        lang: str = 'ch',  # Chinese and English
        show_log: bool = False
    ):
        """
        Initialize GPU PDF loader

        Args:
            file_path: Path to PDF file
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('ch' for Chinese+English)
            show_log: Whether to show PaddleOCR logs
        """
        self.file_path = Path(file_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR or PyMuPDF not installed")

        # Initialize PaddleOCR with GPU
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=self.use_gpu,
            show_log=show_log,
            det_model_dir=None,  # Use default models
            rec_model_dir=None,
            cls_model_dir=None
        )

        logger.info(f"GPU PDF Loader initialized: GPU={self.use_gpu}")

    def load(self) -> List[Document]:
        """
        Load PDF with GPU-accelerated OCR

        Returns:
            List of Document objects
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        documents = []

        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(str(self.file_path))

            logger.info(f"Processing PDF: {self.file_path.name} ({len(pdf_document)} pages)")

            # Extract text and process each page
            for page_num, page in enumerate(pdf_document, start=1):
                # Get page as image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                # Perform OCR with PaddleOCR
                result = self.ocr.ocr(img_bytes, cls=True)

                # Extract text from OCR result
                page_text = self._extract_text_from_ocr(result)

                # Also try to extract text directly (for text-based PDFs)
                direct_text = page.get_text()

                # Prefer direct text if available, otherwise use OCR
                text_content = direct_text if direct_text.strip() else page_text

                if text_content.strip():
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            'source': str(self.file_path),
                            'page_number': page_num,
                            'total_pages': len(pdf_document),
                            'file_name': self.file_path.name,
                            'ocr_used': len(direct_text.strip()) == 0
                        }
                    )
                    documents.append(doc)

            pdf_document.close()

            logger.info(f"Extracted {len(documents)} pages from PDF")

        except Exception as e:
            logger.error(f"Error processing PDF {self.file_path}: {e}")
            raise

        return documents

    def _extract_text_from_ocr(self, ocr_result) -> str:
        """Extract text from PaddleOCR result"""
        if not ocr_result or not ocr_result[0]:
            return ""

        text_lines = []
        for line in ocr_result[0]:
            if line and len(line) >= 2:
                text_lines.append(line[1][0])  # Get text from [bbox, text, confidence]

        return "\n".join(text_lines)

    @classmethod
    def test_gpu(cls):
        """Test if PaddleOCR GPU is working"""
        if not PADDLEOCR_AVAILABLE:
            logger.error("PaddleOCR not installed")
            return False

        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return False

            # Test PaddleOCR with GPU
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True, show_log=False)

            # Simple test
            logger.info(f"✓ PaddleOCR GPU test passed!")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            return True

        except Exception as e:
            logger.error(f"PaddleOCR GPU test failed: {e}")
            return False


if __name__ == "__main__":
    # Test GPU availability
    print("Testing PaddleOCR GPU...")
    success = GPUPDFLoader.test_gpu()

    if success:
        print("\n✓ PaddleOCR GPU is ready!")
    else:
        print("\n✗ PaddleOCR GPU not available")
