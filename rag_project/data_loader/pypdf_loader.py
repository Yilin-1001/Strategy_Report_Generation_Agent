"""
使用 PyPDF 解析PDF文件的加载器
PyPDF (原 PyPDF2) 是一个纯Python的PDF解析库
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

from rag_project.utils.logger import logger


@dataclass
class PDFPageInfo:
    """PDF页面信息"""
    page_number: int
    text: str
    char_count: int
    images_count: int
    has_tables: bool = False


@dataclass
class PDFDocumentInfo:
    """PDF文档信息"""
    file_path: str
    total_pages: int
    title: Optional[str] = None
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None


class PyPDFLoader:
    """
    PyPDF PDF加载器

    特点：
    - 纯Python实现，无需额外依赖
    - 支持提取文本、元数据
    - 支持按页分割
    - 支持图像检测
    - 轻量级，适合简单PDF
    """

    def __init__(self, pdf_path: str, extract_images: bool = False):
        """
        初始化PyPDF加载器

        Args:
            pdf_path: PDF文件路径
            extract_images: 是否提取图像信息
        """
        if not PYPDF_AVAILABLE:
            raise ImportError(
                "PyPDF未安装。请运行: pip install pypdf\n"
                "或者: pip install PyPDF2"
            )

        self.pdf_path = Path(pdf_path)
        self.extract_images = extract_images

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        logger.info(f"初始化PyPDF加载器: {self.pdf_path.name}")

    def load_document(self) -> PdfReader:
        """
        加载PDF文档

        Returns:
            PdfReader对象
        """
        try:
            reader = PdfReader(str(self.pdf_path))
            logger.info(f"成功加载PDF: {self.pdf_path.name} ({len(reader.pages)} 页)")
            return reader
        except Exception as e:
            logger.error(f"加载PDF失败: {e}")
            raise

    def extract_metadata(self, reader: PdfReader) -> PDFDocumentInfo:
        """
        提取PDF元数据

        Args:
            reader: PdfReader对象

        Returns:
            PDF文档信息
        """
        metadata = reader.metadata

        doc_info = PDFDocumentInfo(
            file_path=str(self.pdf_path),
            total_pages=len(reader.pages),
            title=metadata.get('/Title', '') if metadata else None,
            author=metadata.get('/Author', '') if metadata else None,
            creator=metadata.get('/Creator', '') if metadata else None,
            producer=metadata.get('/Producer', '') if metadata else None,
            creation_date=metadata.get('/CreationDate', '') if metadata else None,
            modification_date=metadata.get('/ModDate', '') if metadata else None,
            subject=metadata.get('/Subject', '') if metadata else None,
            keywords=metadata.get('/Keywords', '') if metadata else None
        )

        logger.info(f"PDF元数据: 标题='{doc_info.title}', 作者='{doc_info.author}'")

        return doc_info

    def extract_page_text(self, reader: PdfReader, page_num: int) -> str:
        """
        提取单页文本

        Args:
            reader: PdfReader对象
            page_num: 页码（从0开始）

        Returns:
            页面文本
        """
        try:
            page = reader.pages[page_num]
            text = page.extract_text()

            if not text:
                logger.warning(f"第 {page_num + 1} 页: 未提取到文本（可能是图像页）")
                return ""

            return text
        except Exception as e:
            logger.error(f"第 {page_num + 1} 页: 提取失败 - {e}")
            return ""

    def extract_all_pages(self, reader: PdfReader) -> List[PDFPageInfo]:
        """
        提取所有页面

        Args:
            reader: PdfReader对象

        Returns:
            页面信息列表
        """
        pages_info = []

        for i, page in enumerate(reader.pages):
            try:
                # 提取文本
                text = page.extract_text() or ""

                # 统计信息
                char_count = len(text)

                # 检测图像
                images_count = 0
                if self.extract_images:
                    try:
                        images_count = len(page.images)
                    except:
                        images_count = 0

                # 检测表格（简单启发式）
                has_tables = self._detect_tables(text)

                page_info = PDFPageInfo(
                    page_number=i + 1,
                    text=text,
                    char_count=char_count,
                    images_count=images_count,
                    has_tables=has_tables
                )

                pages_info.append(page_info)

                logger.debug(
                    f"第 {i + 1} 页: {char_count} 字符, "
                    f"{images_count} 图像, {'包含表格' if has_tables else '无表格'}"
                )

            except Exception as e:
                logger.error(f"第 {i + 1} 页处理失败: {e}")

        return pages_info

    def _detect_tables(self, text: str) -> bool:
        """
        检测文本是否可能包含表格

        Args:
            text: 页面文本

        Returns:
            是否包含表格
        """
        # 简单的启发式规则
        # 1. 多个连续的空格或制表符
        tab_count = text.count('\t')
        if tab_count > 10:
            return True

        # 2. 多行竖线分隔符（|）
        line_count = 0
        for line in text.split('\n'):
            if line.count('|') >= 3:
                line_count += 1
        if line_count >= 3:
            return True

        # 3. 多个数字对齐（可能是数据行）
        lines = text.split('\n')
        aligned_lines = 0
        for line in lines:
            # 检查是否有多个数字用空格分隔
            numbers = re.findall(r'\d+\.\d+|\d+', line)
            if len(numbers) >= 4:
                aligned_lines += 1
        if aligned_lines >= 3:
            return True

        return False

    def extract_with_page_markers(
        self,
        reader: PdfReader
    ) -> str:
        """
        提取文本并添加分页标记（与convert_pdf_to_txt.py格式兼容）

        Args:
            reader: PdfReader对象

        Returns:
            带分页标记的完整文本
        """
        total_pages = len(reader.pages)
        text_parts = []

        for page_num, page in enumerate(reader.pages):
            # 添加分页标记
            page_marker = (
                f"\n{'='*80}\n"
                f"第 {page_num + 1} 页 / 共 {total_pages} 页\n"
                f"{'='*80}\n\n"
            )

            # 提取页面文本
            page_text = page.extract_text() or ""
            text_parts.append(page_marker + page_text)

        full_text = '\n'.join(text_parts)

        logger.info(f"提取文本: {len(full_text)} 字符, {total_pages} 页")

        return full_text

    def extract_sections(self, reader: PdfReader) -> List[Tuple[int, int, str]]:
        """
        提取章节标题

        Args:
            reader: PdfReader对象

        Returns:
            [(页码, 位置, 标题), ...]
        """
        sections = []

        # 章节标题模式
        section_patterns = [
            r'^第[一二三四五六七八九十百千零\d]+章[：::\s]*(.*?)$',
            r'^Chapter\s+[IVXLCDM\d]+[：::\s]*(.*?)$',
            r'^(\d+\.\d+)\s{1,5}(.{2,50})$',
            r'^(摘要|前言|序言|目录|参考文献|附录)(?:[：:]|\s|$)',
            r'^【(.{2,30})】$',
        ]

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lines = text.split('\n')

            char_pos = 0
            for line in lines:
                line = line.strip()
                if not line:
                    char_pos += len(line) + 1
                    continue

                for pattern in section_patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        # 提取标题
                        if len(match.groups()) >= 2 and match.group(2):
                            title = match.group(2).strip()
                        elif len(match.groups()) >= 1:
                            title = match.group(1).strip()
                        else:
                            title = line.strip()

                        # 过滤条件
                        if 2 <= len(title) <= 80:
                            sections.append((page_num + 1, char_pos, title))
                            break

                char_pos += len(line) + 1

        logger.info(f"提取到 {len(sections)} 个章节标题")
        return sections

    def save_to_txt(
        self,
        reader: PdfReader,
        output_path: Optional[str] = None
    ) -> str:
        """
        保存为TXT文件

        Args:
            reader: PdfReader对象
            output_path: 输出路径（默认为PDF同名.txt文件）

        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = self.pdf_path.with_suffix('.txt')

        output_path = Path(output_path)

        # 提取带分页标记的文本
        full_text = self.extract_with_page_markers(reader)

        # 保存文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        logger.info(f"已保存TXT文件: {output_path}")

        return str(output_path)

    def get_document_stats(self, reader: PdfReader) -> Dict:
        """
        获取文档统计信息

        Args:
            reader: PdfReader对象

        Returns:
            统计信息字典
        """
        total_chars = 0
        empty_pages = 0
        pages_with_images = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            total_chars += len(text)

            if not text:
                empty_pages += 1

            try:
                if len(page.images) > 0:
                    pages_with_images += 1
            except:
                pass

        stats = {
            'total_pages': len(reader.pages),
            'total_characters': total_chars,
            'empty_pages': empty_pages,
            'pages_with_images': pages_with_images,
            'avg_chars_per_page': total_chars / len(reader.pages) if reader.pages else 0
        }

        logger.info(f"文档统计: {stats}")

        return stats


def convert_pdf_with_pypdf(
    pdf_path: str,
    output_path: Optional[str] = None,
    add_page_markers: bool = True
) -> Dict:
    """
    使用PyPDF转换PDF为TXT

    Args:
        pdf_path: PDF文件路径
        output_path: 输出TXT路径
        add_page_markers: 是否添加分页标记

    Returns:
        转换结果字典
    """
    logger.info("="*80)
    logger.info("PyPDF PDF转TXT工具")
    logger.info("="*80)

    start_time = datetime.now()

    try:
        # 创建加载器
        loader = PyPDFLoader(pdf_path)

        # 加载PDF
        reader = loader.load_document()

        # 提取元数据
        metadata = loader.extract_metadata(reader)

        # 提取文本
        if add_page_markers:
            full_text = loader.extract_with_page_markers(reader)
        else:
            pages = loader.extract_all_pages(reader)
            full_text = '\n\n'.join([p.text for p in pages])

        # 保存文件
        if output_path is None:
            output_path = Path(pdf_path).with_suffix('.txt')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        # 获取统计信息
        stats = loader.get_document_stats(reader)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = {
            'status': 'success',
            'pdf_path': str(pdf_path),
            'output_path': str(output_path),
            'metadata': {
                'title': metadata.title,
                'author': metadata.author,
                'total_pages': metadata.total_pages,
            },
            'stats': stats,
            'duration_seconds': duration
        }

        logger.info(f"\n转换成功！")
        logger.info(f"输出文件: {output_path}")
        logger.info(f"处理时间: {duration:.2f} 秒")
        logger.info("="*80)

        return result

    except Exception as e:
        logger.error(f"转换失败: {e}")
        return {
            'status': 'failed',
            'pdf_path': str(pdf_path),
            'error': str(e)
        }


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) < 2:
        print("用法: python pypdf_loader.py <pdf文件路径> [输出txt路径]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    txt_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert_pdf_with_pypdf(pdf_file, txt_file)

    if result['status'] == 'success':
        print(f"\n✓ 转换成功!")
        print(f"  输出: {result['output_path']}")
        print(f"  页数: {result['metadata']['total_pages']}")
        print(f"  字符数: {result['stats']['total_characters']:,}")
    else:
        print(f"\n✗ 转换失败: {result.get('error')}")
        sys.exit(1)
