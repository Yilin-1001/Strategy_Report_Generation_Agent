from typing import List, Dict, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger
from rag_project.data_loader.metadata_builder_v2 import (
    MetadataBuilderV2,
    SectionTitleExtractor
)
import re

class ConfigurableChunker:
    """Configurable document chunker with YAML-based configuration"""

    def __init__(
        self,
        config_path: str = "config/chunking_config.yaml",
        base_dir: str = "知识库/知识库",
        use_v2_metadata: bool = True
    ):
        """
        Initialize configurable chunker

        Args:
            config_path: Path to chunking configuration YAML file
            base_dir: Knowledge base root directory (for metadata extraction)
            use_v2_metadata: Whether to use V2 metadata builder
        """
        self.config_path = config_path
        self.splitters = self._create_splitters()
        self.base_dir = base_dir
        self.use_v2_metadata = use_v2_metadata

        if use_v2_metadata:
            self.metadata_builder: Optional[MetadataBuilderV2] = MetadataBuilderV2(base_dir)
        else:
            self.metadata_builder = None

        logger.info(f"ConfigurableChunker initialized with {len(self.splitters)} splitters")
        if use_v2_metadata:
            logger.info("Using V2 metadata builder (enhanced metadata structure)")

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

        all_chunks = []

        for doc in documents:
            # 检查是否使用V2 metadata
            if self.use_v2_metadata and self.metadata_builder:
                # 使用新的metadata构建器
                chunks = self._split_with_v2_metadata(doc, splitter, doc_type)
            else:
                # 使用原有逻辑
                chunks = self._split_with_legacy_metadata(doc, splitter, doc_type)

            all_chunks.extend(chunks)

        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks (type={doc_type})")

        return all_chunks

    def _split_with_v2_metadata(
        self,
        doc: Document,
        splitter,
        doc_type: str
    ) -> List[Document]:
        """使用V2 metadata构建器分割文档"""
        chunks = []

        # Assert to help type checker - this method is only called when metadata_builder is not None
        assert self.metadata_builder is not None

        # 检查文档是否有分页标记
        has_page_markers = self._has_page_markers(doc.page_content)

        if has_page_markers:
            # 按页分割文档
            page_docs = self._split_by_pages(doc)

            # 检测目录页面范围
            toc_page_range = self._find_toc_page_range(doc)
            if toc_page_range[0]:
                logger.info(f"检测到目录页面: 第{toc_page_range[0]}至{toc_page_range[1]}页")

            # 预先提取章节（从完整文档，包括被移除的部分）
            # 这样可以保证章节标题识别正常工作
            sections = SectionTitleExtractor.extract_sections(doc.page_content)

            # 检测前置页面（封面、版权页、编委会等）
            front_matter_range = self._find_front_matter_page_range(doc)

            # 对每页分别进行chunking
            for page_doc in page_docs:
                page_num = page_doc.metadata.get('page_number', 1)

                # 跳过前置页面（封面、版权页等）
                if self._is_front_matter_page(page_doc, front_matter_range):
                    logger.debug(f"跳过前置页面: 第{page_num}页")
                    continue

                # 跳过目录页面
                if self._is_toc_page(page_doc, toc_page_range):
                    logger.debug(f"跳过目录页面: 第{page_num}页")
                    continue

                page_chunks = splitter.split_documents([page_doc])

                # 为每个chunk添加metadata
                for chunk in page_chunks:
                    # 计算chunk在文档中的位置
                    chunk_position = doc.page_content.find(chunk.page_content)

                    # 使用V2 metadata构建器
                    # 优先使用full_path提取tags，如果没有则使用source
                    file_path_for_metadata = doc.metadata.get('full_path', doc.metadata.get('source', ''))

                    metadata = self.metadata_builder.build_metadata(
                        chunk_content=chunk.page_content,
                        chunk_position=chunk_position,
                        file_path=file_path_for_metadata,
                        page_number=page_num,
                        sections=sections
                    )

                    # 更新chunk的metadata
                    chunk.metadata.update(metadata)
                    chunks.append(chunk)
        else:
            # 没有分页标记，直接处理
            doc_chunks = splitter.split_documents([doc])

            # 预先提取章节
            sections = SectionTitleExtractor.extract_sections(doc.page_content)

            for i, chunk in enumerate(doc_chunks, 1):
                chunk_position = doc.page_content.find(chunk.page_content)

                # 使用V2 metadata构建器
                # 优先使用full_path提取tags，如果没有则使用source
                file_path_for_metadata = doc.metadata.get('full_path', doc.metadata.get('source', ''))

                metadata = self.metadata_builder.build_metadata(
                    chunk_content=chunk.page_content,
                    chunk_position=chunk_position,
                    file_path=file_path_for_metadata,
                    page_number=1,
                    sections=sections
                )

                chunk.metadata.update(metadata)
                chunks.append(chunk)

        return chunks

    def _split_with_legacy_metadata(
        self,
        doc: Document,
        splitter,
        doc_type: str
    ) -> List[Document]:
        """使用原有逻辑分割文档"""
        chunks = []

        # 检查文档是否有分页标记
        has_page_markers = self._has_page_markers(doc.page_content)

        if has_page_markers:
            # 按页分割文档
            page_docs = self._split_by_pages(doc)

            for page_doc in page_docs:
                page_num = page_doc.metadata.get('page_number', 1)
                page_chunks = splitter.split_documents([page_doc])

                for chunk in page_chunks:
                    chunk.metadata['doc_type'] = doc_type
                    chunk.metadata['page_number'] = page_num
                    chunks.append(chunk)
        else:
            # 没有分页标记，直接处理
            doc_chunks = splitter.split_documents([doc])
            for chunk in doc_chunks:
                chunk.metadata['doc_type'] = doc_type
                chunk.metadata['page_number'] = chunk.metadata.get('page_number', 1)
            chunks.extend(doc_chunks)

        return chunks

    def _has_page_markers(self, text: str) -> bool:
        """
        检查文本是否包含PyMuPDF的分页标记

        Args:
            text: 文本内容

        Returns:
            是否包含分页标记
        """
        pattern = r'\n={80}\n第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\n={80}\n'
        return bool(re.search(pattern, text))

    def _split_by_pages(self, doc: Document) -> List[Document]:
        """
        按照分页标记分割文档为多个页面

        Args:
            doc: 原始文档

        Returns:
            页面文档列表
        """
        # PyMuPDF分页标记模式
        page_marker_pattern = re.compile(
            r'(\n={80}\n第\s*(\d+)\s*页\s*/\s*共\s*\d+\s*页\n={80}\n)',
            re.MULTILINE
        )

        text = doc.page_content
        pages = []

        # 分割文本
        parts = page_marker_pattern.split(text)

        # parts结构: [text1, marker1, page_num1, marker2, page_num2, ..., textN]
        # 第一个元素是第一页的内容（可能为空）
        current_text = parts[0] if parts[0] else ""

        for i in range(1, len(parts), 3):
            if i + 2 < len(parts):
                marker = parts[i]
                page_num = int(parts[i + 1])
                page_content = parts[i + 2] if (i + 2 < len(parts)) else ""

                # 如果有前一页的内容，先保存
                if current_text.strip():
                    page_doc = Document(
                        page_content=current_text.strip(),
                        metadata=doc.metadata.copy()
                    )
                    # 上一页的页码（当前页码-1，或者如果没有标记则为1）
                    page_doc.metadata['page_number'] = page_num - 1 if page_num > 1 else 1
                    pages.append(page_doc)

                # 保存当前页内容
                if page_content.strip():
                    page_doc = Document(
                        page_content=page_content.strip(),
                        metadata=doc.metadata.copy()
                    )
                    page_doc.metadata['page_number'] = page_num
                    pages.append(page_doc)

                current_text = ""

        # 处理最后一页
        if current_text.strip():
            last_page_num = pages[-1].metadata.get('page_number', 0) + 1 if pages else 1
            page_doc = Document(
                page_content=current_text.strip(),
                metadata=doc.metadata.copy()
            )
            page_doc.metadata['page_number'] = last_page_num
            pages.append(page_doc)

        return pages

    def _find_toc_page_range(self, doc: Document) -> Tuple[Optional[int], Optional[int]]:
        """
        查找目录页面的页码范围

        Args:
            doc: 原始文档（带分页标记）

        Returns:
            (起始页码, 结束页码) 或 (None, None) 如果未找到目录
        """
        # 首先按页分割
        page_docs = self._split_by_pages(doc)

        # 查找"目录"所在的页面
        toc_start_page: Optional[int] = None
        toc_end_page: Optional[int] = None

        for page_doc in page_docs:
            page_num = page_doc.metadata.get('page_number', 1)
            content = page_doc.page_content

            # 检查是否包含"目录"标题
            toc_pattern = re.compile(r'^(目录|CONTENTS|Contents|目　录)', re.MULTILINE)
            if toc_pattern.search(content):
                toc_start_page = page_num

                # 找到目录后，继续查找第一个实际章节
                # 检查后续页面，找到第一���包含"第X章"的页面
                for next_page_doc in page_docs:
                    next_page_num = next_page_doc.metadata.get('page_number', 1)
                    if next_page_num <= page_num:
                        continue

                    next_content = next_page_doc.page_content

                    # 检查是否包含章节标题
                    chapter_pattern = re.compile(r'^第[一二三四五六七八九十百千零\d]+章', re.MULTILINE)
                    if chapter_pattern.search(next_content):
                        toc_end_page = next_page_num - 1
                        break

                    # 另一种检查：如果页面内容很少（通常是目录页）
                    # 而下一页内容很多，说明目录结束了
                    if len(next_content.strip()) > 500:  # 假设实际内容超过500字符
                        toc_end_page = next_page_num - 1
                        break

                break

        # 如果只找到了目录开始，但没有找到结束，假设目录占1-2页
        if toc_start_page and not toc_end_page:
            toc_end_page = toc_start_page + 1

        return (toc_start_page, toc_end_page)

    def _find_front_matter_page_range(self, doc: Document) -> Tuple[Optional[int], Optional[int]]:
        """
        查找文档的前置部分（封面、版权页、编委会等）

        这些页面包含书名、作者、机构信息，通常不是实际内容

        Args:
            doc: 原始文档（带分页标记）

        Returns:
            (起始页码, 结束页码) ��� (None, None) 如果未找到
        """
        # 首先按页分割
        page_docs = self._split_by_pages(doc)

        front_matter_end = None

        # 查找实际内容开始的标志
        content_start_keywords = [
            r'前言',
            r'序言',
            r'摘要',
            r'Abstract',
            r'Preface',
            r'Introduction',
            r'第[一二三四五六七八九十百千零\d]+章',  # 第一章
        ]

        for page_doc in page_docs:
            page_num = page_doc.metadata.get('page_number', 1)
            content = page_doc.page_content

            # 检查是否包含内容开始的标志
            for keyword_pattern in content_start_keywords:
                if re.search(keyword_pattern, content, re.MULTILINE):
                    front_matter_end = page_num - 1
                    logger.info(f"检测到实际内容从第{page_num}页开始（包含关键词: {keyword_pattern}）")
                    return (1, front_matter_end)

        # 如果没有找到明确的开始标志，使用启发式规则
        # 通常封面和版权页在前3-5页
        if len(page_docs) > 6:
            logger.info("未检测到明确的内容开始标志，使用默认规则：跳过前5页")
            return (1, 5)

        return (None, None)

    def _is_front_matter_page(self, page_doc: Document, front_matter_range: Tuple[Optional[int], Optional[int]]) -> bool:
        """
        检查页面是否是前置部分（封面、版权页等）

        Args:
            page_doc: 页面文档
            front_matter_range: 前置部分页码范围 (起始, 结束)

        Returns:
            是否为前置页面
        """
        if not front_matter_range or front_matter_range[0] is None:
            return False

        page_num = page_doc.metadata.get('page_number', 1)
        start, end = front_matter_range

        return start <= page_num <= end

    def _is_toc_page(self, page_doc: Document, toc_page_range: Tuple[Optional[int], Optional[int]]) -> bool:
        """
        检查页面是否是目录页

        Args:
            page_doc: 页面文档
            toc_page_range: 目录页码范围 (起始, 结束)

        Returns:
            是否为目录页
        """
        if not toc_page_range or toc_page_range[0] is None:
            return False

        page_num = page_doc.metadata.get('page_number', 1)
        toc_start, toc_end = toc_page_range

        return toc_start <= page_num <= toc_end

    def reload_config(self, config_path: Optional[str] = None):
        """
        Reload configuration from file

        Args:
            config_path: Optional path to new configuration file
        """
        config_path = config_path or self.config_path
        self.config_path = config_path
        self.splitters = self._create_splitters()
        logger.info(f"Configuration reloaded from {config_path}")
