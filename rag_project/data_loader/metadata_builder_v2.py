"""
增强的Metadata构建器 - 第二版
支持自定义metadata结构
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from langchain_core.documents import Document
from rag_project.utils.logger import logger


class DocumentIDGenerator:
    """生成document_id"""

    @staticmethod
    def generate(file_path: str) -> str:
        """
        根据文件路径生成document_id
        规则: 移除扩展名，保留所有中文字符和特殊字符

        Args:
            file_path: 文件路径

        Returns:
            document_id
        """
        # 获取文件名（含扩展名）
        file_name = Path(file_path).name

        # 移除扩展名
        document_id = Path(file_name).stem

        return document_id


class DateExtractor:
    """提取文档内容日期"""

    # 日期识别模式（按优先级排序）
    DATE_PATTERNS = [
        # YYYY-MM-DD
        (r'(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        # YYYY/MM/DD
        (r'(\d{4})/(\d{2})/(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        # YYYY.MM.DD
        (r'(\d{4})\.(\d{2})\.(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        # YYYY年MM月DD日
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        # YYYYMMDD (8位数字)
        (r'(\d{4})(\d{2})(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    ]

    # 年份识别模式（仅年份）
    YEAR_PATTERNS = [
        r'(\d{4})年',
        r'(\d{4})年度',
        r'(\d{4})',
    ]

    # 关键词日期模式（包含特定关键词的日期行）
    KEYWORD_DATE_PATTERNS = [
        (r'(?:发布时间|Publish Date|报导日期|日期|时间|Date|Time)[：:]\s*(\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2}[日号]?)',
         lambda m: m.group(1)),
    ]

    @staticmethod
    def extract_from_filename(file_path: str) -> Optional[str]:
        """
        从文件名提取日期（最高优先级）

        Args:
            file_path: 文件路径

        Returns:
            日期字符串 (YYYY-MM-DD) 或年份 (YYYY) 或 None
        """
        file_name = Path(file_path).name

        # 优先尝试提取完整日期
        for pattern, formatter in DateExtractor.DATE_PATTERNS:
            match = re.search(pattern, file_name)
            if match:
                try:
                    return formatter(match)
                except:
                    continue

        # 如果没有完整日期，尝试提取年份
        for pattern in DateExtractor.YEAR_PATTERNS:
            match = re.search(pattern, file_name)
            if match:
                year = match.group(1)
                # 验证是合理年份（1900-2100）
                if 1900 <= int(year) <= 2100:
                    return year

        return None

    @staticmethod
    def extract_from_content(content: str) -> Optional[str]:
        """
        从文件内容提取日期（第二优先级）

        Args:
            content: 文档内容

        Returns:
            日期字符串 (YYYY-MM-DD) 或年份 (YYYY) 或 None
        """
        # 只检查前2000个字符（通常在头部）
        header = content[:2000]

        # 优先查找关键词日期
        for pattern, formatter in DateExtractor.KEYWORD_DATE_PATTERNS:
            match = re.search(pattern, header, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # 标准化为 YYYY-MM-DD
                return DateExtractor._normalize_date(date_str)

        # 查找普通日期模式
        for pattern, formatter in DateExtractor.DATE_PATTERNS:
            match = re.search(pattern, header)
            if match:
                try:
                    return formatter(match)
                except:
                    continue

        # 查找年份（在头部前500字符）
        header_first_500 = header[:500]
        for pattern in DateExtractor.YEAR_PATTERNS:
            match = re.search(pattern, header_first_500)
            if match:
                year = match.group(1)
                if 1900 <= int(year) <= 2100:
                    return year

        return None

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """标准化日期为 YYYY-MM-DD 格式"""
        # 移除中文字符
        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
        date_str = date_str.replace('/', '-').replace('.', '-')

        # 尝试解析
        try:
            # 处理 "2025-1-5" → "2025-01-05"
            parts = date_str.split('-')
            if len(parts) == 3:
                year, month, day = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            pass

        return date_str

    @staticmethod
    def extract(file_path: str, content: str) -> Optional[str]:
        """
        提取文档日期（综合策略）

        优先级: 文件名 > 内容头部 > null

        Args:
            file_path: 文件路径
            content: 文档内容

        Returns:
            日期字符串 (YYYY-MM-DD 或 YYYY) 或 None
        """
        # 优先级1: 文件名
        date_from_filename = DateExtractor.extract_from_filename(file_path)
        if date_from_filename:
            return date_from_filename

        # 优先级2: 内容头部
        date_from_content = DateExtractor.extract_from_content(content)
        if date_from_content:
            return date_from_content

        # 找不到日期
        return None


class SectionTitleExtractor:
    """提取章节标题"""

    # 章节标题识别模式（更严格的模式）
    SECTION_PATTERNS = [
        # 第X章格式
        (r'^第[一二三四五六七八九十百千零\d]+章[：::\s]*(.*?)$', re.MULTILINE),
        # Chapter X格式（英文）
        (r'^Chapter\s+[IVXLCDM\d]+[：::\s]*(.*?)$', re.MULTILINE),
        # X.Y 格式（小节，如"3.2 实验设计"）
        (r'^(\d+\.\d+)\s{1,5}(.{2,50})$', re.MULTILINE),
        # 特定关键词标题
        (r'^(摘要|前言|序言|目录|参考文献|附录|Abstract|Preface|Contents|Introduction|Conclusion)(?:[：:]|\s|$)', re.MULTILINE),
        # 【标题】格式
        (r'^【(.{2,30})】$', re.MULTILINE),
        # 观察专栏等特殊章节
        (r'^观察专栏\d+[：::\s]*(.{2,50})$', re.MULTILINE),
    ]

    @staticmethod
    def extract_sections(content: str) -> List[Tuple[int, int, str]]:
        """
        从内容中提取所有章节标题

        Args:
            content: 文档内容

        Returns:
            [(start_pos, end_pos, title), ...]
        """
        sections = []

        for pattern, _ in SectionTitleExtractor.SECTION_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                start_pos = match.start()
                matched_text = match.group(0)

                # 提取标题文本
                if len(match.groups()) >= 2 and match.group(2):
                    title = match.group(2).strip()
                elif len(match.groups()) >= 1:
                    title = match.group(1).strip()
                else:
                    title = matched_text.strip()

                # 如果匹配的是"第X章"格式，提取完整标题
                if '章' in matched_text[:10]:
                    title = matched_text.strip()

                # 如果匹配的是Chapter格式，提取完整标题
                if matched_text.startswith('Chapter'):
                    title = matched_text.strip()

                # 严格过滤条件
                # 1. 标题不能太长（章节标题一般不超过80字符）
                if len(title) > 80:
                    continue

                # 2. 标题不能太短（至少2个字符，排除单个数字）
                if len(title) < 2:
                    continue

                # 3. 标题不应该包含过多数字（超过3个数字可能是数据）
                digit_count = sum(c.isdigit() for c in title)
                if digit_count > 3:
                    continue

                # 4. 排除包含统计数据的特征
                skip_keywords = ['%', '％', '万架次', '万公里', '万吨', '万人次', '亿元', '万辆',
                               '公里/百', '同比增长', '同比下降', '增长', '下降', '提高', '地震', '台风']
                if any(kw in title for kw in skip_keywords):
                    continue

                # 5. ���题应该包含中文字符（除非是纯英文标题如"Abstract"）
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in title)
                has_english_keyword = any(kw in title for kw in ['Abstract', 'Preface', 'Contents', 'Introduction', 'Conclusion'])
                if not has_chinese and not has_english_keyword:
                    continue

                # 6. 标题不应该包含过多标点符号（超过2个可能是句子）
                punctuation_count = sum(c in '，。！？；：、,.!?:;' for c in title)
                if punctuation_count > 2:
                    continue

                # 7. 特殊格式【标题】或"第X章"格式直接接受
                if title.startswith('【') and title.endswith('】'):
                    sections.append((start_pos, start_pos + len(title), title))
                    continue
                if title.startswith('第') and '章' in title[:10]:
                    sections.append((start_pos, start_pos + len(title), title))
                    continue

                sections.append((start_pos, start_pos + len(title), title))

        # 按位置排序
        sections.sort(key=lambda x: x[0])

        return sections

    @staticmethod
    def get_section_for_position(position: int, sections: List[Tuple[int, int, str]]) -> Optional[str]:
        """
        根据文本位置获取对应的章节标题

        找到最后一个位置在当前position之前的章节标题
        （章节标题持续到下一个章节标题出现）

        Args:
            position: 字符位置
            sections: 章节列表 [(start, end, title), ...]

        Returns:
            章节标题或None
        """
        current_section = None

        for start_pos, end_pos, title in sections:
            if start_pos <= position:
                # 这个章节标题在当前position之前
                # 不断更新，最后保留最接近的一个
                current_section = title
            else:
                # 已经超过当前position，停止查找
                break

        return current_section


class TagsExtractor:
    """从文件路径提取tags"""

    @staticmethod
    def extract_from_path(file_path: str, base_dir: str = "知识库/知识库") -> List[str]:
        """
        从文件路径提取tags

        规则: 从base_dir下一级开始，保留所有层级

        Args:
            file_path: 文件路径
            base_dir: 知识库根目录

        Returns:
            tags列表
        """
        path = Path(file_path)
        base = Path(base_dir)

        # 获取相对路径
        try:
            rel_path = path.relative_to(base)
        except ValueError:
            # 文件不在base_dir下
            return []

        # 提取所有层级的目录名
        tags = []
        for part in rel_path.parts[:-1]:  # 排除文件名
            if part and part != '.':  # 排除空和当前目录
                tags.append(part)

        return tags

    @staticmethod
    def extract_doc_type(file_path: str, base_dir: str = "知识库/知识库") -> str:
        """
        从文件路径提取文档类型

        根据文件所在的文件夹名称判断文档类型

        Args:
            file_path: 文件路径
            base_dir: 知识库根目录

        Returns:
            文档类型字符串
        """
        path = Path(file_path)
        base = Path(base_dir)

        try:
            rel_path = path.relative_to(base)
        except ValueError:
            rel_path = path

        # 检查父目录名称来判断类型
        parent_dirs = [part.lower() for part in rel_path.parts[:-1]]

        if 'news' in parent_dirs or '新闻' in parent_dirs:
            return 'news'
        elif '研报' in parent_dirs or 'report' in parent_dirs:
            return 'report'
        elif '论文' in parent_dirs or 'paper' in parent_dirs or 'research' in parent_dirs:
            return 'paper'
        elif '政策' in parent_dirs or 'policy' in parent_dirs or '规定' in parent_dirs:
            return 'policy'
        elif '法规' in parent_dirs or 'regulation' in parent_dirs:
            return 'regulation'
        else:
            return 'document'


class ChunkIndexCounter:
    """管理chunk_index计数（文档级别）"""

    def __init__(self):
        self.document_counters = {}  # {document_id: current_index}

    def get_next_index(self, document_id: str) -> int:
        """
        获取下一个chunk_index

        Args:
            document_id: 文档ID

        Returns:
            下一个可用的chunk_index
        """
        if document_id not in self.document_counters:
            self.document_counters[document_id] = 0

        self.document_counters[document_id] += 1
        return self.document_counters[document_id]

    def reset(self, document_id: Optional[str] = None):
        """
        重置计数器

        Args:
            document_id: 文档ID，如果为None则重置所有
        """
        if document_id:
            self.document_counters.pop(document_id, None)
        else:
            self.document_counters = {}


class MetadataBuilderV2:
    """增强的Metadata构建器（第二版）"""

    def __init__(self, base_dir: str = "知识库/知识库"):
        """
        初始化Metadata构建器

        Args:
            base_dir: 知识库根目录
        """
        # 将base_dir转换为绝对路径，以便与full_path进行比较
        base_path = Path(base_dir)
        if base_path.is_absolute():
            self.base_dir = str(base_path)
        else:
            # 如果是相对路径，转换为绝对路径
            self.base_dir = str(base_path.resolve())
        self.chunk_index_counter = ChunkIndexCounter()
        logger.info(f"MetadataBuilderV2 initialized with base_dir: {self.base_dir}")

    def build_metadata(
        self,
        chunk_content: str,
        chunk_position: int,
        file_path: str,
        page_number: int,
        document_id: Optional[str] = None,
        sections: Optional[List[Tuple[int, int, str]]] = None
    ) -> Dict[str, Any]:
        """
        构建完整的chunk metadata

        Args:
            chunk_content: chunk内容
            chunk_position: chunk在文档中的字符位置
            file_path: 原始文件路径
            page_number: 页码
            document_id: 文档ID（可选，如果不提供则自动生成）
            sections: 章节列表（可选）

        Returns:
            完整的metadata字典
        """
        # 生成document_id（如果未提供）
        if document_id is None:
            document_id = DocumentIDGenerator.generate(file_path)

        # 生成file_name
        file_name = Path(file_path).name

        # 提取section_title
        section_title = None
        if sections:
            section_title = SectionTitleExtractor.get_section_for_position(
                chunk_position, sections
            )

        # 提取tags
        tags = TagsExtractor.extract_from_path(file_path, self.base_dir)

        # 提取created_at
        created_at = DateExtractor.extract(file_path, chunk_content)

        # 获取chunk_index
        chunk_index = self.chunk_index_counter.get_next_index(document_id)

        # 生成chunk_id
        chunk_id = f"{document_id}_chunk_{chunk_index}"

        # 构建metadata
        # 将tags列表转换为字符串，用逗号分隔
        tags_str = ", ".join(tags) if tags else ""

        # 提取doc_type和title
        doc_type = TagsExtractor.extract_doc_type(file_path, self.base_dir)
        title = DocumentIDGenerator.generate(file_path)  # 使用document_id作为title

        metadata = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "file_name": file_name,
            "source": file_path,  # 添加完整���件路径作为source
            "doc_type": doc_type,   # 添加文档类型
            "title": title,         # 添加标题
            "page_number": page_number,
            "section_title": section_title,
            "chunk_index": chunk_index,
            "created_at": created_at,
            "tags": tags_str
        }

        return metadata

    def build_from_document(
        self,
        document: Document,
        chunk_position: int,
        page_number: int
    ) -> Dict[str, Any]:
        """
        从LangChain Document构建metadata

        Args:
            document: LangChain Document对象
            chunk_position: chunk在文档中的位置
            page_number: 页码

        Returns:
            完整的metadata字典
        """
        file_path = document.metadata.get('source', '')

        # 预先提取章节（如果文档中有多个chunks，只提取一次）
        if not hasattr(document, '_sections_extracted'):
            sections = SectionTitleExtractor.extract_sections(document.page_content)
            # 使用 setattr 来动态添加属性，避免类型检查错误
            object.__setattr__(document, '_sections', sections)
            object.__setattr__(document, '_sections_extracted', True)
        else:
            sections = getattr(document, '_sections', [])

        return self.build_metadata(
            chunk_content=document.page_content,
            chunk_position=chunk_position,
            file_path=file_path,
            page_number=page_number,
            sections=sections
        )


# 便捷函数
def create_metadata_builder(base_dir: str = "知识库/知识库") -> MetadataBuilderV2:
    """创建MetadataBuilder实例"""
    return MetadataBuilderV2(base_dir)
