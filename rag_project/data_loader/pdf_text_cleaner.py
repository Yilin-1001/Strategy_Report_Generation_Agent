"""
PDF转换后的文本清洗器
用于清理从PDF转换的TXT文件，提高chunking质量
"""
import re
from typing import Dict, List, Tuple, Optional, Any, Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

from rag_project.utils.logger import logger


@dataclass
class PageMetadata:
    """页面元数据"""
    page_number: int
    original_text: str
    cleaned_text: str
    has_header: bool = False
    has_footer: bool = False
    is_toc: bool = False
    section_title: Optional[str] = None


class PDFTextCleaner:
    """PDF文本清洗器"""

    def __init__(self):
        """初始化清洗器"""
        # 页码识别模式（多种格式）
        self.page_number_patterns = [
            r'^\s*第\s*(\d+)\s*页\s*$',  # 第1页
            r'^\s*Page\s*(\d+)\s*$',  # Page 1
            r'^\s*(\d+)\s*/\s*\d+\s*$',  # 1/100
            r'^\s*-\s*(\d+)\s*-\s*$',  # - 1 -
            r'^\s*第\s*(\d+)\s*页\s*/\s*共\s*\d+\s*页\s*$',  # 第1页/共100页
        ]

        # 目录识别模式
        self.toc_patterns = [
            r'.*目\s*录.*',  # 目录
            r'.*目\s*　\s*录.*',  # 目录（全角空格）
            r'.*Contents.*',  # 英文目录
            r'.*CONTENTS.*',
            r'.*索\s*引.*',  # 索引
        ]

        # 页眉页脚识别模式
        self.header_footer_patterns = [
            r'.*保密.*',  # 保密声明
            r'.*机密.*',
            r'.*内部.*',
            r'.*版权.*',
            r'.*Copyright.*',
            r'.*第.*期.*',  # 期刊信息
            r'.*Vol\.\d+.*',
        ]

        # 分页标记模式（PyMuPDF添加的）
        self.pympdf_page_marker = re.compile(
            r'\n={80}\n第\s*(\d+)\s*页\s*/\s*共\s*(\d+)\s*页\n={80}\n',
            re.MULTILINE
        )

        logger.info("PDF文本清洗器初始化完成")

    def clean_document(self, text: str, source_file: str) -> Tuple[str, List[PageMetadata]]:
        """
        清洗整个文档

        Args:
            text: 原始文本
            source_file: 源文件名（用于metadata）

        Returns:
            (清洗后的文本, 页面元数据列表)
        """
        logger.info(f"开始清洗文档: {source_file}")

        # 1. 分割成页面（使用PyMuPDF的分页标记）
        pages = self._split_into_pages(text)

        logger.info(f"文档共 {len(pages)} 页")

        # 2. 清洗每一页
        cleaned_pages = []
        page_metadata_list = []

        for page_num, page_text in enumerate(pages, 1):
            metadata = PageMetadata(
                page_number=page_num,
                original_text=page_text,
                cleaned_text=""
            )

            # 检查是否是目录页
            if self._is_toc_page(page_text):
                metadata.is_toc = True
                logger.debug(f"  页 {page_num}: 识别为目录页，将删除")
                page_metadata_list.append(metadata)
                continue

            # 清洗页面内容
            cleaned = self._clean_single_page(page_text, page_num, metadata)

            metadata.cleaned_text = cleaned
            cleaned_pages.append(cleaned)
            page_metadata_list.append(metadata)

        # 3. 合并清洗后的页面
        final_text = '\n\n'.join(cleaned_pages)

        removed_pages = sum(1 for m in page_metadata_list if m.is_toc or not m.cleaned_text.strip())
        logger.info(f"清洗完成: 保留 {len(cleaned_pages)} 页, 删除 {removed_pages} 页")

        return final_text, page_metadata_list

    def _split_into_pages(self, text: str) -> List[str]:
        """
        按照PyMuPDF的分页标记分割文本

        Args:
            text: 原始文本

        Returns:
            页面列表
        """
        # 使用PyMuPDF的分页标记分割
        pages = self.pympdf_page_marker.split(text)

        # 第一个元素可能是文档开头（没有分页标记前的内容）
        # 之后的模式是: 分页标记 + 页面内容
        result = []
        current_page_num = 1

        # 处理第一个元素（如果有内容）
        if pages[0].strip():
            result.append(pages[0])

        # 后续元素成对出现：(页码, 总页数, 内容)
        for i in range(1, len(pages), 3):
            if i + 2 < len(pages):
                page_num = pages[i]
                total_pages = pages[i + 1]
                content = pages[i + 2]
                result.append(content.strip())

        return result

    def _is_toc_page(self, page_text: str) -> bool:
        """
        判断是否是目录页

        Args:
            page_text: 页面文本

        Returns:
            是否是目录页
        """
        # 检查是否包含目录关键词
        lines = page_text.split('\n')

        # 目录通常在前面几行
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue

            for pattern in self.toc_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return True

        # 检查是否有很多页码引用（目录的特征）
        # 例如："...15......23......45" 模式
        page_ref_pattern = re.compile(r'\.{3,}\s*\d+\s*')
        page_refs = page_ref_pattern.findall(page_text)

        if len(page_refs) > 3:  # 如果有3个以上页码引用，可能是目录
            return True

        return False

    def _clean_single_page(self, page_text: str, page_num: int, metadata: PageMetadata) -> str:
        """
        清洗单页内容

        Args:
            page_text: 页面文本
            page_num: 页码
            metadata: 页面元数据（会更新）

        Returns:
            清洗后的文本
        """
        lines = page_text.split('\n')
        cleaned_lines: List[str] = []

        # 识别页眉（前3行）
        header_lines = []
        for i, line in enumerate(lines[:3]):
            if self._is_header_footer_line(line):
                header_lines.append(i)
                metadata.has_header = True

        # 识别页脚（最后3行）
        footer_lines = []
        for i, line in enumerate(lines[-3:], len(lines) - 3):
            if self._is_header_footer_line(line):
                footer_lines.append(i)
                metadata.has_footer = True

        # 保留正文行
        for i, line in enumerate(lines):
            # 跳过页眉
            if i in header_lines:
                continue

            # 跳过页脚
            if i in footer_lines:
                continue

            # 跳过空行（但保留段落间的一个空行）
            stripped = line.strip()
            if not stripped:
                # 检查前一行是否也是空行
                if cleaned_lines and not cleaned_lines[-1].strip():
                    continue
                else:
                    cleaned_lines.append('')
                    continue

            # 移除行首行尾空白
            cleaned_lines.append(stripped)

        # 合并行
        cleaned = '\n'.join(cleaned_lines)

        # 移除多余的空行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return cleaned.strip()

    def _is_header_footer_line(self, line: str) -> bool:
        """
        判断是否是页眉页脚行

        Args:
            line: 文本行

        Returns:
            是否是页眉页脚
        """
        line = line.strip()

        # 空行不是页眉页脚
        if not line:
            return False

        # 检查页码模式
        for pattern in self.page_number_patterns:
            if re.match(pattern, line):
                return True

        # 检查页眉页脚模式
        for pattern in self.header_footer_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        # 检查是否是纯数字或短数字行
        if re.match(r'^\d{1,3}$', line):
            return True

        # 检查是否是短字符行（可能是页眉页脚的装饰线等）
        if len(line) <= 3 and not re.search(r'[a-zA-Z\u4e00-\u9fa5]', line):
            return True

        return False

    def extract_section_info(self, text: str) -> Dict[str, Any]:
        """
        提取章节信息

        Args:
            text: 文本内容

        Returns:
            章节信息字典
        """
        info: Dict[str, Any] = {
            'title': '',
            'sections': []
        }

        # 提取标题（第一个非空行，可能是标题）
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                info['title'] = line
                break

        # 提取章节标题
        chapter_patterns = [
            r'^(第[一二三四五六七八九十\d]+章|Chapter\s+\d+)\s+(.+)$',
            r'^(\d+\.?\s*)([^\d]+)$',  # "1. 标题" 或 "1.1 标题"
            r'^([一二三四五六七八九十]+、\s*)(.+)$',
        ]

        for line in lines:
            line = line.strip()
            for pattern in chapter_patterns:
                match = re.match(pattern, line)
                if match:
                    info['sections'].append(line)
                    break

        return info


def clean_converted_pdf_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.txt"
) -> Dict[str, Any]:
    """
    ���量清洗转换后的PDF文本

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式

    Returns:
        清洗统计信息
    """
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cleaner = PDFTextCleaner()

    # 查找所有TXT文件
    txt_files = list(input_path.glob(pattern))

    logger.info(f"找到 {len(txt_files)} 个文件待清洗")

    stats = {
        'total': len(txt_files),
        'cleaned': 0,
        'failed': 0,
        'total_pages': 0,
        'removed_pages': 0
    }

    for txt_file in txt_files:
        try:
            logger.info(f"\n清洗: {txt_file.name}")

            # 读取原始文本
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_text = f.read()

            # 清洗文本
            cleaned_text, page_metadata = cleaner.clean_document(
                original_text,
                txt_file.name
            )

            # 保存清洗后的文本
            output_file = output_path / txt_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            # 保存元数据
            metadata_file = output_path / f"{txt_file.stem}_metadata.json"
            import json
            metadata_data = [
                {
                    'page_number': m.page_number,
                    'has_header': m.has_header,
                    'has_footer': m.has_footer,
                    'is_toc': m.is_toc,
                    'section_title': m.section_title
                }
                for m in page_metadata
            ]
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, ensure_ascii=False, indent=2)

            # 更新统计
            stats['cleaned'] += 1
            stats['total_pages'] += len(page_metadata)
            stats['removed_pages'] += sum(1 for m in page_metadata if m.is_toc)

            logger.info(f"  ✓ 保存到: {output_file.name}")
            logger.info(f"  ✓ 元数据: {metadata_file.name}")

        except Exception as e:
            logger.error(f"  ✗ 清洗失败: {e}")
            stats['failed'] += 1

    logger.info(f"\n{'='*80}")
    logger.info("清洗完成")
    logger.info(f"{'='*80}")
    logger.info(f"总文件数: {stats['total']}")
    logger.info(f"成功清洗: {stats['cleaned']}")
    logger.info(f"清洗失败: {stats['failed']}")
    logger.info(f"总页数: {stats['total_pages']}")
    logger.info(f"删除页数: {stats['removed_pages']}")

    return stats


def clean_converted_pdf_files(
    knowledge_base_dir: str,
    pdf_conversion_log: Optional[str] = None
) -> Dict[str, Any]:
    """
    批量清洗转换后的PDF文本（文件分散在各目录中）

    Args:
        knowledge_base_dir: 知识库根目录
        pdf_conversion_log: PDF转换日志路径（用于识别哪些TXT是从PDF转换的）

    Returns:
        清洗统计信息
    """
    from pathlib import Path
    import json

    kb_path = Path(knowledge_base_dir)

    cleaner = PDFTextCleaner()

    # 加载PDF转换日志，识别哪些TXT是从PDF转换的
    converted_files = set()
    if pdf_conversion_log:
        log_path = Path(pdf_conversion_log)
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                conversion_log = json.load(f)
                # 提取已转换的PDF文件名
                for pdf_name, info in conversion_log.items():
                    if info.get("status") == "success":
                        # PDF转TXT的文件名
                        pdf_stem = Path(pdf_name).stem
                        converted_files.add(pdf_stem)

    # 递归查找所有TXT文件
    txt_files = list(kb_path.rglob("*.txt"))

    # 过滤：只处理从PDF转换的TXT文件
    if converted_files:
        txt_files = [f for f in txt_files if f.stem in converted_files]
        logger.info(f"根据转换日志，找到 {len(txt_files)} 个PDF转换的TXT文件")
    else:
        logger.info(f"在 {kb_path} 中找到 {len(txt_files)} 个TXT文件")
        logger.info("提示: 未提供转换日志，将清洗所有TXT文件")

    stats = {
        'total': len(txt_files),
        'cleaned': 0,
        'failed': 0,
        'total_pages': 0,
        'removed_pages': 0,
        'skipped': 0
    }

    for txt_file in txt_files:
        try:
            # 相对路径，便于显示
            rel_path = txt_file.relative_to(kb_path)
            logger.info(f"\n清洗: {rel_path}")

            # 读取原始文本
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_text = f.read()

            if not original_text.strip():
                logger.warning(f"  跳过空文件")
                stats['skipped'] += 1
                continue

            # 清洗文本
            cleaned_text, page_metadata = cleaner.clean_document(
                original_text,
                txt_file.name
            )

            # 保存清洗后的文本（覆盖原文件）
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            # 保存元数据（在同一目录）
            metadata_file = txt_file.parent / f"{txt_file.stem}_metadata.json"
            metadata_data = [
                {
                    'page_number': m.page_number,
                    'has_header': m.has_header,
                    'has_footer': m.has_footer,
                    'is_toc': m.is_toc,
                    'section_title': m.section_title
                }
                for m in page_metadata
            ]
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, ensure_ascii=False, indent=2)

            # 更新统计
            stats['cleaned'] += 1
            stats['total_pages'] += len(page_metadata)
            stats['removed_pages'] += sum(1 for m in page_metadata if m.is_toc)

            rel_metadata = metadata_file.relative_to(kb_path)
            logger.info(f"  已清洗并覆盖原文件")
            logger.info(f"  元数据: {rel_metadata}")

        except Exception as e:
            logger.error(f"  清洗失败: {e}")
            stats['failed'] += 1

    logger.info(f"\n{'='*80}")
    logger.info("清洗完成")
    logger.info(f"{'='*80}")
    logger.info(f"总文件数: {stats['total']}")
    logger.info(f"成功清洗: {stats['cleaned']}")
    logger.info(f"清洗失败: {stats['failed']}")
    logger.info(f"跳过文件: {stats['skipped']}")
    logger.info(f"总页数: {stats['total_pages']}")
    logger.info(f"删除页数: {stats['removed_pages']}")

    return stats


if __name__ == "__main__":
    # 测试单个文件清洗
    test_text = """
================================================================================
第 1 页 / 共 10 页
================================================================================

中国通用航空2021年度报告

第一章  概述

通用航空是指...

第 1 页

"""

    cleaner = PDFTextCleaner()
    cleaned, metadata = cleaner.clean_document(test_text, "test.pdf")

    print("清洗后的文本:")
    print(cleaned)
    print("\n元数据:")
    for m in metadata:
        print(f"  页 {m.page_number}: header={m.has_header}, footer={m.has_footer}, toc={m.is_toc}")
