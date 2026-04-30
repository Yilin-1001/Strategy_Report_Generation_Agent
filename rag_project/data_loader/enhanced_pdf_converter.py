"""
增强型PDF转换器 - 集成行合并功能

处理流程：
1. PDF → 文本提取
2. 行合并（合并非段落换行）
3. 添加分页标记
4. 保存TXT文件
"""
import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# PDF加载��
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from rag_project.data_loader.text_line_merger import TextLineMerger
from rag_project.utils.logger import logger


class EnhancedPDFConverter:
    """
    增强型PDF转换器

    特点：
    1. 使用PyPDF提取文本
    2. 智能合并非段落换行
    3. 添加分页标记（与chunking系统兼容）
    4. 保留段落结构
    """

    def __init__(
        self,
        enable_line_merge: bool = True,
        merge_config: Optional[object] = None
    ):
        """
        初始化增强型PDF转换器

        Args:
            enable_line_merge: 是否启用行合并
            merge_config: 行合并配置
        """
        if not PYPDF_AVAILABLE:
            raise ImportError("PyPDF未安装，请运行: pip install pypdf")

        self.enable_line_merge = enable_line_merge
        self.merger = TextLineMerger(merge_config) if enable_line_merge else None

        logger.info(f"增强型PDF转换器初始化 (行合并: {'启用' if enable_line_merge else '禁用'})")

    def convert_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        add_page_markers: bool = True
    ) -> Dict:
        """
        转换PDF为TXT（带行合并）

        Args:
            pdf_path: PDF文件路径
            output_path: 输出TXT路径
            add_page_markers: 是否添加分页标记

        Returns:
            转换结果字典
        """
        start_time = datetime.now()

        logger.info("="*80)
        logger.info("增强型PDF转换")
        logger.info("="*80)
        logger.info(f"PDF文件: {pdf_path}")
        logger.info(f"行合并: {'启用' if self.enable_line_merge else '禁用'}")

        try:
            # 1. 加载PDF
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            logger.info(f"加载PDF: {total_pages} 页")

            # 2. 提取所有页面文本
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                # 应用行合并（每页单独处理）
                if self.enable_line_merge and page_text:
                    page_text = self.merger.merge_text(page_text)

                # 添加分页标记
                if add_page_markers:
                    page_marker = (
                        f"\n{'='*80}\n"
                        f"第 {page_num + 1} 页 / 共 {total_pages} 页\n"
                        f"{'='*80}\n\n"
                    )
                    text_parts.append(page_marker + page_text)
                else:
                    text_parts.append(page_text)

            # 3. 合并所有页面
            full_text = '\n'.join(text_parts)

            # 4. 保存文件
            pdf_path_obj = Path(pdf_path)
            if output_path is None:
                output_path = pdf_path_obj.parent / f"{pdf_path_obj.stem}.txt"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            # 5. 统计信息
            duration = (datetime.now() - start_time).total_seconds()

            result = {
                'status': 'success',
                'pdf_path': str(pdf_path),
                'output_path': str(output_path),
                'total_pages': total_pages,
                'total_chars': len(full_text),
                'line_merge_enabled': self.enable_line_merge,
                'duration_seconds': duration
            }

            logger.info(f"\n转换成功!")
            logger.info(f"输出: {output_path}")
            logger.info(f"页数: {total_pages}")
            logger.info(f"字符: {len(full_text):,}")
            logger.info(f"耗时: {duration:.2f} 秒")
            logger.info("="*80)

            return result

        except Exception as e:
            logger.error(f"转换失败: {e}")
            return {
                'status': 'failed',
                'pdf_path': str(pdf_path),
                'error': str(e)
            }


def convert_pdf_with_line_merge(
    pdf_path: str,
    output_path: Optional[str] = None,
    enable_line_merge: bool = True
) -> Dict:
    """
    转换PDF为TXT（带行合并）- 便捷函数

    Args:
        pdf_path: PDF文件路径
        output_path: 输出TXT路径
        enable_line_merge: 是否启用行合并

    Returns:
        转换结果字典
    """
    converter = EnhancedPDFConverter(enable_line_merge=enable_line_merge)
    return converter.convert_pdf(pdf_path, output_path)


# 命令行使用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python enhanced_pdf_converter.py <pdf文件> [输出txt] [--no-merge]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    txt_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    enable_merge = '--no-merge' not in sys.argv

    result = convert_pdf_with_line_merge(pdf_file, txt_file, enable_merge)

    if result['status'] == 'success':
        print(f"\n转换成功!")
        print(f"  输出: {result['output_path']}")
        print(f"  页数: {result['total_pages']}")
        print(f"  字符: {result['total_chars']:,}")
        print(f"  行合并: {'是' if result['line_merge_enabled'] else '否'}")
    else:
        print(f"\n转换失败: {result.get('error')}")
        sys.exit(1)
