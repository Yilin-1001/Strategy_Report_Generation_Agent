"""
使用PyPDF批量转换PDF为TXT
"""
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json
from rag_project.data_loader.pypdf_loader import PyPDFLoader, convert_pdf_with_pypdf
from rag_project.utils.logger import logger


def batch_convert_pdfs_with_pypdf(
    kb_path: str = "知识库/知识库",
    output_dir: Optional[str] = None,
    skip_existing: bool = True
) -> Dict:
    """
    批量转换PDF为TXT（使用PyPDF）

    Args:
        kb_path: 知识库路径
        output_dir: 输出目录（默认保存在PDF同目录）
        skip_existing: 是否跳过已转换的文件

    Returns:
        转换统计结果
    """
    kb = Path(kb_path)
    pdf_files = list(kb.rglob("*.pdf"))

    if not pdf_files:
        logger.warning(f"在 {kb_path} 中未找到PDF文件")
        return {'total': 0, 'converted': 0, 'failed': 0}

    logger.info("="*80)
    logger.info(f"PyPDF批量PDF转TXT工具")
    logger.info("="*80)
    logger.info(f"找到PDF文件: {len(pdf_files)} 个")

    stats = {
        'total': len(pdf_files),
        'converted': 0,
        'failed': 0,
        'skipped': 0,
        'total_pages': 0,
        'total_chars': 0,
        'files': []
    }

    start_time = datetime.now()

    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = pdf_path.name
        logger.info(f"\n[{i}/{len(pdf_files)}] 处理: {pdf_name}")

        # 确定输出路径
        if output_dir:
            output_path = Path(output_dir) / f"{pdf_path.stem}.txt"
        else:
            output_path = pdf_path.parent / f"{pdf_path.stem}.txt"

        # 检查是否已存在
        if skip_existing and output_path.exists():
            logger.info(f"  [跳过] TXT文件已存在")
            stats['skipped'] += 1
            continue

        try:
            # 转换
            result = convert_pdf_with_pypdf(
                str(pdf_path),
                str(output_path),
                add_page_markers=True
            )

            if result['status'] == 'success':
                stats['converted'] += 1
                stats['total_pages'] += result['metadata']['total_pages']
                stats['total_chars'] += result['stats']['total_characters']

                file_info = {
                    'pdf_name': pdf_name,
                    'status': 'success',
                    'output_path': str(output_path),
                    'pages': result['metadata']['total_pages'],
                    'chars': result['stats']['total_characters']
                }
                stats['files'].append(file_info)

                logger.info(f"  [成功] {result['metadata']['total_pages']}页, {result['stats']['total_characters']}字符")

            else:
                stats['failed'] += 1
                logger.error(f"  [失败] {result.get('error')}")

        except Exception as e:
            stats['failed'] += 1
            logger.error(f"  [异常] {e}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("批量转换完成")
    logger.info("="*80)
    logger.info(f"总PDF文件: {stats['total']}")
    logger.info(f"成功转换: {stats['converted']}")
    logger.info(f"转换失败: {stats['failed']}")
    logger.info(f"跳过文件: {stats['skipped']}")
    logger.info(f"总页数: {stats['total_pages']}")
    logger.info(f"总字符数: {stats['total_chars']:,}")
    logger.info(f"处理时间: {duration:.2f} 秒")
    logger.info(f"平均每个文件: {duration/stats['total']:.2f} 秒")
    logger.info("="*80)

    return stats


def compare_pypdf_vs_pymupdf(pdf_path: str) -> Dict:
    """
    对比PyPDF和PyMuPDF的转换结果

    Args:
        pdf_path: PDF文件路径

    Returns:
        对比结果
    """
    logger.info("="*80)
    logger.info(f"PDF转换方法对比: {Path(pdf_path).name}")
    logger.info("="*80)

    results = {}

    # PyPDF方法
    logger.info("\n1. PyPDF方法:")
    try:
        pypdf_start = datetime.now()
        pypdf_result = convert_pdf_with_pypdf(pdf_path, add_page_markers=True)
        pypdf_duration = (datetime.now() - pypdf_start).total_seconds()

        if pypdf_result['status'] == 'success':
            results['pypdf'] = {
                'status': 'success',
                'duration': pypdf_duration,
                'chars': pypdf_result['stats']['total_characters'],
                'pages': pypdf_result['metadata']['total_pages']
            }
            logger.info(f"  ✓ 成功: {pypdf_duration:.2f}秒, {pypdf_result['stats']['total_characters']}字符")
        else:
            results['pypdf'] = {
                'status': 'failed',
                'error': pypdf_result.get('error')
            }
            logger.error(f"  ✗ 失败: {pypdf_result.get('error')}")

    except Exception as e:
        results['pypdf'] = {
            'status': 'error',
            'error': str(e)
        }
        logger.error(f"  ✗ 异常: {e}")

    # PyMuPDF方法（如果可用）
    logger.info("\n2. PyMuPDF方法:")
    try:
        import fitz
        pymupdf_start = datetime.now()

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        text_parts = []
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_marker = f"\n{'='*80}\n第 {page_num + 1} 页 / 共 {total_pages} 页\n{'='*80}\n\n"
            text_parts.append(page_marker + text)

        full_text = '\n'.join(text_parts)
        doc.close()

        pymupdf_duration = (datetime.now() - pymupdf_start).total_seconds()

        results['pymupdf'] = {
            'status': 'success',
            'duration': pymupdf_duration,
            'chars': len(full_text),
            'pages': total_pages
        }
        logger.info(f"  ✓ 成功: {pymupdf_duration:.2f}秒, {len(full_text)}字符")

    except ImportError:
        logger.warning("  - PyMuPDF未安装，跳过对比")
    except Exception as e:
        results['pymupdf'] = {
            'status': 'error',
            'error': str(e)
        }
        logger.error(f"  ✗ 异常: {e}")

    # 打印对比
    logger.info("\n" + "="*80)
    logger.info("对比结果")
    logger.info("="*80)

    if 'pypdf' in results and 'pymupdf' in results:
        if results['pypdf']['status'] == 'success' and results['pymupdf']['status'] == 'success':
            pypdf_chars = results['pypdf']['chars']
            pymupdf_chars = results['pymupdf']['chars']
            diff = pymupdf_chars - pypdf_chars
            diff_pct = (diff / pymupdf_chars) * 100

            logger.info(f"字符数对比:")
            logger.info(f"  PyPDF:    {pypdf_chars:,} 字符")
            logger.info(f"  PyMuPDF:   {pymupdf_chars:,} 字符")
            logger.info(f"  差异:      {diff:+,} 字符 ({diff_pct:+.1f}%)")

            logger.info(f"\n处理速度:")
            logger.info(f"  PyPDF:    {results['pypdf']['duration']:.2f} 秒")
            logger.info(f"  PyMuPDF:   {results['pymupdf']['duration']:.2f} 秒")

            speed_ratio = results['pypdf']['duration'] / results['pymupdf']['duration']
            logger.info(f"  速度比:    PyMuPDF 比 PyPDF 快 {speed_ratio:.1f}x")

    return results


if __name__ == "__main__":
    import sys

    # 示例1: 转换单个PDF
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.pdf'):
        pdf_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        logger.info("使用PyPDF转换单个PDF文件")
        result = convert_pdf_with_pypdf(pdf_file, output_file)

        if result['status'] == 'success':
            print(f"\n✓ 转换成功!")
            print(f"  输出: {result['output_path']}")
            print(f"  页数: {result['metadata']['total_pages']}")
            print(f"  字符: {result['stats']['total_characters']:,}")
        else:
            print(f"\n✗ 转换失败: {result.get('error')}")

    # 示例2: 批量转换
    elif len(sys.argv) == 1:
        logger.info("使用PyPDF批量转换所有PDF文件")
        stats = batch_convert_pdfs_with_pypdf(
            kb_path="知识库/知识库",
            skip_existing=True
        )

        print(f"\n{'='*80}")
        print(f"批量转换完成")
        print(f"{'='*80}")
        print(f"成功: {stats['converted']}/{stats['total']}")
        print(f"失败: {stats['failed']}")
        print(f"字符: {stats['total_chars']:,}")

    else:
        print("用法:")
        print("  转换单个PDF: python pypdf_batch_converter.py <pdf文件> [输出txt]")
        print("  批量转换:   python pypdf_batch_converter.py")
