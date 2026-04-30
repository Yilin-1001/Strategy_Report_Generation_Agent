"""
测试"中国交通运输2021"文档的chunking和页码保留
"""
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from rag_project.utils.logger import logger
from pathlib import Path


def test_china_transport_2021():
    """测试中国交通运输2021文档"""

    logger.info("="*80)
    logger.info("测试文档: 中国交通运输2021")
    logger.info("="*80)

    # 文档路径
    doc_path = "知识库/知识库/相关研报/中国交通运输2021.txt"

    # 检查文件是否存在
    if not Path(doc_path).exists():
        logger.error(f"文件不存在: {doc_path}")
        logger.info("提示: 请先运行 convert_pdf_to_txt.py 转换PDF文件")
        return False

    # 读取文档
    logger.info(f"\n读取文档: {doc_path}")
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()

    logger.info(f"文档大小: {len(text):,} 字符")
    logger.info(f"行数: {text.count(chr(10))}")

    # 检查是否包含分页标记
    has_markers = "第" in text and "页" in text
    marker_count = text.count("第 ") if has_markers else 0

    logger.info(f"包含分页标记: {'是' if has_markers else '否'}")
    if has_markers:
        logger.info(f"分页标记数量: {marker_count}")

    # 显示文档开头
    logger.info("\n文档开头 (前500字符):")
    logger.info("-" * 80)
    logger.info(text[:500])
    logger.info("...")

    # 创建Document对象
    doc = Document(
        page_content=text,
        metadata={
            "source": "中国交通运输2021.txt",
            "doc_type": "pdf"
        }
    )

    # 初始化chunker
    logger.info("\n初始化Chunker...")
    chunker = ConfigurableChunker("config/chunking_config.yaml")

    # 执行chunking
    logger.info("\n开始chunking...")
    chunks = chunker.split_documents([doc], doc_type='pdf')

    logger.info(f"\n生成chunks: {len(chunks)} 个")
    logger.info("="*80)

    # 统计页码分布
    page_distribution = {}
    page_sizes = {}

    for chunk in chunks:
        page_num = chunk.metadata.get('page_number', 'Unknown')
        page_distribution[page_num] = page_distribution.get(page_num, 0) + 1

        # 记录每页的chunk大小
        if page_num not in page_sizes:
            page_sizes[page_num] = []
        page_sizes[page_num].append(len(chunk.page_content))

    # 显示页码分布
    logger.info("\n页码分布统计:")
    logger.info("-" * 80)
    for page_num in sorted(page_distribution.keys()):
        count = page_distribution[page_num]
        sizes = page_sizes[page_num]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        logger.info(f"  第 {page_num} 页: {count} 个chunks, 平均大小 {avg_size:.0f} 字符 (最小 {min_size}, 最大 {max_size})")

    # 显示前10个chunks的详情
    logger.info("\n" + "="*80)
    logger.info("前10个Chunks详情:")
    logger.info("="*80)

    for i, chunk in enumerate(chunks[:10], 1):
        page_num = chunk.metadata.get('page_number', 'N/A')
        char_count = len(chunk.page_content)

        # 获取内容预览（前150字符）
        preview = chunk.page_content[:150].replace('\n', ' ').strip()

        logger.info(f"\nChunk {i}:")
        logger.info(f"  页码: {page_num}")
        logger.info(f"  字符数: {char_count}")
        logger.info(f"  来源: {chunk.metadata.get('source', 'N/A')}")
        logger.info(f"  内容: {preview}...")

        # 检查是否包含分页标记（不应该包含）
        has_marker = "第 " in chunk.page_content and "页 / 共" in chunk.page_content
        if has_marker:
            logger.warning(f"  ⚠ 警告: chunk包含分页标记！")
        else:
            logger.info(f"  ✓ 确认: 不包含分页标记")

    if len(chunks) > 10:
        logger.info(f"\n... 还有 {len(chunks) - 10} 个chunks")

    # 验证结果
    logger.info("\n" + "="*80)
    logger.info("验证结果:")
    logger.info("="*80)

    all_have_page_numbers = all('page_number' in chunk.metadata for chunk in chunks)
    none_have_markers = not any("第 " in c.page_content and "页 / 共" in c.page_content for c in chunks)
    all_pages_covered = len(page_distribution) == 305  # 中国交通运输2021有305页

    logger.info(f"所有chunks都有页码: {'✓' if all_have_page_numbers else '✗'}")
    logger.info(f"所有chunks都不含分页标记: {'✓' if none_have_markers else '✗'}")
    logger.info(f"覆盖页数: {len(page_distribution)} / 305 ({'✓' if all_pages_covered else '✗'})")

    # 显示一些特殊页码的chunks
    logger.info("\n" + "="*80)
    logger.info("特定页码的Chunks示例:")
    logger.info("="*80)

    test_pages = [1, 50, 100, 200, 305]
    for test_page in test_pages:
        page_chunks = [c for c in chunks if c.metadata.get('page_number') == test_page]
        if page_chunks:
            logger.info(f"\n第 {test_page} 页 ({len(page_chunks)} 个chunks):")
            for i, chunk in enumerate(page_chunks[:2], 1):  # 最多显示2个
                preview = chunk.page_content[:100].replace('\n', ' ')
                logger.info(f"  Chunk {i}: {preview}...")
        else:
            logger.warning(f"第 {test_page} 页: 没有找到chunks")

    # 最终结果
    success = all_have_page_numbers and none_have_markers

    logger.info("\n" + "="*80)
    if success:
        logger.info("✓ 测试通过！页码保留功能正常工作")
    else:
        logger.error("✗ 测试失败！")
    logger.info("="*80)

    return success


if __name__ == "__main__":
    success = test_china_transport_2021()
    exit(0 if success else 1)
