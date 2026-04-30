"""
测试chunking时保留页码的功能
"""
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from rag_project.utils.logger import logger

def test_page_number_preservation():
    """测试页码保留功能"""

    logger.info("="*80)
    logger.info("测试页码保留功能")
    logger.info("="*80)

    # 初始化chunker
    chunker = ConfigurableChunker("config/chunking_config.yaml")

    # 创建测试文档（包含分页标记）
    test_text = """
================================================================================
第 1 页 / 共 3 页
================================================================================

这是第一页的内容。
通用航空是指使用民用航空器从事公共航空运输以外的民用航空活动，包括从事工业、农业、林业、渔业和建筑业的作业飞行以及医疗卫生、抢险救灾、气象探测、海洋监测、科学实验、教育训练、文化体育等方面的飞行活动。

通用航空产业是国家战略性新兴产业，对经济社会发展和科技创新具有重要意义。

================================================================================
第 2 页 / 共 3 页
================================================================================

这是第二页的内容。
通用航空产业包括通用航空器研发制造、通用航空运营、通用航空服务保障等环节。

中国通用航空产业经过多年发展，已经形成了较为完整的产业体系，在经济社会发展中发挥着重要作用。

================================================================================
第 3 页 / 共 3 页
================================================================================

这是第三页的内容。
通用航空产业的发展需要政府支持、市场需求和技术创新三者协同推进。

未来，中国通用航空产业将迎来新的发展机遇。
"""

    doc = Document(
        page_content=test_text,
        metadata={
            "source": "test_document.pdf",
            "doc_type": "pdf"
        }
    )

    logger.info("\n原始文档内容:")
    logger.info("-" * 80)
    logger.info(test_text[:200] + "...")
    logger.info(f"总字符数: {len(test_text)}")

    # 执行chunking
    logger.info("\n开始chunking...")
    chunks = chunker.split_documents([doc], doc_type='pdf')

    logger.info(f"\n生成 {len(chunks)} 个chunks")
    logger.info("="*80)

    # 显示每个chunk的信息
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"\nChunk {i}:")
        logger.info(f"  页码: {chunk.metadata.get('page_number', 'N/A')}")
        logger.info(f"  来源: {chunk.metadata.get('source', 'N/A')}")
        logger.info(f"  文档类型: {chunk.metadata.get('doc_type', 'N/A')}")
        logger.info(f"  字符数: {len(chunk.page_content)}")

        # 显示前100个字符
        preview = chunk.page_content[:100].replace('\n', ' ')
        logger.info(f"  内容预览: {preview}...")

        # 检查是否包含分页标记（不应该包含）
        has_marker = "第" in chunk.page_content and "页" in chunk.page_content
        if has_marker:
            logger.warning(f"  ⚠️  警告: chunk中可能包含分页标记！")
        else:
            logger.info(f"  ✓ 确认: chunk中不包含分页标记")

    # 统计页码分布
    page_distribution = {}
    for chunk in chunks:
        page_num = chunk.metadata.get('page_number', 'Unknown')
        page_distribution[page_num] = page_distribution.get(page_num, 0) + 1

    logger.info("\n" + "="*80)
    logger.info("页码分布统计:")
    logger.info("="*80)
    for page_num in sorted(page_distribution.keys()):
        count = page_distribution[page_num]
        logger.info(f"  第 {page_num} 页: {count} 个chunks")

    # 验证
    logger.info("\n" + "="*80)
    logger.info("验证结果:")
    logger.info("="*80)

    all_have_page_numbers = all('page_number' in chunk.metadata for chunk in chunks)
    none_have_markers = not any("第" in c.page_content and "页" in c.page_content for c in chunks)

    if all_have_page_numbers:
        logger.info("✓ 所有chunks都有页码信息")
    else:
        logger.warning("✗ 部分chunks缺少页码信息")

    if none_have_markers:
        logger.info("✓ 所有chunks都不包含分页标记")
    else:
        logger.warning("✗ 部分chunks包含分页标记")

    success = all_have_page_numbers and none_have_markers

    logger.info("\n" + "="*80)
    if success:
        logger.info("✓ 测试通过！页码保留功能正常工作")
    else:
        logger.error("✗ 测试失败！")
    logger.info("="*80)

    return success


if __name__ == "__main__":
    success = test_page_number_preservation()
    exit(0 if success else 1)
