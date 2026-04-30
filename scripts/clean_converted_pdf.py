"""
批量清洗转换后的PDF文本文件
（TXT文件保存在PDF原文件同目录）
"""
from pathlib import Path
from rag_project.data_loader.pdf_text_cleaner import clean_converted_pdf_files
from rag_project.utils.logger import logger


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("PDF文本清洗工具")
    logger.info("="*80)

    knowledge_base_path = "知识库/知识库"
    pdf_conversion_log = "知识库/知识库/pdf_conversion_log.json"

    # 检查知识库目录
    kb_path = Path(knowledge_base_path)
    if not kb_path.exists():
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        return False

    logger.info(f"知识库目录: {knowledge_base_path}")

    # 检查转换日志
    log_path = Path(pdf_conversion_log)
    if log_path.exists():
        logger.info(f"找到转换日志: {pdf_conversion_log}")
        logger.info("将根据日志清洗PDF转换的TXT文件")
    else:
        logger.warning(f"未找到转换日志: {pdf_conversion_log}")
        logger.warning("将清洗所有TXT文件")

    # 执行批量清洗
    try:
        stats = clean_converted_pdf_files(
            knowledge_base_dir=knowledge_base_path,
            pdf_conversion_log=pdf_conversion_log if log_path.exists() else None
        )

        # 打印总结
        logger.info("\n" + "="*80)
        logger.info("清洗总结")
        logger.info("="*80)
        logger.info(f"处理文件: {stats['total']} 个")
        logger.info(f"成功清洗: {stats['cleaned']} 个")
        logger.info(f"跳过文件: {stats['skipped']} 个")
        logger.info(f"清洗失败: {stats['failed']} 个")

        if stats['cleaned'] > 0:
            success_rate = (stats['cleaned'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"成功率: {success_rate:.1f}%")
            logger.info(f"总页数: {stats['total_pages']} 页")
            logger.info(f"删除页数: {stats['removed_pages']} 页 (主要是目录页)")

            logger.info(f"\n清洗方式: 覆盖原TXT文件")
            logger.info(f"元数据文件: 保存在TXT文件同目录 (*_metadata.json)")

        return stats['failed'] == 0

    except Exception as e:
        logger.error(f"清洗过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
