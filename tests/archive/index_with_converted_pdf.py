"""
索引知识库 - 包含原始TXT和PDF转换后的TXT
跳过DOCX文件
"""
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger
import time


def index_with_converted_pdf():
    """索引文档（包含转换后的PDF文本）"""
    logger.info("="*80)
    logger.info("KNOWLEDGE BASE INDEXING - TXT + Converted PDF")
    logger.info("Processing: Original TXT + Converted PDF TXT")
    logger.info("Skipping: DOCX files")
    logger.info("Embedding: GPU acceleration")
    logger.info("="*80)

    knowledge_base_path = "知识库/知识库"
    knowledge_base_dir = Path(knowledge_base_path)

    if not knowledge_base_dir.exists():
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        return False

    # 收集文档
    logger.info("\n扫描文档...")
    file_paths = []

    # 1. 原始TXT文件
    original_txt_files = list(knowledge_base_dir.glob("*.txt"))
    file_paths.extend(original_txt_files)
    logger.info(f"  原始TXT文件: {len(original_txt_files)} 个")

    # 2. 转换后的PDF文本文件
    converted_dir = knowledge_base_dir / "converted_txt"
    if converted_dir.exists():
        converted_txt_files = list(converted_dir.glob("*.txt"))
        file_paths.extend(converted_txt_files)
        logger.info(f"  转换的PDF文本: {len(converted_txt_files)} 个")
    else:
        logger.warning(f"  转换目录不存在: {converted_dir}")
        logger.warning(f"  提示: 请先运行 convert_pdf_to_txt.py 转换PDF文件")

    logger.info(f"\n总文件数: {len(file_paths)} 个")

    # 显示文件列表
    logger.info("\n文件列表:")
    for i, fp in enumerate(file_paths[:10], 1):  # 只显示前10个
        rel_path = fp.relative_to(knowledge_base_dir)
        logger.info(f"  {i}. {rel_path}")

    if len(file_paths) > 10:
        logger.info(f"  ... 还有 {len(file_paths) - 10} 个文件")

    # 创建RAG Pipeline
    logger.info("\n初始化RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
        chunks_storage_path="data/knowledge_base_with_converted_pdf.json"
    )

    # 开始索引
    logger.info("\n开始处理文档...")
    logger.info("-"*80)

    start_time = time.time()

    try:
        chunk_count = pipeline.index_documents([str(fp) for fp in file_paths])

        elapsed_time = time.time() - start_time

        logger.info("\n" + "="*80)
        logger.info("索引完成！")
        logger.info("="*80)
        logger.info(f"Files processed: {len(file_paths)}")
        logger.info(f"Chunks generated: {chunk_count}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Average speed: {len(file_paths)/elapsed_time:.2f} files/second")

        # 获取统计信息
        stats = pipeline.get_pipeline_stats()
        logger.info(f"\n向量数据库统计:")
        logger.info(f"  集合名称: {stats['milvus_collection']['name']}")
        logger.info(f"  实体总数: {stats['milvus_collection']['num_entities']}")
        logger.info(f"  嵌入模型: {stats['embedding_model']['model_name']}")
        logger.info(f"  设备: GPU (CUDA)")
        logger.info(f"  向量维度: {stats['embedding_model']['dimension']}")

        # 验证GPU使用情况
        import torch
        logger.info(f"\nGPU使用验证:")
        logger.info(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU内存分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"  GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

        logger.info("\n" + "="*80)
        logger.info("所有操作成功完成！")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error(f"\n索引失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = index_with_converted_pdf()
    exit(0 if success else 1)
