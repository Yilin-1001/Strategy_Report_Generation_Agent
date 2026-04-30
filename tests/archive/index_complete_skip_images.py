"""
完整的知识库索引 - PDF跳过图片，使用GPU加速embedding
Complete knowledge base indexing - Skip images in PDF, use GPU acceleration
"""
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger
import time

def index_all_documents():
    """索引所有文档（PDF跳过图片）"""
    logger.info("="*80)
    logger.info("COMPLETE KNOWLEDGE BASE INDEXING")
    logger.info("PDF: Text only (skip images)")
    logger.info("Embedding: GPU acceleration")
    logger.info("="*80)

    knowledge_base_path = "知识库/知识库"
    knowledge_base_dir = Path(knowledge_base_path)

    if not knowledge_base_dir.exists():
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        return False

    # 收集所有���档文件
    logger.info("\n扫描文档...")
    file_paths = []

    for ext in ['*.txt', '*.pdf', '*.docx', '*.doc']:
        file_paths.extend(knowledge_base_dir.rglob(ext))

    logger.info(f"找到 {len(file_paths)} 个文档")

    # 按类型统计
    file_types = {}
    for fp in file_paths:
        ext = fp.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    logger.info("\n��件类型分布:")
    for ext, count in sorted(file_types.items()):
        logger.info(f"  {ext}: {count} 个文件")

    # 创建RAG Pipeline
    logger.info("\n初始化RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
        chunks_storage_path="data/knowledge_base_complete.json"
    )

    # 开始索引
    logger.info("\n开始处理文档...")
    logger.info("提示: PDF文件将只提取文本，跳过图片")
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

        logger.info("\n" + "="*80)
        logger.info("✓ 所有操作成功完成！")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error(f"\n索引失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = index_all_documents()
    exit(0 if success else 1)
