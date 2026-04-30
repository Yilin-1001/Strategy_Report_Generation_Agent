"""
索引知识库文档到向量数据库
Index knowledge base documents to vector database
"""
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger
import time

def main():
    # 配置
    knowledge_base_path = "知识库/知识库"
    chunks_output_path = "data/knowledge_base_chunks.json"

    logger.info("=" * 80)
    logger.info("Starting knowledge base indexing")
    logger.info("=" * 80)

    # 收集所有文档文件
    logger.info(f"扫描目录: {knowledge_base_path}")
    file_paths = []

    knowledge_base_dir = Path(knowledge_base_path)
    if not knowledge_base_dir.exists():
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        return

    # 递归查找所有支持的文档类型
    for ext in ['*.txt', '*.pdf', '*.docx', '*.doc']:
        file_paths.extend(knowledge_base_dir.rglob(ext))

    logger.info(f"找到 {len(file_paths)} 个文档文件")

    # 按类型统计
    file_types = {}
    for fp in file_paths:
        ext = fp.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    logger.info("文件类型分布:")
    for ext, count in sorted(file_types.items()):
        logger.info(f"  {ext}: {count} 个文件")

    # 创建RAG Pipeline
    logger.info("\n初始化RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
        chunks_storage_path=chunks_output_path
    )

    # 开始索引
    logger.info("\n开始处理文档...")
    logger.info("-" * 80)

    start_time = time.time()

    try:
        chunk_count = pipeline.index_documents([str(fp) for fp in file_paths])

        elapsed_time = time.time() - start_time

        logger.info("-" * 80)
        logger.info(f"\n[Indexing Complete]")
        logger.info(f"Files processed: {len(file_paths)}")
        logger.info(f"Chunks generated: {chunk_count}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Average speed: {len(file_paths)/elapsed_time:.2f} files/second")

        # 获取pipeline统计信息
        stats = pipeline.get_pipeline_stats()
        logger.info(f"\n向量数据库统计:")
        logger.info(f"  集合名称: {stats['milvus_collection']['name']}")
        logger.info(f"  实体总数: {stats['milvus_collection']['num_entities']}")
        logger.info(f"  嵌入模型: {stats['embedding_model']['model_name']}")
        logger.info(f"  向量维度: {stats['embedding_model']['dimension']}")

        # 查看分块摘要
        from rag_project.data_loader.chunk_storage import ChunkStorage
        storage = ChunkStorage()
        summary = storage.get_chunks_summary(chunks_output_path)

        logger.info(f"\n分块摘要:")
        logger.info(f"  总块数: {summary['total_chunks']}")
        logger.info(f"  总字符数: {summary['total_characters']:,}")
        logger.info(f"  平均块长度: {summary['avg_chunk_length']:.2f} 字符")
        logger.info(f"  文档类型分布: {summary['doc_types_distribution']}")

        logger.info(f"\n分块文件已保存到: {chunks_output_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n[Indexing Failed]: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("\n[All operations completed]")

if __name__ == "__main__":
    main()
