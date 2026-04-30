"""
只处理TXT文件并测试GPU Embedding
"""
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger
import time

def index_txt_only():
    """只索引TXT文件，测试GPU Embedding"""
    logger.info("="*80)
    logger.info("TXT文件索引 - GPU Embedding测试")
    logger.info("="*80)

    knowledge_base_path = "知识库/知识库"
    knowledge_base_dir = Path(knowledge_base_path)

    if not knowledge_base_dir.exists():
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        return False

    # 只收集TXT文件
    logger.info("\n扫描TXT文件...")
    txt_files = list(knowledge_base_dir.rglob("*.txt"))

    logger.info(f"找到 {len(txt_files)} 个TXT文件")

    # 创建RAG Pipeline
    logger.info("\n初始化RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
        chunks_storage_path="data/txt_only.json"
    )

    # 开始索引
    logger.info("\n开始处理TXT文件...")
    logger.info("-"*80)

    start_time = time.time()

    try:
        chunk_count = pipeline.index_documents([str(fp) for fp in txt_files])

        elapsed_time = time.time() - start_time

        logger.info("\n" + "="*80)
        logger.info("索引完成！")
        logger.info("="*80)
        logger.info(f"Files processed: {len(txt_files)}")
        logger.info(f"Chunks generated: {chunk_count}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")

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
        logger.info("✓ 所有操作成功完成！")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error(f"\n索引失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = index_txt_only()
    exit(0 if success else 1)
