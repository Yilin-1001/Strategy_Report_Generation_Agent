"""
清除Milvus中的数据并重新使用GPU索引
Clear Milvus data and re-index with GPU acceleration
"""
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.logger import logger
from pathlib import Path
from rag_project.pipeline import RAGPipeline
import time

def clear_and_reindex():
    """Delete existing collection and re-index all documents"""
    logger.info("="*80)
    logger.info("CLEAR AND RE-INDEX KNOWLEDGE BASE")
    logger.info("="*80)

    # Step 1: Clear existing collection
    logger.info("\n[Step 1/4] Clearing existing Milvus collection...")
    logger.info("-"*80)

    try:
        milvus_manager = MilvusManager()

        # Get stats before dropping
        stats_before = milvus_manager.get_collection_stats()
        logger.info(f"Current collection: {stats_before['name']}")
        logger.info(f"Current entities: {stats_before['num_entities']}")

        # Drop the collection
        logger.info("\nDropping collection...")
        milvus_manager.drop_collection()
        logger.info("Collection dropped successfully!")

    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        return False

    # Step 2: Verify collection is cleared
    logger.info("\n[Step 2/4] Verifying collection is cleared...")
    logger.info("-"*80)

    try:
        from pymilvus import utility, connections

        # Reconnect to check
        connections.connect(alias="default", host="localhost", port="19530")

        if not utility.has_collection("enterprise_docs"):
            logger.info("✓ Collection successfully removed")
        else:
            logger.error("✗ Collection still exists!")
            return False

    except Exception as e:
        logger.error(f"Error verifying: {e}")
        return False

    # Step 3: Scan knowledge base
    logger.info("\n[Step 3/4] Scanning knowledge base...")
    logger.info("-"*80)

    knowledge_base_path = "知识库/知识库"
    knowledge_base_dir = Path(knowledge_base_path)

    if not knowledge_base_dir.exists():
        logger.error(f"知识库目录不存在: {knowledge_base_path}")
        return False

    # Collect all document files
    file_paths = []
    for ext in ['*.txt', '*.pdf', '*.docx', '*.doc']:
        file_paths.extend(knowledge_base_dir.rglob(ext))

    logger.info(f"Found {len(file_paths)} documents")

    # Count by type
    file_types = {}
    for fp in file_paths:
        ext = fp.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    logger.info("File type distribution:")
    for ext, count in sorted(file_types.items()):
        logger.info(f"  {ext}: {count} files")

    # Step 4: Re-index with GPU
    logger.info("\n[Step 4/4] Re-indexing with GPU acceleration...")
    logger.info("-"*80)

    try:
        # Create RAG pipeline
        pipeline = RAGPipeline(
            chunking_config_path="config/chunking_config.yaml",
            milvus_config_path="config/milvus_config.yaml",
            chunks_storage_path="data/knowledge_base_chunks.json"
        )

        # Index all documents
        logger.info("\nStarting document processing...")
        start_time = time.time()

        chunk_count = pipeline.index_documents([str(fp) for fp in file_paths])

        elapsed_time = time.time() - start_time

        logger.info("\n" + "="*80)
        logger.info("RE-INDEXING COMPLETE")
        logger.info("="*80)
        logger.info(f"Files processed: {len(file_paths)}")
        logger.info(f"Chunks generated: {chunk_count}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Average speed: {len(file_paths)/elapsed_time:.2f} files/second")

        # Get final statistics
        stats = pipeline.get_pipeline_stats()
        logger.info(f"\nFinal vector database stats:")
        logger.info(f"  Collection: {stats['milvus_collection']['name']}")
        logger.info(f"  Total entities: {stats['milvus_collection']['num_entities']}")
        logger.info(f"  Embedding model: {stats['embedding_model']['model_name']}")
        logger.info(f"  Device: GPU (CUDA)")
        logger.info(f"  Vector dimension: {stats['embedding_model']['dimension']}")

        logger.info("\n" + "="*80)
        logger.info("✓ ALL OPERATIONS COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error(f"\nRe-indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = clear_and_reindex()
    exit(0 if success else 1)
