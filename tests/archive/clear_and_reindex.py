# -*- coding: utf-8 -*-
"""
Clear Milvus collection and re-index with GPU
"""
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.logger import logger
from pymilvus import utility
import subprocess
import sys
import argparse

def clear_collection():
    """Clear existing collection"""
    logger.info("=" * 80)
    logger.info("Step 1: Clearing existing collection")
    logger.info("=" * 80)

    manager = MilvusManager("config/milvus_config.yaml")

    # Check if collection exists
    if utility.has_collection(manager.collection_name):
        logger.info(f"Found existing collection: {manager.collection_name}")
        logger.info(f"Current entity count: {manager.collection.num_entities}")

        # Drop collection directly
        logger.info(f"Dropping collection: {manager.collection_name}")
        manager.drop_collection()
        logger.info("Collection dropped successfully")
    else:
        logger.info(f"No existing collection found: {manager.collection_name}")

    logger.info("")
    return True

def reindex_with_gpu(mode="all"):
    """Re-index with GPU

    Args:
        mode: "txt" for TXT files only, "all" for all file types
    """
    logger.info("=" * 80)
    logger.info("Step 2: Re-indexing with GPU")
    logger.info("=" * 80)

    # Confirm GPU usage
    logger.info("Configuration:")
    logger.info("  Device: CUDA (GPU)")
    logger.info("  Model: BAAI/bge-m3")
    logger.info("  Knowledge base: 知识库/知识库")

    if mode == "txt":
        logger.info("\nStarting TXT indexing...")
        script = "index_txt_only.py"
    else:
        logger.info("\nStarting all files indexing...")
        script = "index_knowledge_base.py"

    # Run indexing script
    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd="E:\\02 Final Year Project\\RAG Project",
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return False

def verify_indexing():
    """Verify indexing results"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Verifying indexing results")
    logger.info("=" * 80)

    try:
        manager = MilvusManager("config/milvus_config.yaml")
        stats = manager.get_collection_stats()

        logger.info(f"\nCollection name: {stats['name']}")
        logger.info(f"Vector count: {stats['num_entities']:,}")

        if stats['num_entities'] > 0:
            logger.info("\nIndexing completed successfully!")
        else:
            logger.warning("\nNo data in collection")

        return stats['num_entities'] > 0

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def main(mode="all"):
    """Main function"""
    logger.info("\n" + "=" * 80)
    logger.info("GPU Re-indexing Tool")
    logger.info("=" * 80)
    logger.info("\nThis tool will:")
    logger.info("1. Clear existing Milvus collection")
    logger.info("2. Re-index documents with GPU")
    logger.info("3. Verify indexing results")

    # Step 1: Clear collection
    if not clear_collection():
        logger.info("\nProcess terminated")
        return

    # Step 2: Re-index
    if not reindex_with_gpu(mode):
        logger.error("\nIndexing failed")
        return

    # Step 3: Verify results
    verify_indexing()

    logger.info("\n" + "=" * 80)
    logger.info("All operations completed!")
    logger.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear Milvus collection and re-index with GPU")
    parser.add_argument(
        "--mode",
        choices=["txt", "all"],
        default="all",
        help="Indexing mode: 'txt' for TXT files only, 'all' for all file types (default: all)"
    )
    args = parser.parse_args()

    main(mode=args.mode)
