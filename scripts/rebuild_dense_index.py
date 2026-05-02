"""
Rebuild dense index — re-index all source documents into dense collection.

Uses RAGPipeline with SiliconFlow API for embeddings. Indexes all .txt files
from the knowledge base into the "enterprise_docs" collection.

Usage:
    python scripts/rebuild_dense_index.py
    python scripts/rebuild_dense_index.py --knowledge-base "知识库/知识库"
"""

import os
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import connections, utility
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger


def collect_files(knowledge_base_path: str) -> List[str]:
    """Collect all indexable files from knowledge base."""
    kb = Path(knowledge_base_path)
    if not kb.exists():
        print(f"ERROR: Knowledge base path not found: {knowledge_base_path}")
        return []

    # Collect .txt files (primary format)
    files = sorted([str(f) for f in kb.rglob("*.txt")])
    print(f"Found {len(files)} .txt files in {knowledge_base_path}")
    return files


def rebuild_dense_index(
    knowledge_base_path: str = "知识库/知识库",
    collection_name: str = "enterprise_docs",
):
    """Re-index all documents into dense collection."""

    # Connect to Milvus to drop existing collection
    connections.connect(alias="default", host="localhost", port=19530)
    if utility.has_collection(collection_name):
        print(f"Dropping existing collection '{collection_name}'...")
        utility.drop_collection(collection_name)
        print("Dropped.")
    connections.disconnect("default")

    # Collect files
    files = collect_files(knowledge_base_path)
    if not files:
        print("No files to index!")
        return

    print(f"\n{'='*60}")
    print(f"Starting dense indexing")
    print(f"  Files: {len(files)}")
    print(f"  Collection: {collection_name}")
    print(f"  Embedding: SiliconFlow API (BGE-M3)")
    print(f"{'='*60}\n")

    t_start = time.time()

    # Create dense pipeline and index
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
        knowledge_base_path=knowledge_base_path,
    )

    total_chunks = pipeline.index_documents(files)

    elapsed = time.time() - t_start

    # Final stats
    print(f"\n{'='*60}")
    print(f"Dense indexing complete!")
    print(f"  Files processed: {len(files)}")
    print(f"  Total chunks:    {total_chunks}")
    print(f"  Collection:      {collection_name}")
    print(f"  Time:            {elapsed:.1f}s")

    # Verify
    stats = pipeline.get_pipeline_stats()
    milvus_stats = stats["milvus_collection"]
    print(f"\nCollection stats:")
    print(f"  Name:     {milvus_stats['name']}")
    print(f"  Entities: {milvus_stats['num_entities']}")
    print(f"  Search:   dense_vector")
    print(f"  Index:    HNSW (IP)")
    print(f"\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild dense index from source files")
    parser.add_argument("--knowledge-base", default="知识库/知识库",
                        help="Knowledge base directory path")
    parser.add_argument("--collection", default="enterprise_docs",
                        help="Dense collection name")

    args = parser.parse_args()
    rebuild_dense_index(args.knowledge_base, args.collection)
