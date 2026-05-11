"""
RAG Project - Main Entry Point
主入口文件
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="RAG Document Processing Pipeline")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('paths', nargs='+', help='File or directory paths')
    index_parser.add_argument('--chunks-output', default='data/chunks.json', help='Chunks output file')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--doc-type', action='append', help='Filter by document type')

    args = parser.parse_args()

    if args.command == 'index':
        # Collect file paths
        file_paths = []
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_file():
                file_paths.append(str(path))
            elif path.is_dir():
                file_paths.extend([str(p) for p in path.rglob('*') if p.is_file()])

        if not file_paths:
            logger.error("No files found to index")
            return

        logger.info(f"Found {len(file_paths)} files to index")

        # Create pipeline and index
        pipeline = RAGPipeline(chunks_storage_path=args.chunks_output)
        count = pipeline.index_documents(file_paths)

        logger.info(f"Indexing complete: {count} chunks")

    elif args.command == 'search':
        # Create pipeline
        pipeline = RAGPipeline()

        # Build filters
        filters = {}
        if args.doc_type:
            filters['doc_type'] = args.doc_type

        # Search
        results = pipeline.search(args.query, top_k=args.top_k, filters=filters)

        # Display results
        print(f"\n搜索结果 (query: {args.query})")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] 相关度: {result['score']:.4f}")
            print(f"来源: {result['metadata']['source']}")
            print(f"类型: {result['metadata']['doc_type']}")
            if result['metadata']['page_number']:
                print(f"页码: {result['metadata']['page_number']}")
            print(f"内容: {result['text'][:200]}...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
