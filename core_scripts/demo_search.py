"""
演示RAG系统搜索功能
Demo RAG system search functionality with proper UTF-8 output
"""
import sys
import io
from rag_project.pipeline import RAGPipeline

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 80)
    print("RAG系统搜索���示 | RAG System Search Demo")
    print("=" * 80)

    # 初始化Pipeline
    print("\n正在初始化RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
    )

    # 测试查询
    queries = [
        "江西省交通投资集团",
        "高速公路建设",
        "安全生产管理"
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"查询: {query}")
        print(f"{'=' * 80}")

        results = pipeline.search(query, top_k=3)

        if not results:
            print("未找到相关结果 | No results found")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n结果 #{i}")
            print(f"相关度: {result['score']:.4f}")
            print(f"来源: {result['metadata'].get('source', 'Unknown')}")
            print(f"文档类型: {result['metadata'].get('doc_type', 'Unknown')}")
            print(f"标题: {result['metadata'].get('title', 'N/A')}")
            print(f"内容预览: {result['text'][:150]}...")

    print("\n" + "=" * 80)
    print("搜索演示完成 | Search demo completed")
    print("=" * 80)

if __name__ == "__main__":
    main()
