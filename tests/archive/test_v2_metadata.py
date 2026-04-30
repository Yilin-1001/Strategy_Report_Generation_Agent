"""
测试新的Metadata构建器（V2）
"""
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from pathlib import Path
import json


def test_v2_metadata():
    """测试V2 metadata生成"""

    print("="*80)
    print("测试V2 Metadata生成器")
    print("="*80)

    # 测试文件1: 新闻文件（文件名包含日期）
    test_news_path = "知识库/江西省交通运输厅新闻动态/厅领导春节前走访慰问离退休干部_2025-01-26 10_50.txt"

    if not Path(test_news_path).exists():
        print(f"Warning: Test file not found: {test_news_path}")
        print("Using sample content instead...")

        # 使用示例内容
        sample_content = """2025年1月26日
厅领导春节前走访慰问离退休干部
发布时间：2025-01-26 10:50

厅领导在春节前夕走访慰问离退休老干部，传递组织温暖...
"""
        test_news_path = "sample_news.txt"
        with open(test_news_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)

    # 读取测试文档
    print(f"\n测试文档: {test_news_path}")
    with open(test_news_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 创建Document
    doc = Document(page_content=content, metadata={'source': test_news_path})

    # 初始化chunker（使用V2 metadata）
    print("\n初始化Chunker (V2 metadata)...")
    chunker = ConfigurableChunker(
        config_path="config/chunking_config.yaml",
        base_dir="知识库/知识库",
        use_v2_metadata=True
    )

    # 执行chunking
    print("开始chunking...")
    chunks = chunker.split_documents([doc], doc_type='news')

    print(f"\n生成 {len(chunks)} 个chunks")
    print("="*80)

    # 显示第一个chunk的完整metadata
    if chunks:
        chunk = chunks[0]
        print("\n第一个chunk的metadata:")
        print("-"*80)
        for key, value in chunk.metadata.items():
            print(f"  {key}: {value}")

    # 保存到JSON文件
    output_file = "data/v2_metadata_test.json"
    Path("data").mkdir(exist_ok=True)

    chunks_data = []
    for i, chunk in enumerate(chunks, 1):
        chunks_data.append({
            "metadata": chunk.metadata,
            "content_preview": chunk.page_content[:100]
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f"\n测试结果保存到: {output_file}")
    print("="*80)

    # 验证metadata字段
    print("\n验证结果:")
    print("-"*80)

    required_fields = [
        'chunk_id', 'document_id', 'file_name', 'page_number',
        'section_title', 'chunk_index', 'created_at', 'tags'
    ]

    chunk = chunks[0] if chunks else None
    if chunk:
        for field in required_fields:
            if field in chunk.metadata:
                value = chunk.metadata[field]
                print(f"  ✓ {field}: {value}")
            else:
                print(f"  ✗ {field}: MISSING")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)

    return True


if __name__ == "__main__":
    success = test_v2_metadata()
    exit(0 if success else 1)
