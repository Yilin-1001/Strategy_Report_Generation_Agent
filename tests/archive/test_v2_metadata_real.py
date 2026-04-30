"""
测试V2 Metadata - 使用真实文档
"""
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from pathlib import Path
import json


def test_with_real_document():
    """使用真实文档测试V2 metadata"""

    doc_path = "知识库/知识库/相关研报/中国交通运输2021.txt"

    if not Path(doc_path).exists():
        return False

    # 读取文档
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 创建Document
    doc = Document(page_content=content, metadata={'source': doc_path})

    # 初始化chunker（V2 metadata）
    chunker = ConfigurableChunker(
        config_path="config/chunking_config.yaml",
        base_dir="知识库/知识库",
        use_v2_metadata=True
    )

    # 执行chunking
    chunks = chunker.split_documents([doc], doc_type='pdf')

    # 准备结果
    result = {
        "total_chunks": len(chunks),
        "document_id": chunks[0].metadata.get("document_id") if chunks else None,
        "sample_chunks": []
    }

    # 保存前5个chunks的metadata
    for i, chunk in enumerate(chunks[:5], 1):
        result["sample_chunks"].append({
            "chunk_number": i,
            "metadata": chunk.metadata,
            "content_preview": chunk.page_content[:100]
        })

    # 保存到文件
    output_file = "data/china_transport_v2_metadata_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 统计tags
    all_tags = set()
    for chunk in chunks:
        tags = chunk.metadata.get('tags', [])
        if tags:
            all_tags.update(tags)

    # 打印摘要
    with open("data/metadata_test_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"V2 Metadata Test Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Document: {doc_path}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Document ID: {result['document_id']}\n\n")
        f.write(f"All unique tags found: {list(all_tags)}\n\n")
        f.write("="*80 + "\n")
        f.write("Sample chunks metadata:\n")
        f.write("="*80 + "\n\n")

        for i, sample in enumerate(result['sample_chunks'], 1):
            meta = sample['metadata']
            f.write(f"Chunk {i}:\n")
            f.write(f"  chunk_id: {meta.get('chunk_id')}\n")
            f.write(f"  document_id: {meta.get('document_id')}\n")
            f.write(f"  file_name: {meta.get('file_name')}\n")
            f.write(f"  page_number: {meta.get('page_number')}\n")
            f.write(f"  section_title: {meta.get('section_title')}\n")
            f.write(f"  chunk_index: {meta.get('chunk_index')}\n")
            f.write(f"  created_at: {meta.get('created_at')}\n")
            f.write(f"  tags: {meta.get('tags')}\n")
            f.write("\n")

    print(f"Test completed!")
    print(f"Results saved to: {output_file}")
    print(f"Summary saved to: data/metadata_test_summary.txt")

    return True


if __name__ == "__main__":
    success = test_with_real_document()
    exit(0 if success else 1)
