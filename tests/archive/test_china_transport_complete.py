"""
测试中国交通运输2021文档的完整chunking和metadata
"""
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from pathlib import Path
import json
from datetime import datetime


def test_china_transport_complete():
    """完整测试中国交通运输2021文档"""

    doc_path = "知识库/知识库/相关研报/中国交通运输2021.txt"

    if not Path(doc_path).exists():
        print(f"Error: File not found: {doc_path}")
        return False

    # 读取文档
    print(f"Reading document: {doc_path}")
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Document size: {len(content):,} characters")
    print(f"Lines: {content.count(chr(10))}")

    # 创建Document
    doc = Document(page_content=content, metadata={'source': doc_path})

    # 初始化chunker（V2 metadata）
    print("\nInitializing chunker with V2 metadata...")
    chunker = ConfigurableChunker(
        config_path="config/chunking_config.yaml",
        base_dir="知识库/知识库",
        use_v2_metadata=True
    )

    # 执行chunking
    print("Starting chunking...")
    chunks = chunker.split_documents([doc], doc_type='pdf')

    print(f"\nTotal chunks: {len(chunks)}")
    print("="*80)

    # 准备输出数据
    output_data = {
        "document_info": {
            "file_path": doc_path,
            "size": len(content),
            "total_chunks": len(chunks)
        },
        "metadata_summary": {
            "document_id": chunks[0].metadata.get('document_id') if chunks else None,
            "created_at": chunks[0].metadata.get('created_at') if chunks else None,
            "tags": chunks[0].metadata.get('tags') if chunks else [],
            "page_range": (chunks[0].metadata.get('page_number', 1),
                          chunks[-1].metadata.get('page_number', 1))
        },
        "chunks": []
    }

    # 收集所有chunks的信息
    for i, chunk in enumerate(chunks, 1):
        chunk_info = {
            "chunk_number": i,
            "metadata": chunk.metadata,
            "char_count": len(chunk.page_content),
            "content": chunk.page_content
        }
        output_data["chunks"].append(chunk_info)

    # 保存完整JSON
    json_file = "data/china_transport_2021_complete.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nComplete chunks saved to: {json_file}")

    # 生成可读的文本报告
    report_file = "data/china_transport_2021_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("中国交通运输2021 - Chunking完整报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"文档信息:\n")
        f.write(f"  文件路径: {doc_path}\n")
        f.write(f"  文档大小: {len(content):,} 字符\n")
        f.write(f"  生成Chunks: {len(chunks)} 个\n\n")

        f.write(f"Metadata摘要:\n")
        f.write(f"  Document ID: {output_data['metadata_summary']['document_id']}\n")
        f.write(f"  Created At: {output_data['metadata_summary']['created_at']}\n")
        f.write(f"  Tags: {output_data['metadata_summary']['tags']}\n")
        f.write(f"  Page Range: 第 {output_data['metadata_summary']['page_range'][0]} "
                 f"至 第 {output_data['metadata_summary']['page_range'][1]} 页\n\n")

        # 显示前20个chunks
        f.write("="*80 + "\n")
        f.write(f"前20个Chunks详情\n")
        f.write("="*80 + "\n\n")

        for i in range(min(20, len(chunks))):
            chunk = output_data["chunks"][i]
            meta = chunk["metadata"]

            f.write("-"*80 + "\n")
            f.write(f"Chunk #{chunk['chunk_number']}\n")
            f.write("-"*80 + "\n")
            f.write(f"  chunk_id      : {meta.get('chunk_id')}\n")
            f.write(f"  document_id   : {meta.get('document_id')}\n")
            f.write(f"  file_name     : {meta.get('file_name')}\n")
            f.write(f"  page_number   : {meta.get('page_number')}\n")
            f.write(f"  section_title : {meta.get('section_title')}\n")
            f.write(f"  chunk_index   : {meta.get('chunk_index')}\n")
            f.write(f"  created_at    : {meta.get('created_at')}\n")
            f.write(f"  tags          : {meta.get('tags')}\n")
            f.write(f"  char_count    : {chunk['char_count']}\n")
            f.write(f"\n  内容预览:\n")
            f.write("  " + chunk["content"][:200].replace("\n", "\\n") + "...\n")
            f.write("\n")

        if len(chunks) > 20:
            f.write("="*80 + "\n")
            f.write(f"... 还有 {len(chunks) - 20} 个chunks\n")
            f.write(f"完整内容请查看: {json_file}\n")

        # 统计分析
        section_titles = {}
        for chunk in chunks:
            section = chunk.metadata.get('section_title')
            if section:
                section_titles[section] = section_titles.get(section, 0) + 1

        f.write("\n" + "="*80 + "\n")
        f.write("章节标题统计\n")
        f.write("="*80 + "\n")

        if section_titles:
            for section, count in sorted(section_titles.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {section}: {count} 个chunks\n")
        else:
            f.write("  (未识别到章节标题)\n")

        # 按页码分布统计
        page_dist = {}
        for chunk in chunks:
            page_num = chunk.metadata.get('page_number', 0)
            page_dist[page_num] = page_dist.get(page_num, 0) + 1

        f.write("\n页码分布 (前30页):\n")
        f.write("-"*80 + "\n")
        for page_num in sorted(page_dist.keys())[:30]:
            count = page_dist[page_num]
            f.write(f"  第 {page_num} 页: {count} 个chunks\n")

    print(f"\n报告已生成: {report_file}")
    print(f"完整JSON数据: {json_file}")
    print("\n" + "="*80)
    print("✓ 测试完成！请查看以下文件:")
    print(f"  1. {report_file}")
    print(f"  2. {json_file}")
    print("="*80)

    return True


if __name__ == "__main__":
    success = test_china_transport_complete()
    exit(0 if success else 1)
