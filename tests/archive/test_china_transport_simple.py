"""
测试"中国交通运输2021"文档的chunking和页码保留（简化版）
"""
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from pathlib import Path


def test_china_transport_2021_simple():
    """测试中国交通运输2021文档（简化版）"""

    # 文档路径
    doc_path = "知识库/知识库/相关研报/中国交通运输2021.txt"

    # 检查文件是否存在
    if not Path(doc_path).exists():
        print(f"Error: File not found: {doc_path}")
        return False

    # 读取文档
    print(f"Reading document: {doc_path}")
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Document size: {len(text):,} characters")
    print(f"Lines: {text.count(chr(10))}")

    # 检查分页标记
    marker_count = text.count("第 ")
    print(f"Page markers found: {marker_count}")

    # 创建Document对象
    doc = Document(
        page_content=text,
        metadata={
            "source": "中国交通运输2021.txt",
            "doc_type": "pdf"
        }
    )

    # 初始化chunker
    print("\nInitializing chunker...")
    chunker = ConfigurableChunker("config/chunking_config.yaml")

    # 执行chunking
    print("Starting chunking...")
    chunks = chunker.split_documents([doc], doc_type='pdf')

    print(f"\nTotal chunks generated: {len(chunks)}")
    print("="*80)

    # 统计页码分布
    page_distribution = {}
    for chunk in chunks:
        page_num = chunk.metadata.get('page_number', 'Unknown')
        page_distribution[page_num] = page_distribution.get(page_num, 0) + 1

    # 显示页码分布
    print(f"\nPages covered: {len(page_distribution)}")
    print(f"Expected pages: 305")
    print(f"Coverage: {len(page_distribution)/305*100:.1f}%")

    # 显示每页的chunk数量统计
    chunk_counts = {}
    for count in page_distribution.values():
        chunk_counts[count] = chunk_counts.get(count, 0) + 1

    print("\nChunks per page distribution:")
    for chunks_per_page in sorted(chunk_counts.keys())[:10]:
        num_pages = chunk_counts[chunks_per_page]
        print(f"  {chunks_per_page} chunks/page: {num_pages} pages")

    # 显示前10个chunks
    print("\n" + "="*80)
    print("First 10 chunks:")
    print("="*80)

    for i, chunk in enumerate(chunks[:10], 1):
        page_num = chunk.metadata.get('page_number', 'N/A')
        char_count = len(chunk.page_content)
        preview = chunk.page_content[:100].replace('\n', ' ').strip()

        print(f"\nChunk {i}:")
        print(f"  Page: {page_num}")
        print(f"  Size: {char_count} chars")
        print(f"  Preview: {preview}...")

    # 验证
    all_have_page_numbers = all('page_number' in c.metadata for c in chunks)
    none_have_markers = not any("第 " in c.page_content and "页 / 共" in c.page_content for c in chunks)
    all_pages_covered = len(page_distribution) == 305

    print("\n" + "="*80)
    print("Verification:")
    print("="*80)
    print(f"All chunks have page_number: {all_have_page_numbers}")
    print(f"No chunks contain page markers: {none_have_markers}")
    print(f"All 305 pages covered: {all_pages_covered}")

    # 显示特定页码的chunks
    print("\n" + "="*80)
    print("Sample chunks from specific pages:")
    print("="*80)

    for test_page in [1, 50, 100, 200, 305]:
        page_chunks = [c for c in chunks if c.metadata.get('page_number') == test_page]
        if page_chunks:
            print(f"\nPage {test_page} ({len(page_chunks)} chunks):")
            preview = page_chunks[0].page_content[:100].replace('\n', ' ')
            print(f"  First chunk: {preview}...")
        else:
            print(f"\nPage {test_page}: No chunks found")

    # 最终结果
    success = all_have_page_numbers and none_have_markers and all_pages_covered

    print("\n" + "="*80)
    if success:
        print("SUCCESS: Page number preservation works correctly!")
    else:
        print("FAILURE: Some checks failed!")
    print("="*80)

    return success


if __name__ == "__main__":
    success = test_china_transport_2021_simple()
    exit(0 if success else 1)
