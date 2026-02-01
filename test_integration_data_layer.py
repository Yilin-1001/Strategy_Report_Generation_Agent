"""
集成测试：测试数据层模块的协同工作
"""
import tempfile
import os
from pathlib import Path
from langchain_core.documents import Document

# 测试导入
from rag_project.utils.config_loader import load_config, get_chunking_config
from rag_project.data_loader.document_type_detector import detect_doc_type, get_loader_for_file
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from rag_project.data_loader.metadata_extractor import MetadataExtractor
from rag_project.data_loader.chunk_storage import ChunkStorage

def test_config_loader():
    """测试1: ��置加载器"""
    print("\n" + "="*60)
    print("测试1: 配置加载器")
    print("="*60)

    # 加载分块配置
    chunking_config = load_config("config/chunking_config.yaml")
    print(f"✓ 成功加载分块配置")
    print(f"  - 新闻分块大小: {chunking_config['chunking']['news']['chunk_size']}")
    print(f"  - PDF分块大小: {chunking_config['chunking']['pdf']['chunk_size']}")

    # 加载Milvus配置
    milvus_config = load_config("config/milvus_config.yaml")
    print(f"✓ 成功加载Milvus配置")
    print(f"  - 集合名称: {milvus_config['milvus']['collection']['name']}")
    print(f"  - 向量维度: {milvus_config['milvus']['collection']['dimension']}")

    # 获取特定文档类型配置
    news_config = get_chunking_config('news')
    print(f"✓ 成功获取新闻配置: chunk_size={news_config['chunk_size']}")

def test_document_type_detection():
    """测试2: 文档类型检测"""
    print("\n" + "="*60)
    print("测试2: 文档类型检测")
    print("="*60)

    test_files = [
        ("news.txt", "news"),
        ("policy.pdf", "pdf"),
        ("regulation.docx", "regulation"),
        ("regulation.doc", "regulation"),
        ("unknown.xyz", "default"),
    ]

    for filename, expected_type in test_files:
        detected_type = detect_doc_type(filename)
        status = "✓" if detected_type == expected_type else "✗"
        print(f"{status} {filename} -> {detected_type} (期望: {expected_type})")

def test_document_loading_and_chunking():
    """测试3: 文档加载和分块"""
    print("\n" + "="*60)
    print("测试3: 文档加载和分块")
    print("="*60)

    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        test_content = """这是第一段测试文本。用来测试新闻文档的分块功能。

这是第二段测试文本。江西省交通建设取得新的进展。

这是第三段测试文本。全省高速公路网络不断完善。这是第四段测试文本，用来测试更长的文本分块效果。""" * 5
        f.write(test_content)
        temp_file = f.name

    try:
        # 检测文档类型
        doc_type = detect_doc_type(temp_file)
        print(f"✓ 检测到文档类型: {doc_type}")

        # 加载文档
        loader = get_loader_for_file(temp_file)
        documents = loader.load()
        print(f"✓ 成功加载文档: {len(documents)} 个文档对象")

        # 创建分块器
        chunker = ConfigurableChunker()

        # 分块
        chunks = chunker.split_documents(documents, doc_type=doc_type)
        print(f"✓ 成功分块: {len(chunks)} 个文本块")

        # 验证元数据
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
            print(f"\n  块 {i+1}:")
            print(f"    - 长度: {len(chunk.page_content)} 字符")
            print(f"    - 文档类型: {chunk.metadata.get('doc_type')}")
            print(f"    - 预览: {chunk.page_content[:50]}...")

    finally:
        # 清理临时文件
        os.unlink(temp_file)

def test_metadata_extraction():
    """测试4: 元数据提取"""
    print("\n" + "="*60)
    print("测试4: 元数据提取")
    print("="*60)

    # 测试核心元数据提取
    doc = Document(page_content="测试内容", metadata={"title": "测试标题"})
    metadata = MetadataExtractor.extract_core_metadata(doc, "news", "test.txt")

    print("✓ 提取的核心元数据:")
    print(f"  - 文档ID: {metadata['doc_id']}")
    print(f"  - 文档类型: {metadata['doc_type']}")
    print(f"  - 来源: {metadata['source']}")
    print(f"  - 标题: {metadata['title']}")

    # 测试文件名元数据提取
    filename = "全省网约车平台上线_2025-02-20 16_21.txt"
    filename_metadata = MetadataExtractor.extract_from_filename(filename)

    print("\n✓ 从文件名提取的元数据:")
    print(f"  - 文件名: {filename}")
    print(f"  - 标题: {filename_metadata.get('title')}")
    print(f"  - 发布日期: {filename_metadata.get('publish_date')}")

def test_chunk_storage():
    """测试5: 块存储"""
    print("\n" + "="*60)
    print("测试5: 块存储管理")
    print("="*60)

    # 创建测试文档
    test_docs = [
        Document(
            page_content="这是第一个测试文本块的内容。",
            metadata={"doc_id": "1", "doc_type": "news", "source": "test1.txt"}
        ),
        Document(
            page_content="这是第二个测试文本块的内容。包含更多详细信息。",
            metadata={"doc_id": "2", "doc_type": "pdf", "source": "test2.pdf"}
        ),
    ]

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        # 保存块
        storage = ChunkStorage()
        storage.save_chunks_to_json(test_docs, temp_file)
        print(f"✓ 成功保存 {len(test_docs)} 个文本块到 {temp_file}")

        # 获取摘要
        summary = storage.get_chunks_summary(temp_file)
        print("✓ 块摘要:")
        print(f"  - 总块数: {summary['total_chunks']}")
        print(f"  - 总字符数: {summary['total_characters']}")
        print(f"  - 平均块长度: {summary['avg_chunk_length']:.2f}")
        print(f"  - 文档类型分布: {summary['doc_types_distribution']}")

        # 加载块
        loaded_docs = storage.load_chunks_from_json(temp_file)
        print(f"✓ 成功加载 {len(loaded_docs)} 个文本块")

        # 验证数据完整性
        assert len(loaded_docs) == len(test_docs), "加载的文档数量不匹配"
        assert loaded_docs[0].page_content == test_docs[0].page_content, "内容不匹配"
        assert loaded_docs[0].metadata['doc_type'] == test_docs[0].metadata['doc_type'], "元数据不匹配"
        print("✓ 数据完整性验证通过")

    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_end_to_end_pipeline():
    """测试6: 端到端数据层流程"""
    print("\n" + "="*60)
    print("测试6: 端到端数据层流程")
    print("="*60)

    # 创建测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        test_content = """
江西省交通运输厅发布最新通知。
全省网约车平台正式上线运营。
这将大大提升交通服务质量。
        """ * 10  # 重复以获得足够的文本进行分块
        f.write(test_content)
        temp_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        chunks_file = f.name

    try:
        print("步骤1: 检测文档类型")
        doc_type = detect_doc_type(temp_file)
        print(f"  ✓ 文档类型: {doc_type}")

        print("\n步骤2: 加载文档")
        loader = get_loader_for_file(temp_file)
        documents = loader.load()
        print(f"  ✓ 加载了 {len(documents)} 个文档")

        print("\n步骤3: 提取元数据")
        for doc in documents:
            metadata = MetadataExtractor.extract_core_metadata(doc, doc_type, Path(temp_file).name)
            doc.metadata.update(metadata)
        print(f"  ✓ 元数据提取完成")

        print("\n步骤4: 分块")
        chunker = ConfigurableChunker()
        chunks = chunker.split_documents(documents, doc_type=doc_type)
        print(f"  ✓ 生成了 {len(chunks)} 个文本块")

        print("\n步骤5: 保存块到JSON")
        storage = ChunkStorage()
        storage.save_chunks_to_json(chunks, chunks_file)
        print(f"  ✓ 保存到 {chunks_file}")

        print("\n步骤6: 验证保存的块")
        summary = storage.get_chunks_summary(chunks_file)
        print(f"  ✓ 总块数: {summary['total_chunks']}")
        print(f"  ✓ 平均块长度: {summary['avg_chunk_length']:.2f} 字符")

        print("\n步骤7: 重新加载块")
        loaded_chunks = storage.load_chunks_from_json(chunks_file)
        print(f"  ✓ 成功加载 {len(loaded_chunks)} 个块")

        print("\n" + "="*60)
        print("✓ 端到端流程测试成功完成！")
        print("="*60)

    finally:
        # 清理临时文件
        os.unlink(temp_file)
        if os.path.exists(chunks_file):
            os.unlink(chunks_file)

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*18 + "数据层集成测试" + " "*18 + "█")
    print("█"*60)

    try:
        test_config_loader()
        test_document_type_detection()
        test_document_loading_and_chunking()
        test_metadata_extraction()
        test_chunk_storage()
        test_end_to_end_pipeline()

        print("\n" + "█"*60)
        print("█" + " "*15 + "所有测试通过！✓" + " "*15 + "█")
        print("█"*60 + "\n")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
