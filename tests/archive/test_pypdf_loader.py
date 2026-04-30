"""
测试PyPDF加载器功能
"""
from pathlib import Path
from rag_project.data_loader.pypdf_loader import PyPDFLoader, convert_pdf_with_pypdf
from rag_project.utils.logger import logger


def test_pypdf_loader():
    """测试PyPDF加载器"""
    print("="*80)
    print("PyPDF加载器功能测试")
    print("="*80)

    # 查找测试PDF文件
    kb_path = Path("知识库/知识库")
    pdf_files = list(kb_path.rglob("*.pdf"))

    if not pdf_files:
        print("\n未找到PDF文件进行测试")
        return

    # 使用第一个PDF进行测试
    test_pdf = pdf_files[0]
    print(f"\n测试PDF: {test_pdf.name}")
    print(f"文件大小: {test_pdf.stat().st_size / 1024:.2f} KB")

    try:
        # 测试1: 基本转换
        print("\n" + "-"*80)
        print("测试1: 基本转换功能")
        print("-"*80)

        result = convert_pdf_with_pypdf(
            str(test_pdf),
            output_path=f"test_pypdf_{test_pdf.stem}.txt",
            add_page_markers=True
        )

        if result['status'] == 'success':
            print(f"✓ 转换成功")
            print(f"  输出: {result['output_path']}")
            print(f"  页数: {result['metadata']['total_pages']}")
            print(f"  字符: {result['stats']['total_characters']:,}")
            print(f"  耗时: {result['duration_seconds']:.2f} 秒")
        else:
            print(f"✗ 转换失败: {result.get('error')}")
            return

        # 测试2: 高级功能
        print("\n" + "-"*80)
        print("测试2: 高级功能")
        print("-"*80)

        loader = PyPDFLoader(str(test_pdf))
        reader = loader.load_document()

        # 提取元数据
        print("\n元数据提取:")
        metadata = loader.extract_metadata(reader)
        print(f"  标题: {metadata.title}")
        print(f"  作者: {metadata.author}")
        print(f"  总页数: {metadata.total_pages}")

        # 提取章节
        print("\n章节提取:")
        sections = loader.extract_sections(reader)
        print(f"  提取到 {len(sections)} 个章节")
        for i, (page_num, pos, title) in enumerate(sections[:5]):
            print(f"    {i+1}. 第 {page_num} 页: {title}")
        if len(sections) > 5:
            print(f"    ... 还有 {len(sections)-5} 个章节")

        # 获取统计
        print("\n文档统计:")
        stats = loader.get_document_stats(reader)
        print(f"  总页数: {stats['total_pages']}")
        print(f"  总字符: {stats['total_characters']:,}")
        print(f"  空页面: {stats['empty_pages']}")
        print(f"  有图像页面: {stats['pages_with_images']}")
        print(f"  平均每页: {stats['avg_chars_per_page']:.0f} 字符")

        # 测试3: 验证分页标记
        print("\n" + "-"*80)
        print("测试3: 验证分页标记")
        print("-"*80)

        output_txt = Path(result['output_path'])
        with open(output_txt, 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        pattern = r'\n={80}\n第\s*(\d+)\s*页\s*/\s*共\s*\d+\s*页\n={80}\n'
        markers = list(re.finditer(pattern, content))

        print(f"\n分页标记验证:")
        print(f"  检测到标记: {len(markers)} 个")
        print(f"  预期标记: {result['metadata']['total_pages']} 个")

        if len(markers) == result['metadata']['total_pages']:
            print(f"  ✓ 分页标记完整")
        else:
            print(f"  ✗ 分页标记不完整")

        # 显示前3个标记
        for i, match in enumerate(markers[:3]):
            page_num = match.group(1)
            context = content[match.start():match.start()+50].replace('\n', ' ')
            print(f"    标记 {i+1}: ...{context}")

    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    test_pypdf_loader()
