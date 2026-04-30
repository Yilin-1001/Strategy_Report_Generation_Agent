"""
测试Pipeline是否会正确跳过PDF文件
"""
from pathlib import Path
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger


def test_pdf_skipping():
    """测试PDF文件跳过功能"""
    kb_path = Path("知识库/知识库")

    # 查找测试文件
    pdf_files = list(kb_path.rglob("*.pdf"))
    txt_files = list(kb_path.rglob("*.txt"))

    print("="*80)
    print("测试PDF跳过功能")
    print("="*80)

    # 获取第一个PDF和对应的TXT
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"\n测试PDF文件: {test_pdf.name}")

        # 创建pipeline
        pipeline = RAGPipeline()

        # 测试处理PDF文件
        print("\n1. 测试处理PDF文件:")
        chunks = pipeline._load_and_chunk_file(str(test_pdf))
        print(f"   返回chunks数量: {len(chunks)}")

        if len(chunks) == 0:
            print("   ✓ PDF文件被正确跳过")
        else:
            print("   ✗ 警告：PDF文件没有被跳过！")
    else:
        print("\n未找到PDF文件进行测试")

    # 测试TXT文件
    if txt_files:
        test_txt = txt_files[0]
        print(f"\n2. 测试处理TXT文件: {test_txt.name}")
        pipeline = RAGPipeline()
        chunks = pipeline._load_and_chunk_file(str(test_txt))
        print(f"   返回chunks数量: {len(chunks)}")

        if len(chunks) > 0:
            print("   ✓ TXT文件被正确处理")
        else:
            print("   注意：TXT文件没有生成chunks（可能为空或其他原因）")

    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("当前chunking策略:")
    print("  ✓ 处理 .txt 文件")
    print("  ✗ 跳过 .pdf 文件（已转换为TXT）")
    print("  ✗ 跳过 .docx/.doc 文件")
    print("="*80)


if __name__ == "__main__":
    test_pdf_skipping()
