"""
测试文本行合并功能
"""
from pathlib import Path
from rag_project.data_loader.text_line_merger import TextLineMerger, merge_pdf_text
from rag_project.utils.logger import logger


def test_line_merger_with_sample():
    """测试基本合并功能"""
    print("="*80)
    print("文本行合���器测试")
    print("="*80)

    # 测试用例1: 简单合并
    test1 = """
这是一个很长的句子，因为PDF版面宽度限制，
它被强制分成了多行，但实际上应该是一段。
"""

    print("\n测试1: 简单句子合并")
    print("-"*80)
    print("原始:")
    print(repr(test1))

    merged1 = merge_pdf_text(test1)
    print("\n合并后:")
    print(repr(merged1))

    # 测试用例2: 段落保留
    test2 = """
第一段内容。
因为这里句号结束了。

第二段内容，
这里换行了但应该保留。
"""

    print("\n\n测试2: 段落保留")
    print("-"*80)
    print("原始:")
    print(test2)

    merged2 = merge_pdf_text(test2)
    print("\n合并后:")
    print(merged2)

    # 测试用例3: 列表
    test3 = """
这是一个列表：
1. 第一项内容，
   继续在下一行。
2. 第二项内容，
   也继续在下一行。

列表结束后的段落。
"""

    print("\n\n测试3: 列表处理")
    print("-"*80)
    print("原始:")
    print(test3)

    merged3 = merge_pdf_text(test3)
    print("\n合并后:")
    print(merged3)

    # 测试用例4: 连字符
    test4 = """
这是一个很长的单词，因为换行被断开了，例如
configuration。
"""

    print("\n\n测试4: 连字符处理")
    print("-"*80)
    print("原始:")
    print(test4)

    merged4 = merge_pdf_text(test4)
    print("\n合并后:")
    print(merged4)


def test_line_merger_with_real_pdf():
    """测试真实PDF文本"""
    print("\n\n" + "="*80)
    print("真实PDF文本测试")
    print("="*80)

    # 查找测试文件
    kb_path = Path("知识库/知识库")
    txt_files = list(kb_path.rglob("*.txt"))

    if not txt_files:
        print("未找到TXT文件")
        return

    # 使用第一个文件
    test_file = txt_files[0]
    print(f"\n测试文件: {test_file.name}")

    # 读取部分内容
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 只测试前3000字符
    sample_text = content[:3000]

    print(f"\n原始文本 (前{len(sample_text)}字符):")
    print("-"*80)
    print(sample_text[:500])
    print("...")

    # 合并
    merger = TextLineMerger()
    merged_text, stats = merger.merge_with_stats(sample_text)

    print(f"\n合并后文本 (前500字符):")
    print("-"*80)
    print(merged_text[:500])
    print("...")

    print(f"\n统计信息:")
    print(f"  原始行数: {stats['original_lines']}")
    print(f"  合并行数: {stats['merged_lines']}")
    print(f"  减少行数: {stats['lines_reduced']}")
    print(f"  减少比例: {stats['reduction_ratio']:.1%}")


def compare_before_after():
    """对比合并前后的chunking效果"""
    print("\n\n" + "="*80)
    print("合并前后对比测试")
    print("="*80)

    # 模拟PDF转换后的文本
    pdf_text = """
================================================================================
第 1 页 / 共 2 页
================================================================================

中国通用航空发展报告

通用航空是指使用民用航空器从事公共航空运输以
外的民用航空活动，包括从事工业、农业、林业、渔业
和建筑业的作业飞行以及医疗卫生、抢险救灾、科
学实验、教育训练、文化体育等方面的飞行活动。

通用航空产业链长、服务领域广、带动效应强，
是综合交通运输体系的重要组成部分。

================================================================================
第 2 页 / 共 2 页
================================================================================

近年来，中国通用航空发展迅速，
产业规模不断扩大，服务能力持续提升。
"""

    print("\n原始PDF文本:")
    print("-"*80)
    print(pdf_text)

    print("\n" + "="*80)
    print("应用行合并...")
    print("="*80)

    merger = TextLineMerger()
    merged_text, stats = merger.merge_with_stats(pdf_text)

    print("\n合并后文本:")
    print("-"*80)
    print(merged_text)

    print("\n" + "="*80)
    print("对比结果:")
    print("="*80)
    print(f"原始: {stats['original_lines']} 行")
    print(f"合并: {stats['merged_lines']} 行")
    print(f"效果: 减少 {stats['lines_reduced']} 行 ({stats['reduction_ratio']:.1%})")

    # 分析chunks质量
    print("\nChunks质量分析:")
    print("-"*80)

    # 模拟按固定chunk_size分割
    chunk_size = 200

    print(f"\n使用固定chunk_size={chunk_size}:")

    # 原始文本的chunks
    original_chunks = []
    for i in range(0, len(pdf_text), chunk_size):
        chunk = pdf_text[i:i+chunk_size]
        original_chunks.append(chunk)

    print(f"原始chunks: {len(original_chunks)} 个")
    for i, chunk in enumerate(original_chunks[:3]):
        last_50 = chunk[-50:] if len(chunk) > 50 else chunk
        print(f"  Chunk {i+1} 结尾: ...{last_50}")

    # 合并文本的chunks
    merged_chunks = []
    for i in range(0, len(merged_text), chunk_size):
        chunk = merged_text[i:i+chunk_size]
        merged_chunks.append(chunk)

    print(f"\n合并chunks: {len(merged_chunks)} 个")
    for i, chunk in enumerate(merged_chunks[:3]):
        last_50 = chunk[-50:] if len(chunk) > 50 else chunk
        print(f"  Chunk {i+1} 结尾: ...{last_50}")


if __name__ == "__main__":
    # 测试1: 基本功能
    test_line_merger_with_sample()

    # 测试2: 真实PDF
    # test_line_merger_with_real_pdf()

    # 测试3: 对比效果
    compare_before_after()
