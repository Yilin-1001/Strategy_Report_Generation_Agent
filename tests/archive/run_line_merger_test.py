"""
测试文本行合并功能（输出到文件）
"""
from rag_project.data_loader.text_line_merger import TextLineMerger, merge_pdf_text


def test_and_save():
    """测试并保存结果到文件"""
    output_file = "line_merger_test_output.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("文本行合并器测试结果\n")
        f.write("="*80 + "\n\n")

        # 测试用例1: 简单合并
        test1 = """这是一个很长的句子，因为PDF版面宽度限制，
它被强制分成了多行，但实际上应该是一段。"""

        f.write("测试1: 简单句子合并\n")
        f.write("-"*80 + "\n")
        f.write("原始文本:\n")
        f.write(test1 + "\n\n")
        f.write("合并后:\n")
        merged1 = merge_pdf_text(test1)
        f.write(merged1 + "\n\n")

        # 测试用例2: 段落保留
        test2 = """第一段内容。
因为这里句号结束了。

第二段内容，
这里换行了但应该保留。"""

        f.write("="*80 + "\n")
        f.write("测试2: 段落保留\n")
        f.write("-"*80 + "\n")
        f.write("原始文本:\n")
        f.write(test2 + "\n\n")
        f.write("合并后:\n")
        merged2 = merge_pdf_text(test2)
        f.write(merged2 + "\n\n")

        # 测试用例3: 真实PDF文本
        test3 = """
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
"""

        f.write("="*80 + "\n")
        f.write("测试3: 真实PDF文本\n")
        f.write("-"*80 + "\n")
        f.write("原始文本:\n")
        f.write(test3 + "\n\n")

        merger = TextLineMerger()
        merged3, stats = merger.merge_with_stats(test3)

        f.write("合并后:\n")
        f.write(merged3 + "\n\n")

        f.write("统计信息:\n")
        f.write(f"  原始行数: {stats['original_lines']}\n")
        f.write(f"  合并行数: {stats['merged_lines']}\n")
        f.write(f"  减少行数: {stats['lines_reduced']}\n")
        f.write(f"  减少比例: {stats['reduction_ratio']:.1%}\n")

    print(f"测试完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    test_and_save()
