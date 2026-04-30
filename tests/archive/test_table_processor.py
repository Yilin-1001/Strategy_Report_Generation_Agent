"""
测试表格处理功能
在中国通用航空2021文档上测试TableProcessor
"""
from pathlib import Path
import os
from rag_project.data_loader.table_processor import TableProcessor
from rag_project.utils.logger import logger

def test_table_processing():
    """测试表格处理"""

    # 1. 查找文档
    base = Path("知识库/知识库")
    target_file = None

    print("搜索文档...")
    for root, dirs, files in os.walk(base):
        for file in files:
            # 使用已知的merged文件进行测试
            if file.endswith('.txt') and '2021' in file and 'merged' in file and '通用航空' in root:
                target_file = Path(root) / file
                break
        if target_file:
            break

    if not target_file:
        print("未找到文档！")
        return

    print(f"\n找到文档: {target_file.name}")
    print(f"位置: {target_file.parent}")

    # 2. 读取文档
    print("\n" + "="*80)
    print("读取文档...")
    print("="*80)

    with open(target_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"字符数: {len(text):,}")
    print(f"行数: {len(text.splitlines()):,}")

    # 3. 处理表格
    print("\n" + "="*80)
    print("开始表格检测与处理...")
    print("="*80)

    processor = TableProcessor()
    result = processor.process_document(text, target_file.stem)

    # 4. 显示统计结果
    print("\n" + "="*80)
    print("处理结果统计")
    print("="*80)

    stats = result['stats']
    print(f"检测到表格: {stats['tables_found']} 个")
    print(f"已处理: {stats['tables_processed']} 个")
    print(f"  -> 转为描述性文本: {stats['converted_to_description']} 个")
    print(f"  -> 保留为Markdown: {stats['converted_to_markdown']} 个")

    print(f"\n文本长度变化:")
    print(f"  处理前: {len(text):,} 字符")
    print(f"  处理后: {len(result['text']):,} 字符")
    print(f"  变化: {len(result['text']) - len(text):+d} 字符")

    # 5. 显示表格详情
    if result.get('tables'):
        print("\n" + "="*80)
        print("检测到的表格详情")
        print("="*80)

        for i, table in enumerate(result['tables'], 1):
            print(f"\n表格 {i}:")
            print(f"  位置: 第 {table.start_line}-{table.end_line} 行")
            print(f"  类型: {table.table_type}")
            print(f"  置信度: {table.confidence:.2f}")
            print(f"  大小: {len(table.headers)} 列 × {len(table.rows)} 行")

            if table.headers:
                print(f"  表头: {', '.join(table.headers[:5])}" +
                      (f" ..." if len(table.headers) > 5 else ""))

            # 显示原始表格片段（前3行）
            lines = table.raw_text.split('\n')
            print(f"  原始内容（前3行）:")
            for line in lines[:3]:
                print(f"    {line[:80]}")

    # 6. 显示转换示例
    if result.get('tables'):
        print("\n" + "="*80)
        print("表格转换示例")
        print("="*80)

        # 显示第一个表格的转换结果
        first_table = result['tables'][0]

        print("\n原始表格:")
        print("-" * 40)
        lines = first_table.raw_text.split('\n')
        for line in lines[:8]:  # 只显示前8行
            print(line)
        if len(lines) > 8:
            print(f"... (共 {len(lines)} 行)")

        print("\n转换为描述性文本:")
        print("-" * 40)
        description = processor.converter.to_description(
            first_table,
            f"{target_file.stem}第{first_table.start_line}页"
        )
        print(description[:500])
        if len(description) > 500:
            print("...")

        print("\n转换为Markdown格式:")
        print("-" * 40)
        markdown = processor.converter.to_markdown(first_table)
        lines = markdown.split('\n')
        for line in lines[:10]:  # 只显示前10行
            print(line)
        if len(lines) > 10:
            print(f"... (共 {len(lines)} 行)")

    # 7. 保存处理后的文档
    output_path = target_file.parent / f"{target_file.stem}_table_processed.txt"

    print("\n" + "="*80)
    print("保存处理后的文档")
    print("="*80)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    print(f"\n已保存到: {output_path.name}")
    print(f"位置: {output_path.parent}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"\n文件对比:")
    print(f"  原始文件: {target_file.name}")
    print(f"  处理后: {output_path.name}")

    return {
        'original_file': str(target_file),
        'processed_file': str(output_path),
        'stats': stats,
        'tables': result.get('tables', [])
    }


if __name__ == "__main__":
    try:
        results = test_table_processing()

        print("\n" + "="*80)
        print("总结")
        print("="*80)
        print(f"测试成功！")
        print(f"检测到 {results['stats']['tables_found']} 个表格")
        print(f"处理后的文件已保存")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
