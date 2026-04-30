"""
在原始文档上测试表格处理
"""
import os
from pathlib import Path
from rag_project.data_loader.table_processor import TableProcessor

base = Path("知识库/知识库")
target_file = None

# 查找原始文档（非merged）
for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith('.txt') and '2021' in file and 'merged' not in file and 'backup' not in file:
            full_path = Path(root) / file
            if full_path.stat().st_size > 100000:
                target_file = full_path
                break
    if target_file:
        break

if not target_file:
    print("未找到原始文档")
    exit(1)

print(f"文档: {target_file.name}")
print(f"大小: {target_file.stat().st_size / 1024:.1f} KB")

with open(target_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"字符数: {len(text):,}")
print(f"行数: {len(text.splitlines()):,}")

print("\n开始表格检测...")
processor = TableProcessor()
result = processor.process_document(text, target_file.stem)

stats = result['stats']
print(f"\n检测结果:")
print(f"  检测到表格: {stats['tables_found']} 个")
print(f"  已处理: {stats['tables_processed']} 个")
print(f"  转为描述性文本: {stats['converted_to_description']} 个")
print(f"  保留为Markdown: {stats['converted_to_markdown']} 个")

if result.get('tables'):
    print(f"\n前3个表格:")
    for i, table in enumerate(result['tables'][:3], 1):
        print(f"\n表格 {i}:")
        print(f"  位置: 第 {table.start_line}-{table.end_line} 行")
        print(f"  类型: {table.table_type}")
        print(f"  大小: {len(table.headers)} 列 x {len(table.rows)} 行")
        if table.headers:
            print(f"  表头: {table.headers[:3]}")
else:
    print("\n未检测到表格")
