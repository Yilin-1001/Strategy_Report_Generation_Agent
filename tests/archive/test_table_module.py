"""
直接测试TableProcessor模块
"""
from pathlib import Path
from rag_project.data_loader.table_processor import TableProcessor

# 测试模块导入
print("测试TableProcessor模块...")

try:
    processor = TableProcessor()
    print("[OK] TableProcessor 创建成功")
except Exception as e:
    print(f"[FAIL] TableProcessor 创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 读取测试文件
target_file = Path("知识库/知识库/研究报告/中国通用航空2021_merged.txt")

if not target_file.exists():
    print(f"文件不存在: {target_file}")
    exit(1)

print(f"\n读取文件: {target_file.name}")

with open(target_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"文件大小: {len(text):,} 字符")
print(f"行数: {len(text.splitlines()):,}")

# 测试表格检测
print("\n开始检测表格...")
try:
    tables = processor.detector.detect_tables(text)
    print(f"检测到 {len(tables)} 个表格")

    for i, table in enumerate(tables[:3], 1):
        print(f"\n表格 {i}:")
        print(f"  位置: 行 {table.start_line}-{table.end_line}")
        print(f"  类型: {table.table_type}")
        print(f"  表头: {table.headers}")
        print(f"  行数: {len(table.rows)}")

except Exception as e:
    print(f"检测失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")
