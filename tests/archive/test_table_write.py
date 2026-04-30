"""
测试TableProcessor模块 - 输出到文件
"""
from pathlib import Path
from rag_project.data_loader.table_processor import TableProcessor
import traceback
import os

output_file = Path("table_test_result.txt")

with open(output_file, 'w', encoding='utf-8') as out:
    def log(msg):
        out.write(msg + '\n')

    log("="*80)
    log("测试TableProcessor模块")
    log("="*80)

    # 1. 创建processor
    try:
        processor = TableProcessor()
        log("[OK] TableProcessor 创建成功")
    except Exception as e:
        log(f"[FAIL] TableProcessor 创建失败: {e}")
        traceback.print_exc(file=out)
        exit(1)

    # 2. 查找并读取文件
    base = Path("知识库/知识库")
    target_file = None

    # 查找包含2021和merged的txt文件
    for root, dirs, files in os.walk(base):
        for file in files:
            # 只检查2021和merged，不检查中文字符避免编码问题
            if file.endswith('.txt') and '2021' in file and 'merged' in file:
                # 验证文件大小合理（不是太小）
                full_path = Path(root) / file
                if full_path.stat().st_size > 10000:  # 至少10KB
                    target_file = full_path
                    break
        if target_file:
            break

    if not target_file:
        log(f"[FAIL] 未找到目标文件")
        exit(1)

    if not target_file.exists():
        log(f"[FAIL] 文件不存在: {target_file}")
        exit(1)

    log(f"\n[OK] 找到文件: {target_file.name}")
    log(f"位置: {target_file.parent}")

    with open(target_file, 'r', encoding='utf-8') as f:
        text = f.read()

    log(f"\n文件信息:")
    log(f"  字符数: {len(text):,}")
    log(f"  行数: {len(text.splitlines()):,}")

    # 3. 检测表格
    log(f"\n开始检测表格...")
    try:
        tables = processor.detector.detect_tables(text)
        log(f"[OK] 检测到 {len(tables)} 个表格")

        if tables:
            for i, table in enumerate(tables[:10], 1):
                log(f"\n表格 {i}:")
                log(f"  位置: 行 {table.start_line}-{table.end_line}")
                log(f"  类型: {table.table_type}")
                log(f"  置信度: {table.confidence:.2f}")
                log(f"  大小: {len(table.headers)} 列 x {len(table.rows)} 行")

                if table.headers:
                    log(f"  表头: {', '.join(table.headers[:5])}")

                # 显示前2行原始内容
                lines = table.raw_text.split('\n')
                log(f"  原始内容（前2行）:")
                for line in lines[:2]:
                    log(f"    {line[:60]}...")

            if len(tables) > 10:
                log(f"\n... 还有 {len(tables) - 10} 个表格")

        # 4. 测试转换
        if tables:
            log(f"\n" + "="*80)
            log("测试表格转换")
            log("="*80)

            first_table = tables[0]

            # 转为描述性文本
            description = processor.converter.to_description(
                first_table,
                f"{target_file.stem}第{first_table.start_line}页"
            )
            log(f"\n转为描述性文本:")
            log(f"  长度: {len(description)} 字符")
            log(f"  内容: {description[:200]}...")

            # 转为Markdown
            markdown = processor.converter.to_markdown(first_table)
            log(f"\n转为Markdown:")
            log(f"  长度: {len(markdown)} 字符")
            lines = markdown.split('\n')
            for line in lines[:5]:
                log(f"  {line}")

    except Exception as e:
        log(f"[FAIL] 检测失败: {e}")
        traceback.print_exc(file=out)

    log(f"\n" + "="*80)
    log("测试完成")
    log("="*80)

print(f"结果已保存到: {output_file}")
print(f"请查看文件内容")
