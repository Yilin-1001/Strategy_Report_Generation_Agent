"""
保存中国通用航空2021的完整合并后版本
"""
from pathlib import Path
from rag_project.data_loader.text_line_merger import TextLineMerger
from datetime import datetime


def save_merged_version():
    """保存合并后的完整文档"""
    # 查找文件
    kb_path = Path("知识库/知识库")
    source_file = None

    for f in kb_path.rglob("*.txt"):
        if "2021" in f.name and "通用" in f.name and "cleaned_backup" not in f.name:
            source_file = f
            break

    if not source_file:
        print("未找到文件: 中国通用航空2021.txt")
        return

    print("="*80)
    print("保存中国通用航空2021 - 行合并版本")
    print("="*80)
    print(f"\n源文件: {source_file}")
    print(f"文件大小: {source_file.stat().st_size / 1024:.1f} KB")

    # 读取完整文件
    print("\n读取文件...")
    with open(source_file, 'r', encoding='utf-8') as f:
        original_text = f.read()

    print(f"原始文本: {len(original_text):,} 字符, {len(original_text.split(chr(10)))} 行")

    # 应用行合并
    print("\n应用行合并...")
    merger = TextLineMerger()
    merged_text, stats = merger.merge_with_stats(original_text)

    print(f"合并后文本: {len(merged_text):,} 字符, {len(merged_text.split(chr(10)))} 行")
    print(f"减少: {stats['lines_reduced']} 行 ({stats['reduction_ratio']:.1%})")

    # 保存合并后版本
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = source_file.parent / f"{source_file.stem}_merged_{timestamp}.txt"

    print(f"\n保存合并后版本...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_text)

    print(f"已保存: {output_file}")

    # 同时保存一个覆盖原文件的版本（可选）
    backup_file = source_file.parent / f"{source_file.stem}_before_merge_backup.txt"
    import shutil
    shutil.copy(source_file, backup_file)
    print(f"已备份原文件: {backup_file}")

    # 询问是否覆盖原文件
    print("\n" + "="*80)
    print("提示:")
    print("="*80)
    print(f"合并后文件: {output_file.name}")
    print(f"原文件备份: {backup_file.name}")
    print("\n如需替换原文件，可以手动重命名:")
    print(f"  1. 删除原文件: {source_file.name}")
    print(f"  2. 重命名合并文件: {output_file.name} -> {source_file.name}")

    return {
        'original_file': str(source_file),
        'backup_file': str(backup_file),
        'merged_file': str(output_file),
        'original_lines': stats['original_lines'],
        'merged_lines': stats['merged_lines'],
        'lines_reduced': stats['lines_reduced']
    }


if __name__ == "__main__":
    result = save_merged_version()

    if result:
        print("\n" + "="*80)
        print("处理完成!")
        print("="*80)
        print(f"合并后版本: {result['merged_file']}")
        print(f"原文件备份: {result['backup_file']}")
