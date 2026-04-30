"""
保存中国通用航空2021的合并版本
"""
from pathlib import Path
import os
from rag_project.data_loader.text_line_merger import TextLineMerger

def find_and_merge():
    # 遍历所有子目录
    base = Path("知识库/知识库")

    print("搜索文件...")

    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith('.txt') and '2021' in file:
                full_path = Path(root) / file

                # 跳过备份文件
                if 'backup' in file.lower() or 'cleaned' in file.lower():
                    continue

                print(f"\n找到文件: {full_path}")
                print(f"文件大小: {full_path.stat().st_size / 1024:.1f} KB")

                # 读取
                print("读取文件...")
                with open(full_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                print(f"字符数: {len(text):,}")
                print(f"行数: {len(text.split(chr(10))}")

                # 合并
                print("\n应用行合并...")
                merger = TextLineMerger()
                merged, stats = merger.merge_with_stats(text)

                print(f"合并后字符数: {len(merged):,}")
                print(f"合并后行数: {len(merged.splitlines())}")
                print(f"减少行数: {stats['lines_reduced']} ({stats['reduction_ratio']:.1%})")

                # 保存
                output_path = full_path.parent / f"{full_path.stem}_merged.txt"

                print(f"\n保存到: {output_path.name}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(merged)

                print(f"\n成功保存!")
                print(f"\n文件位置:")
                print(f"  原文件: {full_path}")
                print(f"  合并文件: {output_path}")

                return str(full_path), str(output_path)

    print("\n未找到匹配的文件")
    return None, None


if __name__ == "__main__":
    original, merged = find_and_merge()

    if original and merged:
        print("\n" + "="*80)
        print("总结")
        print("="*80)
        print(f"原始文件: {Path(original).parent}")
        print(f"合并文件: {Path(merged).name}")
