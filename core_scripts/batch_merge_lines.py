"""
批量应用行合并到所有TXT文档
将非段落换行合并，优化文档质量
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from rag_project.data_loader.text_line_merger import TextLineMerger
from rag_project.utils.logger import logger


def batch_merge_lines(
    base_dir: str = "知识库/知识库",
    backup: bool = True
):
    """
    批量合并所有TXT文档的行

    Args:
        base_dir: 知识库根目录
        backup: 是否备份原始文件
    """
    base = Path(base_dir)

    print("="*80)
    print("批量行合并 - 文档优化")
    print("="*80)

    # 1. 扫描所有TXT文件
    print("\n[步骤1] 扫描TXT文件...")
    txt_files = []

    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith('.txt'):
                full_path = Path(root) / file

                # 跳过备份和已合并文件
                if 'backup' in file.lower():
                    continue
                if 'merged' in file.lower():
                    continue

                # 只处理大于10KB的文件
                if full_path.stat().st_size > 10240:
                    txt_files.append(full_path)

    print(f"找到 {len(txt_files)} 个需要处理的TXT文件")

    if not txt_files:
        print("没有找到需要处理的文件")
        return

    # 2. 备份原始文件
    if backup:
        print("\n[步骤2] 备份原始文件...")
        backup_dir = base / "backup_original"

        if not backup_dir.exists():
            backup_dir.mkdir(parents=True)
            print(f"创建备份目录: {backup_dir}")

        backed_up = 0
        for file_path in txt_files[:5]:  # 先显示前5个
            rel_path = file_path.relative_to(base)
            backup_path = backup_dir / rel_path

            # 创建子目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(file_path, backup_path)
            backed_up += 1

        print(f"已备份 {backed_up} 个文件到: {backup_dir}")
        print(f"  (注: 为节省时间，只显示部分备份信息)")

    # 3. 批量合并
    print("\n[步骤3] 批量合并行...")
    merger = TextLineMerger()

    stats = {
        'total': len(txt_files),
        'processed': 0,
        'failed': 0,
        'total_lines_original': 0,
        'total_lines_merged': 0,
        'total_reduced': 0
    }

    for i, file_path in enumerate(txt_files, 1):
        try:
            # 读取
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            original_lines = len(text.splitlines())
            stats['total_lines_original'] += original_lines

            # 合并
            merged_text, merge_stats = merger.merge_with_stats(text)

            merged_lines = len(merged_text.splitlines())
            stats['total_lines_merged'] += merged_lines
            stats['total_reduced'] += merge_stats['lines_reduced']

            # 保存
            output_path = file_path.parent / f"{file_path.stem}_merged.txt"

            # 如果已存在merged文件，先删除
            if output_path.exists():
                output_path.unlink()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(merged_text)

            stats['processed'] += 1

            # 显示进度
            if i <= 5 or i % 20 == 0 or i == len(txt_files):
                reduction_pct = merge_stats['reduction_ratio'] * 100
                print(f"[{i}/{len(txt_files)}] {file_path.name}")
                print(f"  行数: {original_lines:,} → {merged_lines:,} (-{reduction_pct:.1f}%)")

        except Exception as e:
            stats['failed'] += 1
            logger.error(f"处理失败 {file_path}: {e}")
            print(f"[{i}/{len(txt_files)}] {file_path.name} - 失败: {e}")

    # 4. 统计
    print("\n" + "="*80)
    print("处理完成")
    print("="*80)

    print(f"\n处理统计:")
    print(f"  总文件数: {stats['total']}")
    print(f"  成功处理: {stats['processed']}")
    print(f"  处理失败: {stats['failed']}")

    if stats['processed'] > 0:
        avg_reduction = (stats['total_reduced'] / stats['total_lines_original']) * 100
        print(f"\n行合并效果:")
        print(f"  原始总行数: {stats['total_lines_original']:,}")
        print(f"  合并后行数: {stats['total_lines_merged']:,}")
        print(f"  减少行数: {stats['total_reduced']:,}")
        print(f"  平均减少比例: {avg_reduction:.1f}%")

    print(f"\n合并后的文件保存在原目录，文件名后缀为 '_merged.txt'")
    print(f"原始文件已备份到: {backup_dir if backup else '未备份'}")

    return stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="批量合并TXT文档行")
    parser.add_argument(
        "--dir",
        type=str,
        default="知识库/知识库",
        help="知识库根目录"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份原始文件"
    )

    args = parser.parse_args()

    stats = batch_merge_lines(
        base_dir=args.dir,
        backup=not args.no_backup
    )

    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
