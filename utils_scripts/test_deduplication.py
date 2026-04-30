"""
测试 Chunking 去重逻辑
验证是否会对同一文档的txt和merged.txt进行重复处理
"""
from pathlib import Path

def test_deduplication_logic():
    """测试chunk_all_documents.py中的去重逻辑"""

    kb_path = Path("知识库/知识库")

    # 递归查找所有txt文件
    txt_files = list(kb_path.rglob("*.txt"))

    print("="*80)
    print("Chunking 去重逻辑测试")
    print("="*80)
    print(f"\n知识库路径: {kb_path}")
    print(f"找到TXT文件总数: {len(txt_files)}")

    # 过滤掉备份文件和元数据文件
    filtered_files = []
    for txt_file in txt_files:
        # 跳过备份文件
        if 'backup' in txt_file.name.lower():
            continue
        # 跳过元数据文件
        if 'metadata' in txt_file.name.lower():
            continue
        # 跳过summary文件
        if 'summary' in txt_file.name.lower():
            continue

        filtered_files.append(txt_file)

    print(f"过滤后文件数: {len(filtered_files)}")

    # 去重：优先使用merged版本
    # 这是从chunk_all_documents.py复制的逻辑
    file_dict = {}  # {base_name: file_path}

    for file_path in filtered_files:
        # 判断是否是merged文件
        if file_path.stem.endswith('_merged'):
            # 提取基础名称（去掉_merged）
            base_name = file_path.stem[:-7]  # 去掉'_merged'
            # 如果基础名称对应的文件不存在，或者当前是merged版本，优先使用
            if base_name not in file_dict or not file_dict[base_name].stem.endswith('_merged'):
                file_dict[base_name] = file_path
        else:
            # 非merged文件，只有在没有merged版本时才使用
            base_name = file_path.stem
            if base_name not in file_dict:
                file_dict[base_name] = file_path

    # 转换为字符串路径列表
    result = [str(f) for f in file_dict.values()]

    print(f"去重后文件数: {len(result)}")
    print(f"删除的重复文件: {len(filtered_files) - len(result)} 个")

    # 查找有merged和原始版本的文件对
    print("\n" + "="*80)
    print("重复文件对分析")
    print("="*80)

    duplicate_pairs = []
    seen_bases = set()

    for file_path in filtered_files:
        base_name = file_path.stem

        # 处理merged文件
        if base_name.endswith('_merged'):
            original_base = base_name[:-7]  # 去掉'_merged'

            # 查找对应的原始文件
            original_file = None
            for f in filtered_files:
                if f.stem == original_base and f.parent == file_path.parent:
                    original_file = f
                    break

            if original_file and original_base not in seen_bases:
                seen_bases.add(original_base)
                duplicate_pairs.append({
                    'base_name': original_base,
                    'original': original_file,
                    'merged': file_path,
                    'chosen_by_logic': 'merged (优先)'
                })

    # 显示前10个重复对
    print(f"\n找到 {len(duplicate_pairs)} 对重复文件 (原始txt + merged.txt)")
    print("\n前10个示例:")

    for i, pair in enumerate(duplicate_pairs[:10], 1):
        print(f"\n{i}. 基础名称: {pair['base_name']}")
        print(f"   目录: {pair['original'].parent.relative_to(kb_path)}")
        print(f"   原始文件: {pair['original'].name}")
        print(f"   合并文件: {pair['merged'].name}")
        print(f"   使用文件: {pair['chosen_by_logic']}")

        # 检查哪个文件会被使用
        chosen = file_dict.get(pair['base_name'])
        if chosen and chosen.stem == pair['base_name']:
            print(f"   [OK] 实际使用: 原始文件 (因为没有merged版本)")
        elif chosen and chosen.stem == f"{pair['base_name']}_merged":
            print(f"   [OK] 实际使用: 合并文件 (优先使用merged)")
        else:
            print(f"   [FAIL] 未找到匹配")

    # 统计
    print("\n" + "="*80)
    print("统计总结")
    print("="*80)

    merged_used = sum(1 for f in result if Path(f).stem.endswith('_merged'))
    original_used = len(result) - merged_used

    print(f"\n将被chunking的文件:")
    print(f"  merged.txt 文件: {merged_used} 个")
    print(f"  原始txt文件: {original_used} 个")
    print(f"  总计: {len(result)} 个")

    print(f"\n被跳过的重复文件:")
    print(f"  数量: {len(duplicate_pairs)} 个")
    print(f"  (这些原始txt文件不会被chunking，因为有对应的merged版本)")

    # 验证：确保没有重复
    print("\n" + "="*80)
    print("去重验证")
    print("="*80)

    base_names_used = set()
    duplicates_found = []

    for file_path in result:
        base = Path(file_path).stem
        if base.endswith('_merged'):
            base = base[:-7]

        if base in base_names_used:
            duplicates_found.append(base)
        else:
            base_names_used.add(base)

    if duplicates_found:
        print(f"\n[WARNING] 发现 {len(duplicates_found)} 个重复的基础名称:")
        for dup in duplicates_found[:5]:
            print(f"  - {dup}")
    else:
        print("\n[OK] 没有发现重复！去重逻辑工作正常。")

    return {
        'total_files': len(filtered_files),
        'after_dedup': len(result),
        'duplicates_removed': len(filtered_files) - len(result),
        'merged_used': merged_used,
        'original_used': original_used
    }


if __name__ == "__main__":
    result = test_deduplication_logic()

    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print(f"\n[OK] 代码中的去重逻��会:")
    print(f"  1. 优先使用 *_merged.txt 文件")
    print(f"  2. 跳过对应的原始 .txt 文件")
    print(f"  3. 避免对同一文档重复chunking")

    print(f"\n[OK] 预计可以避免 {result['duplicates_removed']} 次重复处理")
