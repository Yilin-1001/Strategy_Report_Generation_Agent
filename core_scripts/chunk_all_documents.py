"""
批量Chunking所有知识库文档
"""
from pathlib import Path
from typing import List
from rag_project.pipeline import RAGPipeline
from rag_project.utils.logger import logger
import json
from datetime import datetime


def get_all_txt_files(kb_path: str = "知识库/知识库") -> List[str]:
    """
    获取所有TXT文件路径
    优先使用merged文件，如果没有则使用原始文件

    Args:
        kb_path: 知识库路径

    Returns:
        TXT文件路径列表（去重后）
    """
    kb = Path(kb_path)

    # 递归查找所有txt文件
    txt_files = list(kb.rglob("*.txt"))

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

    # 去重：优先使用merged版本
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

    return result


def chunk_all_documents(
    kb_path: str = "知识库/知识库",
    chunks_output_path: str = "data/all_chunks.json",
    log_output_path: str = "chunking_log.txt"
):
    """
    批量处理所有文档

    Args:
        kb_path: 知识库路径
        chunks_output_path: chunks输出路径
        log_output_path: 日志输出路径
    """
    # 获取所有TXT文件
    txt_files = get_all_txt_files(kb_path)

    print("="*80)
    print("批量Chunking所有知识库文档")
    print("="*80)
    print(f"\n知识库路径: {kb_path}")
    print(f"找到TXT文件: {len(txt_files)} 个")
    print(f"\n文件列表:")
    for i, f in enumerate(txt_files, 1):
        p = Path(f)
        print(f"  {i}. {p.name} ({p.parent.name})")

    print("\n" + "="*80)
    print("开始Chunking处理")
    print("="*80)

    # 创建Pipeline，传递knowledge_base_path以支持tags提取
    pipeline = RAGPipeline(
        chunks_storage_path=chunks_output_path,
        knowledge_base_path=kb_path
    )

    # 批量处理
    logger.info(f"开始处理 {len(txt_files)} 个文档...")

    start_time = datetime.now()

    total_chunks = pipeline.index_documents(txt_files)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 打印结果
    print("\n" + "="*80)
    print("Chunking完成总结")
    print("="*80)
    print(f"处理文件数: {len(txt_files)}")
    print(f"生成chunks数: {total_chunks}")
    print(f"处理时间: {duration:.2f} 秒")
    print(f"平均每个文件: {duration/len(txt_files):.2f} 秒")
    print(f"Chunks输出: {chunks_output_path}")
    print("="*80)

    # 保存处理日志
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(txt_files),
        'total_chunks': total_chunks,
        'duration_seconds': duration,
        'files_processed': [str(Path(f).name) for f in txt_files],
        'chunks_output': chunks_output_path
    }

    log_file = Path(log_output_path)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"\n处理日志已保存到: {log_output_path}")

    return {
        'total_files': len(txt_files),
        'total_chunks': total_chunks,
        'duration': duration,
        'chunks_output': chunks_output_path
    }


if __name__ == "__main__":
    result = chunk_all_documents()

    if result['total_chunks'] > 0:
        print("\n✓ Chunking成功完成！")
        print(f"✓ 共生成 {result['total_chunks']} 个chunks")
    else:
        print("\n✗ Chunking失败或没有生成chunks")
