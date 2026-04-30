"""
直接测试中国通用航空2021文档
"""
from pathlib import Path
from rag_project.data_loader.text_line_merger import TextLineMerger
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
import json


def main():
    # 找到文件
    kb = Path('知识库/知识库')
    txt_file = None

    for f in kb.rglob('*.txt'):
        if '2021' in f.name and 'cleaned_backup' not in f.name:
            txt_file = f
            break

    if not txt_file:
        print('File not found')
        return

    # 读取文件（只读前3000字符测试）
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()[:5000]  # 只测试前5000字符

    output = []
    output.append('================================================================================')
    output.append('China Aviation 2021 - Line Merge Test')
    output.append('================================================================================')
    output.append(f'\nFile: {txt_file.name}')
    output.append(f'Test text length: {len(text)} chars')

    # 应用行合并
    merger = TextLineMerger()
    merged_text, stats = merger.merge_with_stats(text)

    output.append(f'\nOriginal lines: {stats["original_lines"]}')
    output.append(f'Merged lines: {stats["merged_lines"]}')
    output.append(f'Lines reduced: {stats["lines_reduced"]} ({stats["reduction_ratio"]:.1%})')

    # Chunking测试
    output.append('\n================================================================================')
    output.append('Chunking Comparison')
    output.append('================================================================================')

    # 不合并
    doc1 = Document(page_content=text, metadata={'source': 'test.txt'})
    chunker1 = ConfigurableChunker(use_v2_metadata=True)
    chunks1 = chunker1.split_documents([doc1], 'news')

    output.append(f'\nWithout merge: {len(chunks1)} chunks')
    output.append(f'  Avg length: {sum(len(c.page_content) for c in chunks1) / len(chunks1):.0f} chars')

    # 合并后
    doc2 = Document(page_content=merged_text, metadata={'source': 'test.txt'})
    chunker2 = ConfigurableChunker(use_v2_metadata=True)
    chunks2 = chunker2.split_documents([doc2], 'news')

    output.append(f'\nWith merge: {len(chunks2)} chunks')
    output.append(f'  Avg length: {sum(len(c.page_content) for c in chunks2) / len(chunks2):.0f} chars')

    # 保存结果
    result_text = '\n'.join(output)

    print(result_text)

    with open('china_2021_test_result.txt', 'w', encoding='utf-8') as f:
        f.write(result_text)

    # 保存JSON结果
    result = {
        'original_lines': stats['original_lines'],
        'merged_lines': stats['merged_lines'],
        'lines_reduced': stats['lines_reduced'],
        'reduction_ratio': stats['reduction_ratio'],
        'chunks_without_merge': len(chunks1),
        'chunks_with_merge': len(chunks2),
        'avg_chunk_length_without_merge': sum(len(c.page_content) for c in chunks1) / len(chunks1),
        'avg_chunk_length_with_merge': sum(len(c.page_content) for c in chunks2) / len(chunks2)
    }

    with open('china_2021_test_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\nResults saved to:')
    print('  - china_2021_test_result.txt')
    print('  - china_2021_test_result.json')


if __name__ == '__main__':
    main()
