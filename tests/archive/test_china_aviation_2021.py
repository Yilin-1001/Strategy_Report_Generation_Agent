"""
测试"中国通用航空2021"文档的行合并效果
"""
from pathlib import Path
from rag_project.data_loader.text_line_merger import TextLineMerger
from rag_project.data_loader.configurable_splitter import ConfigurableChunker
from langchain_core.documents import Document
from rag_project.utils.logger import logger
import json


def find_china_aviation_2021():
    """查找中国通用航空2021.txt文件"""
    kb_path = Path("知识库/知识库")

    for txt_file in kb_path.rglob("*.txt"):
        if "2021" in txt_file.name and "通用" in txt_file.name and len(txt_file.name) < 30:
            return txt_file
    return None


def test_line_merge_on_document():
    """测试文档的行合并效果"""
    print("="*80)
    print("中国通用航空2021 - 行合并效果测试")
    print("="*80)

    # 查找文件
    txt_file = find_china_aviation_2021()

    if not txt_file:
        print("未找到文件: 中国通用航空2021.txt")
        return

    print(f"\n找到文件: {txt_file}")
    print(f"文件大小: {txt_file.stat().st_size / 1024:.2f} KB")

    # 读取文件
    with open(txt_file, 'r', encoding='utf-8') as f:
        original_text = f.read()

    print(f"\n原始文本统计:")
    print(f"  总字符数: {len(original_text):,}")
    print(f"  总行数: {len(original_text.split(chr(10)))}")

    # 应用行合并
    print("\n" + "="*80)
    print("应用行合并...")
    print("="*80)

    merger = TextLineMerger()
    merged_text, stats = merger.merge_with_stats(original_text)

    print(f"\n合并后统计:")
    print(f"  总字符数: {len(merged_text):,}")
    print(f"  总行数: {len(merged_text.split(chr(10)))}")
    print(f"  减少行数: {stats['lines_reduced']}")
    print(f"  减少比例: {stats['reduction_ratio']:.1%}")

    # 测试chunking效果对比
    print("\n" + "="*80)
    print("Chunking效果对比")
    print("="*80)

    # 方案1: 不合并
    doc1 = Document(page_content=original_text, metadata={"source": txt_file.name})
    chunker1 = ConfigurableChunker(use_v2_metadata=True)
    chunks1 = chunker1.split_documents([doc1], "news")

    print(f"\n【不合并】生成Chunks: {len(chunks1)} 个")
    print(f"  平均长度: {sum(len(c.page_content) for c in chunks1) / len(chunks1):.0f} 字符")
    print(f"  总字符数: {sum(len(c.page_content) for c in chunks1):,} 字符")

    # 方案2: 合并后
    doc2 = Document(page_content=merged_text, metadata={"source": txt_file.name})
    chunker2 = ConfigurableChunker(use_v2_metadata=True)
    chunks2 = chunker2.split_documents([doc2], "news")

    print(f"\n【合并后】生成Chunks: {len(chunks2)} 个")
    print(f"  平均长度: {sum(len(c.page_content) for c in chunks2) / len(chunks2):.0f} 字符")
    print(f"  总字符数: {sum(len(c.page_content) for c in chunks2):,} 字符")

    # 分析chunks质量
    print("\n" + "="*80)
    print("Chunks质量分析")
    print("="*80)

    def analyze_chunks(chunks, label):
        """分析chunks质量"""
        # 统计不完整chunks（以句号等结尾）
        end_marks = ('。', '！', '？', '.', '!', '?')
        complete = sum(1 for c in chunks if c.page_content.rstrip()[-1:] in end_marks)
        incomplete = len(chunks) - complete

        # 统计不同长度分布
        lengths = [len(c.page_content) for c in chunks]
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0

        print(f"\n{label}:")
        print(f"  完整结尾: {complete}/{len(chunks)} ({complete/len(chunks)*100:.1f}%)")
        print(f"  不完整: {incomplete}/{len(chunks)} ({incomplete/len(chunks)*100:.1f}%)")
        print(f"  长度范围: {min_len} - {max_len} 字符")
        print(f"  平均长度: {sum(lengths)/len(lengths):.0f} 字符")

        # 显示前3个chunks的结尾
        print(f"  前3个Chunks结尾:")
        for i, chunk in enumerate(chunks[:3]):
            last_60 = chunk.page_content[-60:] if len(chunk.page_content) > 60 else chunk.page_content
            print(f"    Chunk {i+1}: ...{last_60}")

    analyze_chunks(chunks1, "【不合并】")
    analyze_chunks(chunks2, "【合并后】")

    # 检查是否有"通用航空"被切断的情况
    print("\n" + "="*80)
    print("关键词完整性检查（关键词：通用航空）")
    print("="*80)

    keyword = "通用航空"

    def check_keyword_in_chunks(chunks, label):
        """检查关键词是否被chunks切断"""
        cuts = 0
        for chunk in chunks:
            if keyword in chunk.page_content:
                # 检查关键词是否在chunk边界被切断
                text = chunk.page_content
                positions = []
                pos = 0
                while True:
                    pos = text.find(keyword, pos)
                    if pos == -1:
                        break
                    # 检查关键词前面是否被切断
                    if pos > 0 and not (text[pos-1] in [' ', '\n', '，', '。']):
                        cuts += 1
                    # 检查关键词后面是否被切断
                    end_pos = pos + len(keyword)
                    if end_pos < len(text) and not (text[end_pos] in [' ', '\n', '，', '。']):
                        cuts += 1
                    pos += len(keyword)

        print(f"\n{label}:")
        print(f"  包含关键词的Chunks: {sum(1 for c in chunks if keyword in c.page_content)}")
        print(f"  关键词被切断次数: {cuts}")

    check_keyword_in_chunks(chunks1, "【不合并】")
    check_keyword_in_chunks(chunks2, "【合并后】")

    # 保存详细结果
    output_file = "china_aviation_2021_test_result.json"
    result = {
        "file": str(txt_file.name),
        "file_size_kb": txt_file.stat().st_size / 1024,
        "original": {
            "chars": len(original_text),
            "lines": len(original_text.split('\n'))
        },
        "merged": {
            "chars": len(merged_text),
            "lines": len(merged_text.split('\n')),
            "lines_reduced": stats['lines_reduced'],
            "reduction_ratio": stats['reduction_ratio']
        },
        "chunks": {
            "without_merge": {
                "count": len(chunks1),
                "avg_length": sum(len(c.page_content) for c in chunks1) / len(chunks1),
                "total_chars": sum(len(c.page_content) for c in chunks1)
            },
            "with_merge": {
                "count": len(chunks2),
                "avg_length": sum(len(c.page_content) for c in chunks2) / len(chunks2),
                "total_chars": sum(len(c.page_content) for c in chunks2)
            }
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    print(f"详细结果已保存到: {output_file}")

    return result


if __name__ == "__main__":
    test_line_merge_on_document()
