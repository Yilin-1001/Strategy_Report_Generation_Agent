"""
简化的检索评估脚本 - 直接从chunks文件检索（不依赖Milvus）
使用BM25或简单的关键词匹配进行检索
"""
import sys
import json
from pathlib import Path
from typing import List, Dict
import re
from collections import Counter
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleRetriever:
    """简单的基于关键词匹配的检索器"""

    def __init__(self, chunks_file: str):
        """
        初始化检索器

        Args:
            chunks_file: chunks JSON文件路径
        """
        print(f"加载chunks文件: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data

        print(f"加载了 {len(self.chunks)} 个chunks")

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本
            top_n: 返回前N个关键词

        Returns:
            关键词列表
        """
        # 简单的分词（按空格和标点���
        words = re.findall(r'[\w]+', text.lower())

        # 过滤停用词（简单版）
        stopwords = {'的', '了', '是', '在', '和', '与', '或', '等', '及', '对', '为', '中', '有'}
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]

        # 统计词频
        word_freq = Counter(keywords)

        # 返回最常见的词
        return [w for w, _ in word_freq.most_common(top_n)]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索相关chunks

        Args:
            query: 查询问题
            top_k: 返回top-k结果

        Returns:
            检索结果列表
        """
        # 提取查询关键词
        query_keywords = self.extract_keywords(query, top_n=10)

        # 计算每个chunk的匹配分数
        scores = []
        for chunk in self.chunks:
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})

            # 计算匹配分数
            score = 0
            for keyword in query_keywords:
                if keyword in text.lower():
                    score += text.lower().count(keyword)

            # 如果有匹配，添加到结果
            if score > 0:
                scores.append({
                    'chunk': chunk,
                    'score': score,
                    'metadata': metadata
                })

        # 按分数排序，返回top-k
        scores.sort(key=lambda x: x['score'], reverse=True)

        # 格式化结果
        results = []
        for item in scores[:top_k]:
            chunk = item['chunk']
            results.append({
                'score': item['score'],
                'text': chunk.get('text', '')[:200],  # 前200字符
                'metadata': item['metadata'],
                'source': item['metadata'].get('source', ''),
                'doc_id': item['metadata'].get('document_id', '')
            })

        return results


class SimpleEvaluator:
    """简化的评估器"""

    def __init__(self, chunks_file: str):
        """初始化评估器"""
        print("初始化检索器...")
        self.retriever = SimpleRetriever(chunks_file)
        print("初始化完成\n")

    def evaluate_testset(
        self,
        testset_file: str,
        top_k: int = 5,
        max_questions: int = None
    ):
        """评估测试集"""
        # 加载测试集
        print(f"加载测试集: {testset_file}")
        with open(testset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data['questions']

        print(f"测试问题数: {len(questions)}")

        if max_questions:
            questions = questions[:max_questions]
            print(f"限制评估问题数: {max_questions}")

        print(f"\n开始评估...")
        print("="*80)

        # 运行检索
        results = []
        has_results_count = 0
        total_score = 0

        for i, item in enumerate(questions, 1):
            question = item['question']
            ground_truth = item['ground_truth']

            # 显示进度
            if i % 50 == 0 or i == 1:
                print(f"[{i}/{len(questions)}] 处理中...")

            # 执行检索
            search_results = self.retriever.search(question, top_k=top_k)

            # 记录结果
            top1_score = search_results[0]['score'] if search_results else 0
            total_score += top1_score

            if search_results:
                has_results_count += 1

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "num_retrieved": len(search_results),
                "top1_score": top1_score,
                "top1_source": search_results[0]['metadata'].get('source', '') if search_results else '',
                "retrieved_docs": search_results
            })

        # 计算统计指标
        print("\n" + "="*80)
        print("评估统计")
        print("="*80)

        total = len(results)
        avg_score = total_score / total if total > 0 else 0
        has_results = has_results_count
        high_score = sum(1 for r in results if r['top1_score'] >= 5)
        mid_score = sum(1 for r in results if 2 <= r['top1_score'] < 5)
        low_score = sum(1 for r in results if r['top1_score'] < 2)
        no_results = sum(1 for r in results if r['num_retrieved'] == 0)

        print(f"\n总问题数: {total}")
        print(f"有检索结果: {has_results} ({has_results/total*100:.1f}%)")
        print(f"无检索结果: {no_results} ({no_results/total*100:.1f}%)")
        print(f"平均Top-1分数: {avg_score:.2f}")

        print(f"\n分数分布:")
        print(f"  高分(>=5): {high_score} ({high_score/total*100:.1f}%)")
        print(f"  中分[2-5): {mid_score} ({mid_score/total*100:.1f}%)")
        print(f"  低分(<2): {low_score} ({low_score/total*100:.1f}%)")

        # 显示示例
        print("\n" + "="*80)
        print("检索示例（Top-3最好 + Top-3最差）")
        print("="*80)

        # 按分数排序
        results_with_score = [(r, r['top1_score']) for r in results if r['num_retrieved'] > 0]
        results_with_score.sort(key=lambda x: x[1], reverse=True)

        print("\n[OK] Top-3 最好结果:")
        for i, (result, score) in enumerate(results_with_score[:3], 1):
            print(f"\n{i}. 问题: {result['question'][:60]}...")
            print(f"   分数: {score}")
            print(f"   来源: {result['top1_source']}")
            print(f"   检索内容: {result['retrieved_docs'][0]['text'][:100]}...")

        if len(results_with_score) > 3:
            print("\n[FAIL] Top-3 最差结果:")
            for i, (result, score) in enumerate(results_with_score[-3:], 1):
                print(f"\n{i}. 问题: {result['question'][:60]}...")
                print(f"   分数: {score}")
                print(f"   来源: {result['top1_source']}")

        # 保存结果
        output_file = Path(testset_file).parent.parent / "experiments" / "simple_eval_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "statistics": {
                    "total_questions": total,
                    "has_results_rate": has_results / total if total > 0 else 0,
                    "avg_top1_score": avg_score,
                    "score_distribution": {
                        "high": high_score,
                        "mid": mid_score,
                        "low": low_score,
                        "none": no_results
                    }
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {output_file}")

        # 生成CSV
        import pandas as pd
        csv_file = output_file.parent / "simple_eval_results.csv"
        df = pd.DataFrame([{
            "question": r['question'],
            "num_retrieved": r['num_retrieved'],
            "top1_score": r['top1_score'],
            "top1_source": r['top1_source']
        } for r in results])
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV结果: {csv_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="简化检索评估（不依赖Milvus）")
    parser.add_argument(
        "--testset",
        type=str,
        default="rag_eval/evals/datasets/smart_testset_full.json",
        help="测试集文件路径"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="data/all_chunks_with_tags.json",
        help="Chunks文件路径"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索的文档数量"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="最大评估问题数（用于快速测试）"
    )

    args = parser.parse_args()

    # 检查文件
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"错误: 测试集文件不存在: {testset_path}")
        return 1

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"错误: Chunks文件不存在: {chunks_path}")
        return 1

    print("="*80)
    print("简化检索评估系统（基于关键词匹配）")
    print("="*80)
    print(f"测试集: {args.testset}")
    print(f"Chunks: {args.chunks}")
    print(f"Top-K: {args.top_k}")
    if args.max_questions:
        print(f"限制问题数: {args.max_questions}")
    print()

    # 创建评估器
    evaluator = SimpleEvaluator(str(chunks_path))

    # 运行评估
    evaluator.evaluate_testset(
        testset_file=str(testset_path),
        top_k=args.top_k,
        max_questions=args.max_questions
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
