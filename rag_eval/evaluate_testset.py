"""
使用生成的测试集评估 RAG 检索效果
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from ragas.llms import llm_factory

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_project.pipeline import RAGPipeline


# ================================
# API 配置
# ================================
API_KEY = "c844d01915874aa38e5adfdcef05eee9.jZHyjygcan10qEqy"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"


class RetrievalEvaluator:
    """检索评估器"""

    def __init__(self):
        """初始化评估器"""
        print("初始化 RAG Pipeline...")
        self.pipeline = RAGPipeline(
            chunking_config_path="config/chunking_config.yaml",
            milvus_config_path="config/milvus_config.yaml"
        )

        print("创建 GLM 客户端...")
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        print("创建 Ragas LLM...")
        self.ragas_llm = llm_factory(
            MODEL_NAME,
            provider="openai",
            client=self.client
        )
        print("初始化完成\n")

    def evaluate_testset(
        self,
        testset_file: str,
        top_k: int = 5
    ):
        """评估测试集"""
        # 加载测试集
        print(f"加载测试集: {testset_file}")
        with open(testset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data['questions']

        print(f"测试问题数: {len(questions)}\n")

        # 运行检索
        results = []
        for i, item in enumerate(questions, 1):
            question = item['question']
            ground_truth = item['ground_truth']

            print(f"[{i}/{len(questions)}] {question[:50]}...")

            # 执行检索
            search_results = self.pipeline.search(question, top_k=top_k)

            # 记录结果
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "num_retrieved": len(search_results),
                "top1_score": search_results[0]['score'] if search_results else 0,
                "top1_source": search_results[0]['metadata'].get('source', '') if search_results else '',
                "retrieved_docs": search_results
            })

            print(f"  检索到 {len(search_results)} 个文档")
            if search_results:
                print(f"  Top-1: {search_results[0]['score']:.4f}")

        # 计算统计指标
        print("\n" + "="*80)
        print("评估统计")
        print("="*80)

        total = len(results)
        avg_score = sum(r['top1_score'] for r in results) / total if total > 0 else 0
        has_results = sum(1 for r in results if r['num_retrieved'] > 0)
        high_score = sum(1 for r in results if r['top1_score'] >= 0.8)
        mid_score = sum(1 for r in results if 0.6 <= r['top1_score'] < 0.8)
        low_score = sum(1 for r in results if r['top1_score'] < 0.6)

        print(f"\n总问题数: {total}")
        print(f"有检索结果: {has_results} ({has_results/total*100:.1f}%)")
        print(f"平均 Top-1 分数: {avg_score:.4f}")

        print(f"\n分数分布:")
        print(f"  高分(>=0.8): {high_score} ({high_score/total*100:.1f}%)")
        print(f"  中分[0.6-0.8): {mid_score} ({mid_score/total*100:.1f}%)")
        print(f"  低分(<0.6): {low_score} ({low_score/total*100:.1f}%)")

        # 保存结果
        output_file = Path(testset_file).parent.parent / "experiments" / "retrieval_eval_results.json"
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
                        "low": low_score
                    }
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {output_file}")

        # 生成 CSV
        import pandas as pd
        csv_file = output_file.parent / "retrieval_eval_results.csv"
        df = pd.DataFrame([{
            "question": r['question'],
            "num_retrieved": r['num_retrieved'],
            "top1_score": r['top1_score'],
            "top1_source": r['top1_source']
        } for r in results])
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV 结果: {csv_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估生成的测试集")
    parser.add_argument(
        "--testset",
        type=str,
        default="rag_eval/evals/datasets/ragas_testset_doc_51q.json",
        help="测试集文件路径"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索的文档数量"
    )

    args = parser.parse_args()

    # 检查文件
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"错误: 测试集文件不存在: {testset_path}")
        return 1

    # 创建评估器
    evaluator = RetrievalEvaluator()

    # 运行评估
    evaluator.evaluate_testset(
        testset_file=args.testset,
        top_k=args.top_k
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
