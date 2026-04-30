"""
RAG 检索效果评估脚本（不生成��题）
专注于评估检索质量：召回率、精确率、MRR、NDCG 等
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from ragas.llms import llm_factory
from ragas import Dataset
from ragas.metrics import context_recall, context_precision

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_project.pipeline import RAGPipeline


# ================================
# API 配置（从 API_Setting.py）
# ================================
API_KEY = "c844d01915874aa38e5adfdcef05eee9.jZHyjygcan10qEqy"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"


class RetrievalEvaluator:
    """检索质量评估器"""

    def __init__(self, use_reranker: bool = False):
        """初始化评估器

        Args:
            use_reranker: 是否启用重排序
        """
        self.use_reranker = use_reranker
        print("初始化 RAG Pipeline...")
        self.pipeline = RAGPipeline(
            chunking_config_path="config/chunking_config.yaml",
            milvus_config_path="config/milvus_config.yaml"
        )
        print(f"重排序: {'启用' if use_reranker else '禁用'}")

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

    def retrieve_documents(self, question: str, top_k: int = 5) -> List[Dict]:
        """检索文档"""
        results = self.pipeline.search(question, top_k=top_k, use_reranker=self.use_reranker)
        return results

    def evaluate_retrieval_metrics(
        self,
        test_questions: List[Dict],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        使用 Ragas 评估检索质量

        Args:
            test_questions: 测试问题列表 [{"question": "...", "ground_truth": "..."}]
            top_k: 检索的文档数量

        Returns:
            评估指标结果
        """
        print("="*80)
        print("评估检索质量")
        print("="*80)
        print(f"测试问题数: {len(test_questions)}")
        print(f"Top-K: {top_k}")
        print()

        # 构建评估数据集
        dataset_data = []
        retrieval_results = []

        for i, item in enumerate(test_questions, 1):
            question = item['question']
            ground_truth = item['ground_truth']

            print(f"[{i}/{len(test_questions)}] 检索: {question}")

            # 执行检索
            results = self.retrieve_documents(question, top_k=top_k)
            contexts = [doc['text'] for doc in results]

            retrieval_results.append({
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_docs": results,
                "num_retrieved": len(results),
                "top1_score": results[0]['score'] if results else 0,
                "top1_source": results[0]['metadata'].get('source', '') if results else ''
            })

            # 构建简单答案（使用第一个检索结果）
            answer = contexts[0][:500] if contexts else "未找到相关文档"

            dataset_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })

            print(f"  检索到 {len(results)} 个文档")
            print(f"  Top-1 分数: {retrieval_results[-1]['top1_score']:.4f}")
            print()

        # 创建 Dataset
        dataset = Dataset.from_list(dataset_data)

        # 计算 Ragas 检索指标
        print("="*80)
        print("计算检索指标")
        print("="*80)

        metrics = {
            "context_recall": context_recall,
            "context_precision": context_precision
        }

        scores = {}
        for metric_name, metric in metrics.items():
            try:
                print(f"计算 {metric_name}...")
                score = metric.score(dataset, llm=self.ragas_llm)
                scores[metric_name] = score
                print(f"  {metric_name}: {score}")
            except Exception as e:
                print(f"  {metric_name} 失败: {e}")
                scores[metric_name] = None

        return {
            "scores": scores,
            "retrieval_results": retrieval_results,
            "dataset": dataset
        }

    def calculate_traditional_metrics(
        self,
        test_questions: List[Dict],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        计算传统检索指标（基于相关性的简化版本）

        Args:
            test_questions: 测试问题列表
            top_k: 检索的文档数量

        Returns:
            传统检索指标
        """
        print("\n" + "="*80)
        print("计算传统检索指标")
        print("="*80)

        total = len(test_questions)
        precision_at_k = []
        reciprocal_ranks = []

        for item in test_questions:
            question = item['question']

            # 执行检索
            results = self.retrieve_documents(question, top_k=top_k)

            if not results:
                precision_at_k.append(0)
                reciprocal_ranks.append(0)
                continue

            # 简化的相关性判断：基于相似度分数
            # 假设分数 >= 0.7 为相关
            relevant_count = sum(1 for r in results if r['score'] >= 0.7)
            precision_at_k.append(relevant_count / top_k)

            # Reciprocal Rank (第一个相关文档的倒数排名)
            rr = 0
            for i, r in enumerate(results, 1):
                if r['score'] >= 0.7:
                    rr = 1.0 / i
                    break
            reciprocal_ranks.append(rr)

        # 计算平均值
        avg_precision = sum(precision_at_k) / total if total > 0 else 0
        avg_rr = sum(reciprocal_ranks) / total if total > 0 else 0

        print(f"Precision@{top_k}: {avg_precision:.4f}")
        print(f"MRR (Mean Reciprocal Rank): {avg_rr:.4f}")

        return {
            f"precision_at_{top_k}": avg_precision,
            "mrr": avg_rr
        }


# 内置测试问题集
DEFAULT_TEST_QUESTIONS = [
    {
        "question": "江西省交通投资集团有限责任公司的主要职责是什么？",
        "ground_truth": "江西省交通投资集团有限责任公司主要负责江西省内交通基础设施的投资、建设和运营管理，包括高速公路、桥梁等交通项目的投资建设和运营管理。"
    },
    {
        "question": "什么是混合所有制改革？",
        "ground_truth": "混合所有制改革是指在国有资本中引入民间资本，实现股权多元化，通过国有资本、集体资本、非公有资本等交叉持股、相互融合的改革。"
    },
    {
        "question": "江西省交通运输厅的主要职能是什么？",
        "ground_truth": "江西省交通运输厅负责全省交通运输行业的行政管理，包括规划、建设、养护、管理等职能。"
    },
    {
        "question": "PPP模式在交通投资中的应用是什么？",
        "ground_truth": "PPP模式是政府与社会资本合作的模式，用于交通基础设施建设和公共服务，通过引入社会资本提高效率。"
    },
    {
        "question": "江西省交通投资集团的注册资本是多少？",
        "ground_truth": "江西省交通投资集团注册资本100亿元人民币。"
    }
]


def load_test_questions(file_path: str) -> List[Dict]:
    """从文件加载测试问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换格式
    questions = []
    for item in data:
        if "question" in item and "ground_truth" in item:
            questions.append(item)
        elif "question" in item and "answer_outline" in item:
            questions.append({
                "question": item["question"],
                "ground_truth": item["answer_outline"]
            })

    return questions


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索效果评估")
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="测试问题文件路径（JSON格式）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索的文档数量"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rag_eval/evals/experiments/retrieval_eval_results.json",
        help="结果输出路径"
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=False,
        help="启用重排序进行检索评估"
    )

    args = parser.parse_args()

    # 加载测试问题
    if args.questions and Path(args.questions).exists():
        print(f"从文件加载测试问题: {args.questions}")
        test_questions = load_test_questions(args.questions)
    else:
        print("使用内置测试问题集")
        test_questions = DEFAULT_TEST_QUESTIONS

    print(f"共 {len(test_questions)} 个测试问题\n")

    # 创建评估器
    evaluator = RetrievalEvaluator(use_reranker=args.use_reranker)

    # 运行 Ragas 检索指标评估
    results = evaluator.evaluate_retrieval_metrics(
        test_questions=test_questions,
        top_k=args.top_k
    )

    # 计算传统检索指标
    traditional_metrics = evaluator.calculate_traditional_metrics(
        test_questions=test_questions,
        top_k=args.top_k
    )

    # 汇总结果
    final_results = {
        "ragas_metrics": results["scores"],
        "traditional_metrics": traditional_metrics,
        "retrieval_details": results["retrieval_results"],
        "config": {
            "top_k": args.top_k,
            "num_questions": len(test_questions)
        }
    }

    # 显示结果摘要
    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)

    print("\nRagas 指标:")
    for metric_name, score in results["scores"].items():
        if score is not None:
            print(f"  {metric_name}: {score:.4f}")

    print("\n传统检索指标:")
    for metric_name, score in traditional_metrics.items():
        print(f"  {metric_name}: {score:.4f}")

    # 显示检索详情
    print("\n" + "="*80)
    print("检索详情")
    print("="*80)

    for i, detail in enumerate(results["retrieval_results"], 1):
        print(f"\n[{i}] 问题: {detail['question']}")
        print(f"    检索到: {detail['num_retrieved']} 个文档")
        print(f"    Top-1 分数: {detail['top1_score']:.4f}")
        print(f"    Top-1 来源: {detail['top1_source']}")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # 转换 numpy 类型为 Python 类型
        save_data = {
            "ragas_metrics": {
                k: float(v) if v is not None else None
                for k, v in results["scores"].items()
            },
            "traditional_metrics": {
                k: float(v) for k, v in traditional_metrics.items()
            },
            "retrieval_details": results["retrieval_results"],
            "config": final_results["config"]
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")

    # 生成 CSV 格式
    import pandas as pd
    csv_path = output_path.parent / "retrieval_eval_results.csv"
    df = pd.DataFrame(results["retrieval_details"])
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 结果已保存到: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
