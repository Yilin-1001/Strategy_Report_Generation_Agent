"""
快速评估版本 - 只处理前50个问题
用于快速验证Ragas评估是否正常工作
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset as HFDataset
from openai import OpenAI

# 切换到项目根目录（解决绝对路径运行的问题）
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# 添加项目路径
sys.path.insert(0, str(PROJECT_ROOT))
from rag_project.pipeline import RAGPipeline


# ================================
# API 配置
# ================================
API_KEY = "c844d01915874aa38e5adfdcef05eee9.jZHyjygcan10qEqy"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Ragas快速评估（50问版）")
    parser.add_argument(
        "--testset",
        type=str,
        default="rag_eval/evals/datasets/smart_testset_full.json",
        help="测试集文件路径"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="评估的问题数量"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索的文档数量"
    )

    args = parser.parse_args()

    print("="*80)
    print(f"Ragas快速评估 - {args.num_questions}问版")
    print("="*80)
    print(f"测试集: {args.testset}")
    print(f"评估问题数: {args.num_questions}")
    print(f"Top-K: {args.top_k}")
    print()

    # 检查文件
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"错误: 测试集文件不存在: {testset_path}")
        return 1

    # 初始化 Pipeline
    print("初始化 RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml"
    )
    print("Pipeline 初始化完成\n")

    # 加载测试集
    print(f"加载测试集: {args.testset}")
    with open(args.testset, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_questions = data['questions']

    # 限制问题数量
    questions = all_questions[:args.num_questions]
    print(f"总问题数: {len(all_questions)}")
    print(f"评估问题数: {len(questions)}\n")

    # 构建 Ragas 格式的数据
    dataset_data = []

    print("构建 Ragas 数据集...")
    for i, item in enumerate(questions, 1):
        question = item['question']
        ground_truth = item['ground_truth']

        # 执行检索
        search_results = pipeline.search(question, top_k=args.top_k)
        contexts = [doc['text'] for doc in search_results]

        # 构建答案（使用第一个检索结果）
        answer = contexts[0][:500] if contexts else "未找到相关文档"

        dataset_data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

        if i % 10 == 0:
            print(f"  已处理 {i}/{len(questions)}")

    # 创建 HuggingFace Dataset
    hf_dataset = HFDataset.from_list(dataset_data)
    print(f"\n数据集创建完成: {len(hf_dataset)} 个样本")

    # 使用 Ragas evaluate 函数
    print("\n" + "="*80)
    print("使用 Ragas 框架计算指标")
    print("="*80)

    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
    )
    from ragas.llms import llm_factory

    # 创建 OpenAI 客户端（连接到 GLM API）
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 创建 Ragas LLM 实例
    print("\n创建 Ragas LLM 实例...")
    ragas_llm = llm_factory(
        model=MODEL_NAME,
        provider="openai",
        client=client,
        max_tokens=8192
    )
    print(f"LLM 实例创建成功: {MODEL_NAME} (max_tokens=8192)")

    # 定义指标
    metrics = [
        context_precision,
        context_recall
    ]

    print(f"\n评估指标:")
    for m in metrics:
        print(f"  - {type(m).__name__}")

    print(f"\n开始评估 {args.num_questions} 个问题（预计需要5-10分钟）...")

    try:
        # 运行评估
        result = evaluate(
            dataset=hf_dataset,
            metrics=metrics,
            llm=ragas_llm,
            raise_exceptions=True
        )

        # 转换为 DataFrame
        result_df = result.to_pandas()

        # 计算平均值
        scores = {}
        print("\n评估结果:")

        for metric_name in ['context_precision', 'context_recall']:
            if metric_name in result_df.columns:
                valid_scores = result_df[metric_name].dropna()
                if len(valid_scores) > 0:
                    mean_score = valid_scores.mean()
                    scores[metric_name] = float(mean_score)
                    print(f"  {metric_name}: {mean_score:.4f}")
                    print(f"    (样本数: {len(valid_scores)}, 范围: {valid_scores.min():.4f} - {valid_scores.max():.4f})")

        # 计算 F1 分数
        if scores.get("context_precision") and scores.get("context_recall"):
            precision = scores["context_precision"]
            recall = scores["context_recall"]
            f1 = 2 * (precision * recall) / (precision + recall)
            scores["f1_score"] = f1
            print(f"  f1_score: {f1:.4f}")

        # 保存详细结果
        output_dir = Path("rag_eval/evals/experiments")
        output_dir.mkdir(parents=True, exist_ok=True)

        result_csv_path = output_dir / f"ragas_quick_{args.num_questions}q_results.csv"
        result_df.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存: {result_csv_path}")

    except Exception as e:
        print(f"\n评估失败: {e}")
        import traceback
        traceback.print_exc()
        scores = {}

    # 创建输出目录
    output_dir = Path("rag_eval/evals/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 显示结果
    print("\n" + "="*80)
    print("Ragas 评估结果")
    print("="*80)

    print("\n检索质量指标:")
    if scores:
        for metric_name, score in scores.items():
            if score is not None:
                print(f"  {metric_name}: {score:.4f}")
    else:
        print("  未能计算指标")

    # 保存结果
    output_path = output_dir / f"ragas_metrics_{args.num_questions}q.json"

    results_data = {
        "metrics": scores,
        "testset": str(args.testset),
        "top_k": args.top_k,
        "num_samples": len(hf_dataset),
        "total_questions_in_testset": len(all_questions),
        "ragas_version": "0.4.3",
        "model": MODEL_NAME
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_path}")

    # 生成 CSV
    csv_path = output_dir / f"ragas_metrics_{args.num_questions}q.csv"
    df_scores = pd.DataFrame([scores])
    df_scores.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV 结果: {csv_path}")

    print("\n" + "="*80)
    print("快速评估完成！")
    print(f"评估了 {args.num_questions} 个问题")
    print(f"如果满意，可以运行完整评估：")
    print(f"  python rag_eval/ragas_glm_fixed.py")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
