"""
使用 Ragas Faithfulness 指标评估 RAG 生成内容的忠实度。

Faithfulness 衡量：生成的答案是否完全基于检索到的上下文，而非凭空编造。

流程：
1. 加载测试集问题
2. 通过 RAG Pipeline 检索上下文
3. 用 LLM 基于上下文生成答案
4. 用 ragas faithfulness 指标评估答案忠实度
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from datasets import Dataset as HFDataset
from openai import OpenAI

# 切换到项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.pipeline import RAGPipeline

# ================================
# API 配置
# ================================
# 用于生成答案的 LLM
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 用于 ragas 评估的 LLM（可以与生成 LLM 不同）
EVAL_API_KEY = os.environ.get(
    "EVAL_API_KEY",
    "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
)
EVAL_BASE_URL = "https://api.deepseek.com"
EVAL_MODEL = "deepseek-chat"


# ================================
# 答案生成 Prompt
# ================================
ANSWER_PROMPT = """你是一个专业的问答助手。请仅根据下面提供的参考资料来回答问题。
如果参考资料中没有足够的信息来回答问题，请明确说明"根据现有资料无法完全回答该问题"。
不要编造任何参考资料中没有的信息。

## 参考资料
{contexts}

## 问题
{question}

## 回答
"""


def generate_answer(
    client: OpenAI,
    question: str,
    contexts: List[str],
    max_retries: int = 3
) -> str:
    """使用 LLM 基于检索上下文生成答案"""
    contexts_text = "\n\n---\n\n".join(
        f"[文档{i+1}]: {ctx}" for i, ctx in enumerate(contexts)
    )
    prompt = ANSWER_PROMPT.format(contexts=contexts_text, question=question)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    重试 ({attempt+1}/{max_retries})，等待 {wait}s... 错误: {e}")
                time.sleep(wait)
            else:
                print(f"    生成答案失败: {e}")
                return "生成失败"


def build_eval_dataset(
    pipeline: RAGPipeline,
    questions: List[Dict],
    gen_client: OpenAI,
    top_k: int = 5,
    start_idx: int = 0,
) -> List[Dict]:
    """
    构建评估数据集：检索上下文 → 生成答案 → 收集数据

    Returns:
        List of dicts with keys: question, answer, contexts
    """
    dataset_data = []
    total = len(questions)

    for i, item in enumerate(questions):
        idx = start_idx + i + 1
        question = item["question"]

        print(f"  [{idx}/{start_idx + total}] {question[:50]}...")

        # Step 1: 检索上下文
        search_results = pipeline.search(question, top_k=top_k)
        contexts = [doc["text"] for doc in search_results]

        if not contexts:
            print(f"    ⚠ 未检索到相关文档，跳过")
            continue

        # Step 2: LLM 生成答案
        answer = generate_answer(gen_client, question, contexts)

        dataset_data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
        })

        # 简单限流
        if (i + 1) % 10 == 0:
            time.sleep(1)

    return dataset_data


def run_faithfulness_eval(
    dataset_data: List[Dict],
    output_dir: Path,
    top_k: int,
    testset_name: str,
) -> Dict:
    """运行 ragas faithfulness 评估"""
    from ragas import evaluate
    from ragas.metrics import faithfulness
    from ragas.llms import llm_factory

    # 创建 HuggingFace Dataset
    hf_dataset = HFDataset.from_list(dataset_data)
    print(f"\n评估数据集: {len(hf_dataset)} 个样本")

    # 创建 ragas LLM 实例
    eval_client = OpenAI(api_key=EVAL_API_KEY, base_url=EVAL_BASE_URL)
    ragas_llm = llm_factory(
        model=EVAL_MODEL,
        provider="openai",
        client=eval_client,
        max_tokens=8192,
    )
    print(f"评估 LLM: {EVAL_MODEL}")

    # 运行评估
    print(f"\n开始 Faithfulness 评估...")
    start_time = time.time()

    result = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        raise_exceptions=True,
    )

    elapsed = time.time() - start_time
    print(f"评估耗时: {elapsed:.1f}s")

    # 处理结果
    result_df = result.to_pandas()
    valid_scores = result_df["faithfulness"].dropna()

    scores = {}
    if len(valid_scores) > 0:
        scores["faithfulness"] = float(valid_scores.mean())
        print(f"\n{'='*60}")
        print(f"Faithfulness: {scores['faithfulness']:.4f}")
        print(f"  样本数: {len(valid_scores)}")
        print(f"  最小值: {valid_scores.min():.4f}")
        print(f"  最大值: {valid_scores.max():.4f}")
        print(f"  中位数: {valid_scores.median():.4f}")
        print(f"{'='*60}")

    # 保存详细结果 CSV
    result_csv = output_dir / "faithfulness_detailed_results.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f"详细结果: {result_csv}")

    # 保存摘要 JSON
    summary = {
        "metrics": scores,
        "testset": testset_name,
        "top_k": top_k,
        "num_samples": len(dataset_data),
        "gen_model": DEEPSEEK_MODEL,
        "eval_model": EVAL_MODEL,
        "elapsed_seconds": round(elapsed, 1),
        "ragas_version": "0.4.3",
    }
    summary_path = output_dir / "faithfulness_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"评估摘要: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Ragas Faithfulness 评估 - 检测生成内容是否忠实于检索上下文"
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="rag_eval/evals/datasets/smart_testset_full.json",
        help="测试集文件路径",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="评估问题数 (默认20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索上下文数量 (默认5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rag_eval/evals/experiments",
        help="结果输出目录",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从断点恢复 (提供 checkpoint 文件路径)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Ragas Faithfulness 评估")
    print("=" * 60)
    print(f"测试集: {args.testset}")
    print(f"问题数: {args.num_questions}")
    print(f"Top-K:  {args.top_k}")
    print(f"生成模型: {DEEPSEEK_MODEL}")
    print(f"评估模型: {EVAL_MODEL}")
    print()

    # ---- 初始化 Pipeline ----
    print("初始化 RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
    )
    print("Pipeline 就绪\n")

    # ---- 加载测试集 ----
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"错误: 测试集不存在 - {testset_path}")
        return 1

    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions = data["questions"]
    questions = all_questions[: args.num_questions]
    print(f"加载测试集: {len(all_questions)} 题, 取前 {len(questions)} 题\n")

    # ---- 创建生成答案的 LLM 客户端 ----
    gen_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    # ---- 构建评估数据集 ----
    print("-" * 60)
    print("Step 1/2: 检索 + 生成答案")
    print("-" * 60)

    if args.resume:
        # 从断点恢复
        with open(args.resume, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        dataset_data = checkpoint["data"]
        print(f"从断点恢复: {len(dataset_data)} 个样本")
    else:
        dataset_data = build_eval_dataset(
            pipeline=pipeline,
            questions=questions,
            gen_client=gen_client,
            top_k=args.top_k,
        )

    if not dataset_data:
        print("错误: 没有生成任何评估数据")
        return 1

    # 保存断点
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "faithfulness_checkpoint.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"data": dataset_data}, f, ensure_ascii=False, indent=2)
    print(f"断点已保存: {checkpoint_path}")

    # ---- 运行 Faithfulness 评估 ----
    print(f"\n{'-'*60}")
    print("Step 2/2: Ragas Faithfulness 评估")
    print("-" * 60)

    summary = run_faithfulness_eval(
        dataset_data=dataset_data,
        output_dir=output_dir,
        top_k=args.top_k,
        testset_name=str(args.testset),
    )

    print(f"\n{'='*60}")
    print("评估完成!")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())