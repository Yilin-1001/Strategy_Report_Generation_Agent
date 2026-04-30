"""
Ragas 评估 - 并行优化版本 (10并发)
使用 ThreadPoolExecutor 实现并行处理，大幅加速评估

预期加速效果: 464问题从3小时 → 约20分钟
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datasets import Dataset as HFDataset
from openai import OpenAI
import threading

# 切换到项目根目录
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from rag_project.pipeline import RAGPipeline


# ================================
# API 配置
# ================================
API_KEY = "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ================================
# 并行配置
# ================================
MAX_WORKERS = 10  # 并发线程数
BATCH_SIZE = 50   # 每批处理的问题数
TIMEOUT = 120     # 单个评估超时时间（秒）

# 线程锁（用于打印输出）
print_lock = threading.Lock()


def log(message):
    """线程安全的日志打印"""
    with print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def evaluate_single_question(item, pipeline, llm):
    """
    评估单个问题的两个指标

    Args:
        item: 包含question, ground_truth的字典
        pipeline: RAGPipeline实例
        llm: Ragas LLM实例

    Returns:
        dict: 评估结果
    """
    question = item['question']
    ground_truth = item['ground_truth']
    question_id = item.get('id', '')

    try:
        # 执行检索
        search_results = pipeline.search(question, top_k=3)
        contexts = [doc['text'] for doc in search_results]

        if not contexts:
            return {
                'id': question_id,
                'question': question,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'error': 'No contexts found'
            }

        # 构建答案（使用第一个检索结果）
        answer = contexts[0][:500] if contexts else "未找到相关文档"

        # 使用 Ragas 评估单个样本
        from ragas.run import run
        from ragas.metrics import context_recall, context_precision

        # 创建单样本数据集
        sample_dataset = HFDataset.from_list([{
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }])

        # 评估
        result = run(
            dataset=sample_dataset,
            metrics=[context_precision, context_recall],
            llm=llm,
            timeout=TIMEOUT,
            raise_exceptions=True
        )

        # 转换为字典
        result_dict = result.to_pandas().to_dict('records')[0]

        return {
            'id': question_id,
            'question': question,
            'context_precision': float(result_dict.get('context_precision', 0)),
            'context_recall': float(result_dict.get('context_recall', 0)),
            'ground_truth': ground_truth
        }

    except Exception as e:
        return {
            'id': question_id,
            'question': question,
            'context_precision': 0.0,
            'context_recall": 0.0,
            'error': str(e)
        }


def evaluate_batch(batch_items, pipeline, llm, batch_num):
    """
    评估一批问题

    Args:
        batch_items: 问题列表
        pipeline: RAGPipeline实例
        llm: Ragas LLM实例
        batch_num: 批次号

    Returns:
        list: 评估结果列表
    """
    results = []

    log(f"批次 {batch_num}: 开始评估 {len(batch_items)} 个问题...")
    start_time = time.time()

    # 并行评估
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(evaluate_single_question, item, pipeline, llm): item
            for item in batch_items
        }

        # 收集结果
        completed = 0
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result(timeout=TIMEOUT + 10)
                results.append(result)
                completed += 1

                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = completed / elapsed
                    eta = (len(batch_items) - completed) / speed
                    log(f"批次 {batch_num}: 已完成 {completed}/{len(batch_items)} ({completed/len(batch_items)*100:.1f}%), "
                        f"速度: {speed:.2f}个/秒, 预计剩余: {eta:.0f}秒")

            except Exception as e:
                log(f"批次 {batch_num}: 项目失败 - {e}")
                results.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'context_precision': 0.0,
                    'context_recall': 0.0,
                    'error': str(e)
                })

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if 'error' not in r)

    log(f"批次 {batch_num} 完成: {success_count}/{len(batch_items)} 成功, "
        f"耗时: {elapsed:.1f}秒 ({len(batch_items)/elapsed:.2f}个/秒)")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Ragas 并行评估 (10并发)")
    parser.add_argument("--testset", type=str,
                        default="rag_eval/evals/datasets/smart_testset_full.json",
                        help="测试集文件路径")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索的文档数量")
    parser.add_argument("--output", type=str,
                        default="rag_eval/evals/experiments/ragas_parallel_results.json",
                        help="结果输出路径")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"每批处理的问题数 (默认: {BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"并发线程数 (默认: {MAX_WORKERS})")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续")

    args = parser.parse_args()

    # 更新全局配置
    global BATCH_SIZE, MAX_WORKERS
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.workers

    print("="*80)
    print("Ragas 并行评估 - 10并发版本")
    print("="*80)
    print(f"测试集: {args.testset}")
    print(f"Top-K: {args.top_k}")
    print(f"并发数: {MAX_WORKERS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"API: {BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print()

    # 检查文件
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"错误: 测试集文件不存在: {testset_path}")
        return 1

    # 加载测试集
    log("加载测试集...")
    with open(testset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_questions = data['questions']

    total_questions = len(all_questions)
    log(f"测试问题总数: {total_questions}")

    # 添加ID到问题中
    for i, q in enumerate(all_questions):
        q['id'] = i

    # 分批处理
    num_batches = (total_questions + BATCH_SIZE - 1) // BATCH_SIZE
    batches = []
    for i in range(0, total_questions, BATCH_SIZE):
        batch = all_questions[i:i+BATCH_SIZE]
        batches.append((i, batch))

    log(f"分 {num_batches} 批处理, 每批 {BATCH_SIZE} 个问题")

    # 初始化 Pipeline
    log("初始化 RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml"
    )
    log("Pipeline 初始化完成\n")

    # 创建 OpenAI 客户端
    log("创建 DeepSeek API 客户端...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 创建 Ragas LLM 实例
    log("创建 Ragas LLM 实例...")
    from ragas.llms import llm_factory
    ragas_llm = llm_factory(
        model=MODEL_NAME,
        provider="openai",
        client=client,
        max_tokens=8192,
        timeout=TIMEOUT
    )
    log(f"LLM 实例创建成功: {MODEL_NAME}\n")

    # 检查是否恢复
    resume_from_batch = 0
    completed_results = []

    if args.resume:
        # 检查是否有之前的中间结果
        output_path = Path(args.output)
        resume_file = output_path.parent / "ragas_checkpoint.json"

        if resume_file.exists():
            log(f"发现检查点文件: {resume_file}")
            try:
                with open(resume_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                resume_from_batch = checkpoint.get('last_completed_batch', 0)
                completed_results = checkpoint.get('completed_results', [])
                log(f"从批次 {resume_from_batch} 恢复, 已完成 {len(completed_results)} 个问题")
            except Exception as e:
                log(f"无法读取检查点: {e}, 将从头开始")

    # 开始评估
    log("="*80)
    log("开始并行评估")
    log(f"预计总时间: {total_questions * 22 / MAX_WORKERS / 60:.1f} 分钟 (基于 11秒/问题)")
    log("="*80)
    log()

    all_results = []
    start_time = time.time()

    # 逐批评估
    for batch_idx, (start_idx, batch) in enumerate(batches):
        if batch_idx < resume_from_batch:
            log(f"批次 {batch_idx}: 跳过 (已恢复)")
            continue

        batch_num = batch_idx + 1
        batch_results = evaluate_batch(batch, pipeline, ragas_llm, batch_num)

        # 添加到总结果
        all_results.extend(batch_results)

        # 保存检查点
        checkpoint = {
            'last_completed_batch': batch_num,
            'completed_results': all_results,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = Path(args.output).parent / "ragas_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        log(f"检查点已保存: 批次 {batch_num}/{num_batches}")
        log()

    # 计算总时间
    total_elapsed = time.time() - start_time

    # 处理结果
    log("="*80)
    log("评估完成！")
    log("="*80)
    log(f"总耗时: {total_elapsed/60:.1f} 分钟 ({total_elapsed:.0f} 秒)")
    log(f"评估问题数: {len(all_results)}")
    log(f"成功率: {sum(1 for r in all_results if 'error' not in r)/len(all_results)*100:.1f}%")
    log()

    # 计算平均分数
    valid_precision = [r['context_precision'] for r in all_results if 'error' not in r]
    valid_recall = [r['context_recall'] for r in all_results if 'error' not in r]

    if valid_precision:
        avg_precision = sum(valid_precision) / len(valid_precision)
        log(f"Context Precision: {avg_precision:.4f} ({len(valid_precision)}个有效样本)")
    else:
        log("Context Precision: 无有效数据")
        avg_precision = 0.0

    if valid_recall:
        avg_recall = sum(valid_recall) / len(valid_recall)
        log(f"Context Recall: {avg_recall:.4f} ({len(valid_recall)}个有效样本)")
    else:
        log("Context Recall: 无有效数据")
        avg_recall = 0.0

    # 计算 F1 分数
    if avg_precision > 0 or avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        log(f"F1 Score: {f1:.4f}")
    else:
        f1 = 0.0

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    final_results = {
        'metrics': {
            'context_precision': avg_precision,
            'context_recall': avg_recall,
            'f1_score': f1
        },
        'testset': str(args.testset),
        'top_k': args.top_k,
        'num_samples': len(all_results),
        'successful_samples': len(valid_precision),
        'parallel_workers': MAX_WORKERS,
        'batch_size': BATCH_SIZE,
        'total_time_seconds': total_elapsed,
        'avg_time_per_question': total_elapsed / len(all_results),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    log(f"结果已保存: {output_path}")

    # 保存详细结果
    detailed_path = output_path.parent / "ragas_parallel_detailed.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    log(f"详细结果: {detailed_path}")

    # 保存错误报告
    errors = [r for r in all_results if 'error' in r]
    if errors:
        error_path = output_path.parent / "ragas_errors.csv"
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(error_path, index=False, encoding='utf-8-sig')
        log(f"错误报告: {error_path} ({len(errors)}个错误)")

    log()
    log("="*80)
    log("评估总结")
    log("="*80)
    log(f"Context Precision: {avg_precision:.4f}")
    log(f"Context Recall:    {avg_recall:.4f}")
    log(f"F1 Score:          {f1:.4f}")
    log()
    log(f"总耗时:            {total_elapsed/60:.1f} 分钟")
    log(f"平均速度:          {len(all_results)/total_elapsed:.2f} 个问题/秒")
    log(f"加速倍数:          {464*22/total_elapsed:.1f}x (相比串行{464*22/60:.1f}分钟)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
