"""
纯检索测试脚本 - 测试RAG系统的检索质量
加载问题集，执行检索，生成评估报告
"""
import json
import time
from pathlib import Path
from typing import List, Dict
from rag_project.pipeline import RAGPipeline


class RetrievalTester:
    """检索测试器"""

    def __init__(
        self,
        pipeline: RAGPipeline,
        questions_path: str,
        output_path: str = "retrieval_test_report.txt"
    ):
        """
        初始化测试器

        Args:
            pipeline: RAG Pipeline实例
            questions_path: 测试问题JSON文件路径
            output_path: 测试报告输出路径
        """
        self.pipeline = pipeline
        self.questions_path = questions_path
        self.output_path = output_path
        self.questions = self._load_questions()
        self.results = []

    def _load_questions(self) -> List[Dict]:
        """加载测试问题"""
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_test(self, top_k: int = 5):
        """
        运行检索测试

        Args:
            top_k: 每个问题检索的top-k结果数量
        """
        print("="*80)
        print("开始检索测试")
        print("="*80)
        print(f"问题数量: {len(self.questions)}")
        print(f"Top-K: {top_k}")
        print()

        total_time = 0

        for idx, question_item in enumerate(self.questions, 1):
            question = question_item['question']
            q_type = question_item['type']

            print(f"[{idx}/{len(self.questions)}] 测试问题: {question}")

            # 执行检索
            start_time = time.time()
            search_results = self.pipeline.search(question, top_k=top_k)
            elapsed = time.time() - start_time
            total_time += elapsed

            # 记录结果
            result_item = {
                "question": question,
                "type": q_type,
                "elapsed_time": elapsed,
                "num_results": len(search_results),
                "results": search_results,
                "top1_score": search_results[0]['score'] if search_results else 0,
                "top1_source": search_results[0]['metadata'].get('source', '') if search_results else ''
            }
            self.results.append(result_item)

            # 显示结果
            if search_results:
                print(f"  [OK] 找到 {len(search_results)} 个结果")
                print(f"  Top-1 评分: {search_results[0]['score']:.4f}")
                print(f"  Top-1 来源: {search_results[0]['metadata'].get('source', 'unknown')}")
            else:
                print(f"  [FAIL] 未找到结果")

            print(f"  耗时: {elapsed:.3f}秒")
            print()

        # 统计信息
        print("="*80)
        print("测试完成")
        print("="*80)
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均耗时: {total_time/len(self.questions):.3f}秒/问题")

    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*80)
        print("结果分析")
        print("="*80)

        # 1. 基本统计
        total_questions = len(self.results)
        no_result_count = sum(1 for r in self.results if r['num_results'] == 0)
        has_result_count = total_questions - no_result_count

        print(f"\n基本统计:")
        print(f"  总问题数: {total_questions}")
        print(f"  有结果: {has_result_count} ({has_result_count/total_questions*100:.1f}%)")
        print(f"  无结果: {no_result_count} ({no_result_count/total_questions*100:.1f}%)")

        # 2. 评分分布
        scores = [r['top1_score'] for r in self.results if r['num_results'] > 0]

        if scores:
            print(f"\nTop-1评分分布:")
            print(f"  最高分: {max(scores):.4f}")
            print(f"  最低分: {min(scores):.4f}")
            print(f"  平均分: {sum(scores)/len(scores):.4f}")
            print(f"  中位数: {sorted(scores)[len(scores)//2]:.4f}")

            # 评分区间
            high_score = sum(1 for s in scores if s >= 0.8)
            mid_score = sum(1 for s in scores if 0.6 <= s < 0.8)
            low_score = sum(1 for s in scores if s < 0.6)

            print(f"\n评分区间:")
            print(f"  高分(≥0.8): {high_score} ({high_score/len(scores)*100:.1f}%)")
            print(f"  中分[0.6-0.8): {mid_score} ({mid_score/len(scores)*100:.1f}%)")
            print(f"  低分(<0.6): {low_score} ({low_score/len(scores)*100:.1f}%)")

        # 3. 问题类型分析
        type_stats = {}
        for r in self.results:
            qtype = r['type']
            if qtype not in type_stats:
                type_stats[qtype] = {'count': 0, 'no_result': 0, 'scores': []}

            type_stats[qtype]['count'] += 1
            if r['num_results'] == 0:
                type_stats[qtype]['no_result'] += 1
            else:
                type_stats[qtype]['scores'].append(r['top1_score'])

        print(f"\n问题类型分析:")
        for qtype, stats in sorted(type_stats.items(), key=lambda x: -x[1]['count']):
            count = stats['count']
            no_result = stats['no_result']
            scores = stats['scores']

            print(f"\n  {qtype}:")
            print(f"    数量: {count}")
            print(f"    无结果: {no_result} ({no_result/count*100:.1f}%)")
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"    平均分: {avg_score:.4f}")

        # 4. 速度分析
        times = [r['elapsed_time'] for r in self.results]
        print(f"\n响应速度:")
        print(f"  平均: {sum(times)/len(times):.3f}秒")
        print(f"  最快: {min(times):.3f}秒")
        print(f"  最慢: {max(times):.3f}秒")

        return {
            'total_questions': total_questions,
            'has_result_rate': has_result_count / total_questions if total_questions > 0 else 0,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'avg_time': sum(times) / len(times) if times else 0,
            'type_stats': type_stats
        }

    def show_examples(self, n: int = 5):
        """展示测试示例"""
        print("\n" + "="*80)
        print(f"测试示例（前{n}个）")
        print("="*80)

        for i, result in enumerate(self.results[:n], 1):
            print(f"\n示例 {i}:")
            print(f"  问题: {result['question']}")
            print(f"  类型: {result['type']}")

            if result['results']:
                print(f"  Top-1结果:")
                top1 = result['results'][0]
                print(f"    评分: {top1['score']:.4f}")
                print(f"    来源: {top1['metadata'].get('source', 'unknown')}")
                print(f"    内容: {top1['text'][:100]}...")
            else:
                print(f"  [FAIL] 未找到结果")

    def generate_report(self, stats: Dict):
        """生成详细报告"""
        report_lines = []

        report_lines.append("="*80)
        report_lines.append("RAG系统检索质量测试报告")
        report_lines.append("="*80)
        report_lines.append("")

        # 1. 测试概要
        report_lines.append("一、测试概要")
        report_lines.append("-"*80)
        report_lines.append(f"测试问题数量: {stats['total_questions']}")
        report_lines.append(f"检索成功率: {stats['has_result_rate']*100:.1f}%")
        report_lines.append(f"平均检索评分: {stats['avg_score']:.4f}")
        report_lines.append(f"平均响应时间: {stats['avg_time']:.3f}秒")
        report_lines.append("")

        # 2. 问题类型分析
        report_lines.append("二、问题类型分析")
        report_lines.append("-"*80)

        for qtype, type_stats in sorted(stats['type_stats'].items(), key=lambda x: -x[1]['count']):
            count = type_stats['count']
            no_result = type_stats['no_result']
            scores = type_stats['scores']

            report_lines.append(f"\n{qtype}:")
            report_lines.append(f"  问题数量: {count}")
            report_lines.append(f"  无结果比例: {no_result/count*100:.1f}%")
            if scores:
                avg_score = sum(scores) / len(scores)
                report_lines.append(f"  平均评分: {avg_score:.4f}")

        # 3. 详细结果
        report_lines.append("\n" + "="*80)
        report_lines.append("三、详细测试结果")
        report_lines.append("="*80)
        report_lines.append("")

        for i, result in enumerate(self.results, 1):
            report_lines.append(f"\n[{i}] 问题: {result['question']}")
            report_lines.append(f"    类型: {result['type']}")
            report_lines.append(f"    响应时间: {result['elapsed_time']:.3f}秒")

            if result['results']:
                report_lines.append(f"    检索到 {len(result['results'])} 个结果")

                for j, item in enumerate(result['results'][:3], 1):
                    report_lines.append(f"\n    结果 #{j}:")
                    report_lines.append(f"      评分: {item['score']:.4f}")
                    report_lines.append(f"      来源: {item['metadata'].get('source', 'unknown')}")
                    report_lines.append(f"      页码: {item['metadata'].get('page_number', 'N/A')}")
                    report_lines.append(f"      内容: {item['text'][:150]}...")
            else:
                report_lines.append(f"    [FAIL] 未找到结果")

            report_lines.append("-"*80)

        # 保存报告
        report_text = "\n".join(report_lines)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n详细报告已保存到: {self.output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试RAG系统检索质量")
    parser.add_argument(
        "--questions",
        type=str,
        default="test_questions.json",
        help="测试问题JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_test_report.txt",
        help="测试报告输出路径"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索的Top-K数量"
    )

    args = parser.parse_args()

    # 初始化Pipeline
    print("初始化RAG Pipeline...")
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml"
    )

    # 创建测试器
    tester = RetrievalTester(
        pipeline=pipeline,
        questions_path=args.questions,
        output_path=args.output
    )

    # 运行测试
    tester.run_test(top_k=args.top_k)

    # 分析结果
    stats = tester.analyze_results()

    # 展示示例
    tester.show_examples(n=3)

    # 生成报告
    tester.generate_report(stats)

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
