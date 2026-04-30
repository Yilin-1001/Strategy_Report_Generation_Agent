"""
使用 Ragas 框架和 GLM 模型构建测试集（智能分段版）
实施方案C：智能分段策略，实现95%覆盖度
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================
# API 配置（从 API_Setting.py）
# ================================
API_KEY = "c844d01915874aa38e5adfdcef05eee9.jZHyjygcan10qEqy"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"


class SmartTestsetBuilder:
    """智能测试集构建器 - 方案C：智能分段策略"""

    def __init__(
        self,
        checkpoint_dir: str = "rag_eval/evals/checkpoints",
        batch_size: int = 30
    ):
        """
        初始化测试集构建器

        Args:
            checkpoint_dir: 检查点保存目录
            batch_size: 每批处理的文档数
        """
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print("创建 LangChain 客户端...")
        self.client = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=0.7
        )
        print("初始化完成\n")

        # 统计信息
        self.stats = {
            'total_docs': 0,
            'processed_docs': 0,
            'total_segments': 0,
            'total_questions': 0,
            'failed_docs': 0,
            'skipped_docs': 0,
            'start_time': datetime.now().isoformat()
        }

    def classify_document(self, content: str) -> Tuple[str, int, int]:
        """
        分类文档并确定分段策略和问题数量

        Args:
            content: 文档内容

        Returns:
            (文档类型, 分段数, 每段问题数)
        """
        length = len(content)

        if length > 100000:
            # 超长文档 (>100K字): 按章节切分，生成10-15问
            segments = max(10, length // 10000)  # 每10K字一段
            questions_per_segment = 1
            return ('very_long', segments, questions_per_segment)

        elif length > 50000:
            # 长文档 (50K-100K): 分5-7段，生成7-10问
            segments = max(5, length // 15000)
            questions_per_segment = 2
            return ('long', segments, questions_per_segment)

        elif length > 20000:
            # 中长文档 (20K-50K): 分3-5段，生成5-7问
            segments = max(3, length // 15000)
            questions_per_segment = 2
            return ('medium_long', segments, questions_per_segment)

        elif length > 10000:
            # 中等文档 (10K-20K): 不分段，生成3-5问
            return ('medium', 1, 4)

        elif length > 1000:
            # 短文档 (1K-10K): 不分段，生成2-3问
            return ('short', 1, 2)

        else:
            # 过短文档: 跳过
            return ('too_short', 0, 0)

    def split_document_smart(
        self,
        content: str,
        num_segments: int
    ) -> List[str]:
        """
        智能分割文档

        Args:
            content: 文档内容
            num_segments: 分段数

        Returns:
            分段后的内容列表
        """
        if num_segments == 1:
            return [content]

        # 尝试按章节分割
        import re

        # 查找章节标题模式
        patterns = [
            r'\n第[一二三四五六七八九十百千零\d]+章[^\n]{0,50}\n',
            r'\n[一二三四五六七八九十]+、[^\n]{0,50}\n',
            r'\n\d+\.\s+[^\n]{0,50}\n',
            r'\n{2,}',  # 双换行作为备选
        ]

        # 尝试匹配章节
        split_points = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, content))
            if len(matches) >= num_segments - 1:
                # 找到足够的分割点
                split_points = [m.start() for m in matches[:num_segments-1]]
                break

        if not split_points:
            # 如果没有找到章节，按长度均分
            segment_length = len(content) // num_segments
            split_points = [
                i * segment_length
                for i in range(1, num_segments)
            ]

        # 分割内容
        segments = []
        prev = 0
        for point in split_points:
            segments.append(content[prev:point].strip())
            prev = point
        segments.append(content[prev:].strip())

        # 过滤空段
        segments = [s for s in segments if len(s) > 100]

        return segments

    def load_documents(self, docs_dir: str = None) -> List[Dict]:
        """
        加载原始文档（TXT 格式）

        Args:
            docs_dir: 文档目录路径

        Returns:
            文档列表
        """
        if docs_dir is None:
            docs_dir = "知识库/知识库"

        docs_path = Path(docs_dir)
        documents = []

        print(f"扫描文档目录: {docs_path}")

        # 递归查找所有 TXT 文件
        txt_files = list(docs_path.rglob("*.txt"))

        print(f"找到 {len(txt_files)} 个 TXT 文件")

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 只保留有实质内容的文档
                if len(content) > 1000:
                    # 跳过备份和临时文件
                    if "backup" in txt_file.name.lower():
                        continue

                    documents.append({
                        "source": str(txt_file),
                        "filename": txt_file.name,
                        "content": content,
                        "category": txt_file.parent.name
                    })

            except Exception as e:
                print(f"  读取文件失败 {txt_file}: {e}")

        print(f"成功加载 {len(documents)} 个有效文档")
        return documents

    def generate_questions_from_segment(
        self,
        segment: str,
        filename: str,
        segment_index: int,
        num_questions: int
    ) -> List[Dict]:
        """
        从文档片段生成问题

        Args:
            segment: 文档片段内容
            filename: 文件名
            segment_index: 片段索引
            num_questions: 生成问题数量

        Returns:
            问题列表
        """
        # 限制最大长度
        max_length = 8000
        if len(segment) > max_length:
            segment = segment[:max_length]

        prompt = f"""你是一个专业的测试问题生成专家。请基于以下文档片段，生成 {num_questions} 个高质量的测试问题和标准答案。

文档名称：{filename}
片段索引：第 {segment_index} 段
文档内容：
{segment}

要求：
1. 问题应该覆盖该片段的主要内容，包括关键信息、核心概念、重要数据
2. 问题类型应多样化：
   - 事实性问题：询问具体的事实、数据、人名、地名等
   - 概念性问题：询问概念定义、含义等
   - 分析性问题：询问原因、影响、关系等
3. 标准答案应该：
   - 准确、完整，基于文档内容
   - 包含足够细节，能够验证检索质量
   - 避免过于简短或模糊
4. 避免生成过于简单的是非题

请以JSON格式返回：
[
    {{
        "question": "问题1",
        "ground_truth": "标准答案1"
    }},
    {{
        "question": "问题2",
        "ground_truth": "标准答案2"
    }}
]

只返回JSON，不要其他内容。"""

        try:
            response = self.client.invoke(prompt)
            result_text = response.content.strip()

            # 清理 markdown 标记
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            questions = json.loads(result_text.strip())

            return questions

        except Exception as e:
            print(f"    生成失败: {e}")
            return []

    def process_document(self, doc: Dict) -> List[Dict]:
        """
        处理单个文档（支持分段）

        Args:
            doc: 文档对象

        Returns:
            生成的问题列表
        """
        content = doc['content']
        filename = doc['filename']

        # 分类文档
        doc_type, num_segments, questions_per_segment = self.classify_document(content)

        if doc_type == 'too_short':
            print(f"  [SKIP] 文档过短 (<1K字)")
            self.stats['skipped_docs'] += 1
            return []

        print(f"  [分类] {doc_type} -> {num_segments}段 × {questions_per_segment}问")

        # 分段处理
        segments = self.split_document_smart(content, num_segments)
        print(f"  [分段] 实际分为 {len(segments)} 段")

        all_questions = []
        for i, segment in enumerate(segments, 1):
            print(f"    处理第 {i}/{len(segments)} 段 ({len(segment):,}字)...", end='')

            questions = self.generate_questions_from_segment(
                segment=segment,
                filename=filename,
                segment_index=i,
                num_questions=questions_per_segment
            )

            if questions:
                # 添加来源信息
                for q in questions:
                    q['source_document'] = filename
                    q['source_file'] = doc['source']
                    q['segment_index'] = i
                    q['doc_type'] = doc_type

                all_questions.extend(questions)
                print(f" OK ({len(questions)}问)")
                self.stats['total_questions'] += len(questions)
            else:
                print(f" FAIL")

            # 避免请求过快
            time.sleep(0.5)

        self.stats['processed_docs'] += 1
        self.stats['total_segments'] += len(segments)

        return all_questions

    def save_checkpoint(
        self,
        questions: List[Dict],
        batch_index: int
    ):
        """保存检查点"""
        checkpoint_file = self.checkpoint_dir / f"batch_{batch_index}.json"

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "batch_index": batch_index,
                "timestamp": datetime.now().isoformat(),
                "questions": questions,
                "stats": self.stats
            }, f, ensure_ascii=False, indent=2)

        print(f"  [保存] 检查点: {checkpoint_file}")

    def generate_testset(
        self,
        docs_dir: str = None,
        max_docs: int = None
    ) -> List[Dict]:
        """
        生成完整测试集

        Args:
            docs_dir: 文档目录路径
            max_docs: 最大处理文档数

        Returns:
            完整测试集
        """
        # 加载文档
        documents = self.load_documents(docs_dir)

        if not documents:
            print("\n错误: 未找到可用文档")
            return []

        self.stats['total_docs'] = len(documents)

        if max_docs:
            documents = documents[:max_docs]
            print(f"\n限制处理文档数: {max_docs}")

        print(f"\n计划处理 {len(documents)} 个文档\n")

        # 分批处理
        all_questions = []
        failed_docs = []

        for batch_start in range(0, len(documents), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]
            batch_index = batch_start // self.batch_size + 1

            print(f"\n{'='*60}")
            print(f"批次 {batch_index}: 文档 {batch_start+1}-{batch_end}")
            print(f"{'='*60}")

            batch_questions = []

            for i, doc in enumerate(batch_docs, 1):
                doc_num = batch_start + i
                print(f"\n[{doc_num}/{len(documents)}] {doc['filename'][:50]}")
                print(f"  长度: {len(doc['content']):,} 字符")

                try:
                    questions = self.process_document(doc)

                    if questions:
                        batch_questions.extend(questions)
                        print(f"  [OK] 共生成 {len(questions)} 个问题")
                    else:
                        failed_docs.append(doc['filename'])
                        print(f"  [FAIL] 生成失败")
                        self.stats['failed_docs'] += 1

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    failed_docs.append(doc['filename'])
                    self.stats['failed_docs'] += 1

            # 保存批次检查点
            all_questions.extend(batch_questions)
            self.save_checkpoint(batch_questions, batch_index)

        return all_questions

    def save_testset(
        self,
        questions: List[Dict],
        output_path: str
    ):
        """保存测试集"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 计算最终统计
        start_time = datetime.fromisoformat(self.stats['start_time'])
        elapsed = (datetime.now() - start_time).total_seconds()

        # 保存为 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "questions": questions,
                "num_questions": len(questions),
                "generation_method": "smart_segmentation",
                "model": MODEL_NAME,
                "stats": {
                    **self.stats,
                    'elapsed_seconds': elapsed,
                    'elapsed_minutes': elapsed / 60
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"\n测试集已保存: {output_path}")

        # 保存为 CSV
        import pandas as pd
        csv_path = output_path.parent / output_path.stem.replace('.json', '.csv')

        # 准备CSV数据
        csv_data = []
        for q in questions:
            csv_data.append({
                "question": q['question'],
                "ground_truth": q['ground_truth'],
                "source_document": q.get('source_document', ''),
                "segment_index": q.get('segment_index', 1),
                "doc_type": q.get('doc_type', '')
            })

        df_questions = pd.DataFrame(csv_data)
        df_questions.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV 格式: {csv_path}")

    def print_summary(self, questions: List[Dict]):
        """打印总结信息"""
        print("\n" + "="*80)
        print("生成总结")
        print("="*80)

        # 解析start_time
        start_time = datetime.fromisoformat(self.stats['start_time'])
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n处理文档:")
        print(f"  总文档数: {self.stats['total_docs']}")
        print(f"  成功处理: {self.stats['processed_docs']}")
        print(f"  跳过文档: {self.stats['skipped_docs']}")
        print(f"  失败文档: {self.stats['failed_docs']}")

        print(f"\n问题统计:")
        print(f"  总问题数: {len(questions)}")
        print(f"  总分段数: {self.stats['total_segments']}")
        print(f"  平均每文档: {len(questions)/max(self.stats['processed_docs'],1):.1f} 问")

        print(f"\n时间消耗:")
        print(f"  总耗时: {elapsed/60:.1f} 分钟")
        print(f"  平均每文档: {elapsed/max(self.stats['processed_docs'],1):.1f} 秒")

        # 按文档类型统计
        doc_type_stats = {}
        for q in questions:
            doc_type = q.get('doc_type', 'unknown')
            doc_type_stats[doc_type] = doc_type_stats.get(doc_type, 0) + 1

        print(f"\n问题分布（按文档类型）:")
        for doc_type, count in sorted(doc_type_stats.items(), key=lambda x: -x[1]):
            pct = count / len(questions) * 100
            print(f"  {doc_type}: {count} ({pct:.1f}%)")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="智能分段测试集构建系统（方案C）"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="知识库/知识库",
        help="文档目录路径"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="最大处理文档数（用于测试）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="每批处理的文档数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rag_eval/evals/datasets/smart_testset_full.json",
        help="测试集输出路径"
    )

    args = parser.parse_args()

    print("="*80)
    print("智能分段测试集构建系统（方案C）")
    print("="*80)
    print(f"文档目录: {args.docs_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"策略: 智能分段 + 动态问题数")
    print()

    # 检查文档目录
    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        print(f"错误: 文档目录不存在: {docs_path}")
        return 1

    # 创建构建器
    builder = SmartTestsetBuilder(
        batch_size=args.batch_size
    )

    # 生成测试集
    questions = builder.generate_testset(
        docs_dir=args.docs_dir,
        max_docs=args.max_docs
    )

    if not questions:
        print("\n错误: 未能生成任何问题")
        return 1

    # 打印总结
    builder.print_summary(questions)

    # 显示示例问题
    print(f"\n示例问题（前3个）:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\n{i}. {q['question']}")
        print(f"   来源: {q.get('source_document', 'unknown')}")
        print(f"   类型: {q.get('doc_type', 'unknown')}")
        print(f"   答案: {q['ground_truth'][:100]}...")

    # 保存测试集
    output_path = args.output
    builder.save_testset(questions, output_path)

    print("\n" + "="*80)
    print("完成!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
