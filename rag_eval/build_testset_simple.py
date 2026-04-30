"""
使用 Ragas 框架和 GLM 模型构建测试集（简化版）
专注于生成高质量的测试问题
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict

from langchain_openai import ChatOpenAI

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================
# API 配置（从 API_Setting.py）
# ================================
API_KEY = "c844d01915874aa38e5adfdcef05eee9.jZHyjygcan10qEqy"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"


class SimpleTestsetBuilder:
    """简化的测试集构建器 - 专注于生成问题"""

    def __init__(self, target_questions: int = 50):
        """
        初始化测试集构建器

        Args:
            target_questions: 目标问题数量
        """
        self.target_questions = target_questions

        print("创建 LangChain 客户端...")
        self.client = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=0.7
        )
        print("初始化完成\n")

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
                # 跳过合并文件和备份文件
                if "merged" in txt_file.name.lower() or "backup" in txt_file.name.lower():
                    continue

                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 只保留有实质内容的文档
                if len(content) > 500:
                    documents.append({
                        "source": str(txt_file),
                        "filename": txt_file.name,
                        "content": content
                    })

            except Exception as e:
                print(f"  读取文件失败 {txt_file}: {e}")

        print(f"成功加载 {len(documents)} 个文档")
        return documents

    def generate_questions_from_document(
        self,
        document: Dict,
        num_questions: int = 3,
        max_length: int = 8000
    ) -> List[Dict]:
        """
        从完整文档生成问题和标准答案

        Args:
            document: 文档对象（包含 source, filename, content）
            num_questions: 生成问题数量
            max_length: 最大文档长度（字符数）

        Returns:
            问题列表
        """
        content = document['content']
        filename = document['filename']

        # 如果文档太长，截取前面部分
        if len(content) > max_length:
            content = content[:max_length]
            print(f"  文档过长，截取前 {max_length} 字符")

        prompt = f"""你是一个专业的测试问题生成专家。请基于以下完整文档内容，生成 {num_questions} 个高质量的测试问题和标准答案。

文档名称：{filename}
文档内容：
{content}

要求：
1. 问题应该覆盖文档的主要内容，包括关键信息、核心概念、重要数据
2. 问题类型应多样化：
   - 事实性问题：询问具体的事实、数据、人名、地名等
   - 概念性问题：询问概念定义、含义等
   - 分析性问题：询问原因、影响、关系等
   - 综合性问题：需要综合文档多个部分信息才能回答
3. 标准答案应该：
   - 准确、完整，基于文档内容
   - 包含足够细节，能够验证检索质量
   - 避免过于简短或模糊
4. 问题应该能够有效测试 RAG 系统的检索能力
5. 避免生成过于简单的是非题

请以JSON格式返回：
[
    {{
        "question": "问题1",
        "ground_truth": "标准答案1，应该详细、准确，包含足够的信息点"
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

            # 添加来源信息
            for q in questions:
                q['source_document'] = filename
                q['source_file'] = document['source']

            return questions

        except Exception as e:
            print(f"  生成失败: {e}")
            return []

    def generate_testset_from_documents(
        self,
        docs_dir: str = None,
        questions_per_doc: int = 3,
        max_docs: int = None
    ) -> List[Dict]:
        """
        从完整文档生成测试集

        Args:
            docs_dir: 文档目录路径
            questions_per_doc: 每个文档生成的问题数
            max_docs: 最大处理文档数

        Returns:
            完整测试集
        """
        # 加载文档
        documents = self.load_documents(docs_dir)

        if not documents:
            print("\n错误: 未找到可用文档")
            return []

        # 计算需要处理的文档数
        if max_docs is None:
            max_docs = min(
                len(documents),
                (self.target_questions + questions_per_doc - 1) // questions_per_doc
            )

        print(f"\n计划处理 {max_docs} 个文档，生成约 {max_docs * questions_per_doc} 个问题")
        print()

        # 生成问题
        all_questions = []
        failed_docs = []

        for i in range(0, min(max_docs, len(documents))):
            doc = documents[i]

            print(f"[{i+1}/{max_docs}] 处理文档: {doc['filename']}")

            questions = self.generate_questions_from_document(
                document=doc,
                num_questions=questions_per_doc
            )

            if questions:
                all_questions.extend(questions)
                print(f"  [OK] 生成 {len(questions)} 个问题")
            else:
                failed_docs.append(doc['filename'])
                print(f"  [FAIL] 生成失败")

            # 避免请求过快
            import time
            time.sleep(0.5)

        print()
        print(f"成功生成 {len(all_questions)} 个问题")
        print(f"失败 {len(failed_docs)} 个文档")

        if failed_docs:
            print(f"\n失败的文档:")
            for filename in failed_docs:
                print(f"  - {filename}")

        return all_questions

    def save_testset(self, questions: List[Dict], output_path: str):
        """保存测试集"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "questions": questions,
                "num_questions": len(questions),
                "generation_method": "document_based",
                "model": MODEL_NAME
            }, f, ensure_ascii=False, indent=2)

        print(f"\n测试集已保存: {output_path}")

        # 保存为 CSV
        import pandas as pd
        csv_path = output_path.parent / output_path.stem.replace('.json', '.csv')
        df_questions = pd.DataFrame(questions)
        df_questions.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV 格式: {csv_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="使用 Ragas 和 GLM 从完整文档构建测试集"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="知识库/知识库",
        help="文档目录路径"
    )
    parser.add_argument(
        "--target-questions",
        type=int,
        default=50,
        help="目标问题数量"
    )
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=3,
        help="每个文档生成的问题数"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="最大处理文档数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rag_eval/evals/datasets/ragas_testset_doc_50q.json",
        help="测试集输出路径"
    )

    args = parser.parse_args()

    print("="*80)
    print("Ragas 测试集构建系统（基于完整文档）")
    print("="*80)
    print(f"目标问题数: {args.target_questions}")
    print(f"文档目录: {args.docs_dir}")
    print(f"每文档问题数: {args.questions_per_doc}")
    print()

    # 检查文档目录
    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        print(f"错误: 文档目录不存在: {docs_path}")
        return 1

    # 创建构建器
    builder = SimpleTestsetBuilder(target_questions=args.target_questions)

    # 生成测试集
    questions = builder.generate_testset_from_documents(
        docs_dir=args.docs_dir,
        questions_per_doc=args.questions_per_doc,
        max_docs=args.max_docs
    )

    if not questions:
        print("\n错误: 未能生成任何问题")
        return 1

    print(f"\n成功生成 {len(questions)} 个问题")

    # 显示示例
    print("\n示例问题:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\n{i}. {q['question']}")
        print(f"   来源: {q.get('source_document', 'unknown')}")
        print(f"   答案: {q['ground_truth'][:100]}...")

    # 保存测试集
    timestamp = len(questions)
    output_path = args.output.replace('50q', f'{timestamp}q')
    builder.save_testset(questions, output_path)

    # 统计来源文档
    source_docs = {}
    for q in questions:
        source = q.get('source_document', 'unknown')
        source_docs[source] = source_docs.get(source, 0) + 1

    print("\n来源文档统计:")
    for doc, count in sorted(source_docs.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {doc}: {count} 个问题")

    print("\n完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
