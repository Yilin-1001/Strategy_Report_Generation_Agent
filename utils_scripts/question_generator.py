"""
问题生成器 - 基于文档内容生成测试问题
使用规则生成策略，从知识库chunks中提取信息并生成问题
"""
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_core.documents import Document


class QuestionGenerator:
    """基于规则的问题生成器"""

    # 问题生成规则
    RULES = {
        "投资数据": [
            r"(\d{4})年.*?投资.*?(\d+\.?\d*[万亿千百]?)元",
            r"(.*?)投资.*?(\d+\.?\d*[万亿千百]?)元"
        ],
        "统计数据": [
            r"(\d{4})年.*?(\d+\.?\d*[万千百]?)",
            r"(.*?)达到.*?(\d+\.?\d*[万千百]?)"
        ],
        "政策措施": [
            r"(加强|推进|实施|完善)(.*?)工作",
            r"(采取|制定)(.*?)(措施|方案|政策)"
        ],
        "机构职责": [
            r"(.*?)负责(.*?)",
            r"(.*?)主要(.*?)(业务|工作|职责)"
        ]
    }

    QUESTION_TEMPLATES = {
        "投资数据": [
            "{year}年{entity}投资金额是多少？",
            "{entity}在{year}年的投资情况如何？",
            "{year}年{entity}完成了多少投资？"
        ],
        "统计数据": [
            "{year}年{indicator}达到多少？",
            "{indicator}的数��是多少？",
            "{entity}{indicator}的具体数值？"
        ],
        "政策措施": [
            "如何{action}？",
            "{target}有哪些措施？",
            "{action}的具体内容是什么？"
        ],
        "机构职责": [
            "{org}的主要职责是什么？",
            "{org}负责哪些工作？",
            "{org}的业务范围包括什么？"
        ],
        "综合概述": [
            "{topic}的发展情况如何？",
            "{topic}有哪些主要特点？",
            "关于{topic}的介绍"
        ]
    }

    def __init__(self, chunks_json_path: str):
        """
        初始化问题生成器

        Args:
            chunks_json_path: chunks的JSON文件路径
        """
        self.chunks_json_path = chunks_json_path
        self.chunks = self._load_chunks()

    def _load_chunks(self) -> List[Document]:
        """加载chunks数据"""
        with open(self.chunks_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        for item in data:
            # 兼容两种格式
            text = item.get('text') or item.get('page_content')
            metadata = item.get('metadata', {})
            doc = Document(page_content=text, metadata=metadata)
            chunks.append(doc)

        return chunks

    def _extract_numbers(self, text: str) -> List[str]:
        """提取文本中的数字"""
        # 匹配年份（1900-2099）
        years = re.findall(r'20[12][0-9]', text)
        # 匹配金额数字
        amounts = re.findall(r'\d+\.?\d*[万亿千百]?', text)
        # 匹配纯数字
        numbers = re.findall(r'\d+\.?\d*', text)

        return list(set(years + amounts + numbers))

    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的实体（简单版）"""
        entities = []

        # 提取机构名称（2-10个中文字符）
        orgs = re.findall(r'([一-龥]{2,10})(集团|公司|局|委|部|厅)', text)
        entities.extend([org[0] + org[1] for org in orgs])

        # 提取地名（2-6个中文字符+省/市/区）
        places = re.findall(r'([一-龥]{2,6})(省|市|区|县)', text)
        entities.extend([place[0] + place[1] for place in places])

        # 提取领域关键词
        keywords = re.findall(r'(交通运输|高速公路|通用航空|安全生产|绿色发展|投资建设)', text)
        entities.extend(keywords)

        return list(set(entities))

    def _sample_chunks(self, n: int = 50) -> List[Document]:
        """随机采样chunks"""
        return random.sample(self.chunks, min(n, len(self.chunks)))

    def generate_questions(
        self,
        num_questions: int = 30,
        min_chunks: int = 100
    ) -> List[Dict]:
        """
        生成测试问题

        Args:
            num_questions: 要生成的问题数量
            min_chunks: 最少采样chunk数量

        Returns:
            问题列表，每个问题包含：
            - question: 问题文本
            - type: 问题类型
            - source_chunk: 来源chunk内容
            - source_metadata: 来源chunk的metadata
        """
        print(f"从 {len(self.chunks)} 个chunks中采样...")
        sampled_chunks = self._sample_chunks(min_chunks)

        questions = []

        for chunk in sampled_chunks:
            if len(questions) >= num_questions:
                break

            text = chunk.page_content
            metadata = chunk.metadata

            # 跳过太短的chunk
            if len(text) < 50:
                continue

            # 尝试多种方式生成问题
            generated = self._generate_from_chunk(chunk)
            if generated:
                questions.extend(generated)

        # 随机打乱并限制数量
        random.shuffle(questions)
        return questions[:num_questions]

    def _generate_from_chunk(self, chunk: Document) -> List[Dict]:
        """从单个chunk生成问题"""
        text = chunk.page_content
        metadata = chunk.metadata
        questions = []

        # 提取信息
        numbers = self._extract_numbers(text)
        entities = self._extract_entities(text)

        # 策略1: 如果有年份和数字，生成数据问题
        years = [n for n in numbers if re.match(r'20[12][0-9]', n)]
        if years and len(entities) > 0:
            question = {
                "question": f"{years[0]}年{entities[0]}的数据情况如何？",
                "type": "统计数据",
                "source_chunk": text[:200] + "...",
                "source_metadata": {
                    "source": metadata.get("source", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "page_number": metadata.get("page_number", 0)
                }
            }
            questions.append(question)

        # 策略2: 如果有机构名，生成职责问题
        org_keywords = ["集团", "公司", "局", "委", "部"]
        if any(kw in text for kw in org_keywords):
            # 找第一个匹配的实体
            orgs = [e for e in entities if any(kw in e for kw in org_keywords)]
            if orgs:
                question = {
                    "question": f"{orgs[0]}的主要工作是什么？",
                    "type": "机构职责",
                    "source_chunk": text[:200] + "...",
                    "source_metadata": {
                        "source": metadata.get("source", ""),
                        "doc_type": metadata.get("doc_type", ""),
                        "page_number": metadata.get("page_number", 0)
                    }
                }
                questions.append(question)

        # 策略3: 如果有政策/措施关键词，生成措施问题
        policy_keywords = ["加强", "推进", "实施", "完善", "措施", "政策", "方案"]
        if any(kw in text for kw in policy_keywords):
            # 找领域关键词
            domain_keywords = ["安全生产", "交通建设", "投资", "管理", "发展"]
            domains = [kw for kw in domain_keywords if kw in text]

            if domains:
                question = {
                    "question": f"如何加强{domains[0]}？",
                    "type": "政策措施",
                    "source_chunk": text[:200] + "...",
                    "source_metadata": {
                        "source": metadata.get("source", ""),
                        "doc_type": metadata.get("doc_type", ""),
                        "page_number": metadata.get("page_number", 0)
                    }
                }
                questions.append(question)

        # 策略4: 综合问题
        if len(entities) >= 2:
            question = {
                "question": f"关于{entities[0]}的概况介绍",
                "type": "综合概述",
                "source_chunk": text[:200] + "...",
                "source_metadata": {
                    "source": metadata.get("source", ""),
                    "doc_type": metadata.get("doc_type", ""),
                    "page_number": metadata.get("page_number", 0)
                }
            }
            questions.append(question)

        # 限制每个chunk最多生成3个问题
        return questions[:3]

    def save_questions(self, questions: List[Dict], output_path: str):
        """保存问题到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        print(f"已保存 {len(questions)} 个问题到: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="生成RAG测试问题集")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="chunks的JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_questions.json",
        help="输出问题文件路径"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=30,
        help="生成问题数量"
    )

    args = parser.parse_args()

    # 创建生成器
    generator = QuestionGenerator(args.chunks)

    # 生成问题
    print(f"正在生成 {args.num} 个测试问题...")
    questions = generator.generate_questions(num_questions=args.num)

    # 保存问题
    generator.save_questions(questions, args.output)

    # 显示问题统计
    print("\n" + "="*80)
    print("问题生成完成！")
    print("="*80)

    type_count = {}
    for q in questions:
        qtype = q["type"]
        type_count[qtype] = type_count.get(qtype, 0) + 1

    print("\n问题类型分布:")
    for qtype, count in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"  {qtype}: {count}个")

    print("\n前5个问题示例:")
    for i, q in enumerate(questions[:5], 1):
        print(f"\n{i}. {q['question']}")
        print(f"   类型: {q['type']}")
        print(f"   来源: {q['source_metadata']['source']}")


if __name__ == "__main__":
    main()
