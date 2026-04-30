"""
PDF表格处理方案分析与设计
针对RAG系统的表格数据优化
"""

# ============================================================================
# 方案对比分析
# ============================================================================

PDF_TABLE_SOLUTIONS = {
    "方案1_表格检测保留": {
        "名称": "表格检测与格式保留",
        "优点": [
            "保留表格结构",
            "可读性强",
            "便于机器解析",
            "支持Markdown表格"
        ],
        "缺点": [
            "实现复杂度高",
            "需要专门的表格检测算法",
            "可能误判",
            "处理时间增加"
        ],
        "适合场景": [
            "表格密集的报告（如统计年报）",
            "需要保留数据结构",
            "后续需要机器读取"
        ]
    },

    "方案2_表格转结构化数据": {
        "名称": "提取为结构化数据（JSON/CSV）",
        "优点": [
            "数据结构化",
            "便于检索",
            "可以存储到数据库",
            "支持复杂查询"
        ],
        "缺点": [
            "需要定义schema",
            "表格解析困难",
            "多表格需要关联",
            "实现复杂度高"
        ],
        "适合场景": [
            "结构化数据密集的文档",
            "需要数据分析",
            "已有数据库schema"
        ]
    },

    "方案3_表格转自然语言": {
        "名称": "表格转描述性文本",
        "优点": [
            "实现简单",
            "适合RAG检索",
            "保持文本连贯性",
            "无需额外存储"
        ],
        "缺点": [
            "失去表格结构",
            "可能信息冗余",
            "不适合精确数据查询"
        ],
        "适合场景": [
            "一般性文档",
            "表格数量少",
            "主要做语义检索"
        ]
    },

    "方案4_表格作为独立Chunk": {
        "名称": "表格单独处理为特殊Chunk",
        "优点": [
            "保留原始格式",
            "可以针对性检索",
            "不影响正文chunking",
            "灵活"
        ],
        "缺点": [
            "需要表格检测",
            "处理流程复杂",
            "chunk数量增加"
        ],
        "适合场景": [
            "混合内容文档",
            "表格和文本都重要",
            "需要分别处理"
        ]
    },

    "方案5_OCR+表格识别": {
        "名称": "使用专业OCR工具（如PaddleOCR+表格识别）",
        "优点": [
            "识别准确",
            "保留表格结构",
            "支持复杂表格",
            "工业化成熟方案"
        ],
        "缺点": [
            "需要GPU/资源",
            "处理时间较长",
            "依赖外部工具",
            "安装配置复杂"
        ],
        "适合场景": [
            "扫描件PDF",
            "图像密集文档",
            "对准确性要求高"
        ]
    }
}

# ============================================================================
# 针对你的项目的推荐方案
# ============================================================================

def analyze_project_context():
    """
    分析项目上下文，给出针对性建议
    """
    return {
        "项目特点": {
            "文档类型": [
                "中国通用航空2021 - 研究报告（含统计表格）",
                "交通运输统计公报 - 数据密集（大量表格）",
                "学术论文（可能含实验数据表）",
                "政策文件（可能含条例表）"
            ],
            "使用场景": "RAG问答系统",
            "检索需求": "语义检索为主，数据查询为辅"
        },

        "推荐策略": {
            "阶段1_快速实现": {
                "方案": "方案3_表格转自然语言 + 方案4_表格独立Chunk",
                "原理": "将表格转换为描述性文本，表格单独chunking",
                "优点": "实现简单，快速上线",
                "实现难度": "⭐⭐"
            },

            "阶段2_优化提升": {
                "方案": "方案1_表格检测保留 + Markdown格式",
                "原理": "检测表格区域，保留为Markdown表格",
                "优点": "兼顾可读性和结构",
                "实现难度": "⭐⭐⭐"
            },

            "阶段3_高级功能": {
                "方案": "方案5_OCR + 表格识别引擎",
                "原理": "使用PaddleOCR的表格识别功能",
                "优点": "专业级准确度",
                "实现难度": "⭐⭐⭐⭐"
            }
        }
    }


# ============================================================================
# 具体实现代码示例
# ============================================================================

def table_implementation_examples():
    """
    提供各种方案的具体实现示例
    """
    examples = {
        "示例1_表格转描述性文本": {
            "代码": '''
# 转换前：
| 2021年 | 2022年 | 2023年 |
| 100万 | 150万 | 200万 |

# 转换后：
根据统计数据显示，2021年飞行量为100万人次，
2022年增长至150万人次，2023年进一步增长至200万人次。

# 保持表格信息的可检索性
'''
        },

        "示例2_表格转Markdown": {
            "代码": '''
# 转换前（文本流）：
年份    产值（亿元）    增长率
2021    1000          5%
2022    1100          10%

# 转换后（Markdown表格）：
| 年份 | 产值（亿元） | 增长率 |
|------|-------------|--------|
| 2021 | 1000        | 5% |
| 2022 | 1100        | 10% |

# 优点：保留表格结构，支持Markdown渲染
'''
        },

        "示例3_表格独立Chunk": {
            "代码": '''
# 为表格添加特殊metadata
chunk = Document(
    page_content="| 年份 | 产值 |\\n| 2021 | 1000 |",
    metadata={
        "is_table": True,
        "table_type": "data",
        "table_columns": ["年份", "产值"],
        "source": "表1：统计数据"
    }
)
'''
        }
    }

    return examples


# ============================================================================
# 表格检测算法设计
# ============================================================================

class TableDetector:
    """
    PDF表格检测器

    检测规则：
    1. 多个连续行有相似的分隔符（|、空格、制表符）
    2. 数字对齐的行
    3. 重复的列标题模式
    4. 表格标题行（如"年份"、"产值"等）
    """

    @staticmethod
    def detect_table_region(text: str) -> list:
        """
        检测文本中的表格区域

        Returns:
            [(start_pos, end_pos, confidence), ...]
        """
        import re

        regions = []
        lines = text.split('\n')

        # 规则1: 检测包含管道符的行
        pipe_lines = []
        for i, line in enumerate(lines):
            if '|' in line and line.count('|') >= 2:
                pipe_lines.append(i)

        # 将连续的管道行分组
        if pipe_lines:
            start = pipe_lines[0]
            for j in range(1, len(pipe_lines)):
                if pipe_lines[j] != pipe_lines[j-1] + 1:
                    # 不连续
                    if pipe_lines[j-1] >= start:
                        regions.append((start, pipe_lines[j-1] + 1, "pipe_table"))
                    start = pipe_lines[j]

            # 添加最后一组
            if start < len(lines):
                regions.append((start, pipe_lines[-1] + 1, "pipe_table"))

        # 规则2: 检测数字对齐的行
        # ... (省略具体实现)

        return regions


# ============================================================================
# 表格转换器
# ============================================================================

class TableConverter:
    """
    表格转换器 - 将检测到的表格转换为不同格式
    """

    @staticmethod
    def convert_to_description(table_text: str, context: str = "") -> str:
        """
        表格转描述性文本

        Args:
            table_text: 表格文本
            context: 上下文信息

        Returns:
            描述性文本
        """
        import re

        lines = table_text.strip().split('\n')
        if len(lines) < 2:
            return table_text

        # 解析表格
        # 简化实现：假设第一行是表头
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        data_rows = []

        for line in lines[1:]:
            if not line.strip():
                continue
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) == len(headers):
                data_rows.append(cells)

        # 转换为描述
        descriptions = []
        for row in data_rows:
            row_desc = "，".join([f"{h}为{v}" for h, v in zip(headers, row) if v])
            descriptions.append(row_desc + "；")

        if descriptions:
            table_desc = "。".join(descriptions)
            return f"根据{context}中的表格数据显示，{table_desc}"

        return table_text

    @staticmethod
    def convert_to_markdown(table_text: str) -> str:
        """
        表格转Markdown格式

        Args:
            table_text: 表格文本

        Returns:
            Markdown表格
        """
        return table_text  # 已经是Markdown格式或需要转换

    @staticmethod
    def convert_to_structured(table_text: str) -> dict:
        """
        表格转结构化数据（JSON）

        Args:
            table_text: 表格文本

        Returns:
            结构化数据字典
        """
        import json

        # 解析表格
        lines = table_text.strip().split('\n')
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        data = []

        for line in lines[1:]:
            if not line.strip():
                continue
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) == len(headers):
                row_dict = {h: v for h, v in zip(headers, cells)}
                data.append(row_dict)

        return {
            "headers": headers,
            "data": data
        }


# ============================================================================
# 推荐的完整处理流程
# ============================================================================

RECOMMENDED_WORKFLOW = """
# 阶段1：快速实现（当前推荐）

1. PDF转TXT（PyPDF/PyMuPDF）
   ↓
2. 表格检测（TableDetector）
   - 识别表格区域
   - 标记表格类型（数据表/统计表/列表）
   ↓
3. 表格转换策略选择：

   A. 简单表格（< 5列）→ 转为描述性文本
      例："年份2021年产值1000亿元，2022年增长至1100亿元。"

   B. 复杂表格（≥ 5列）→ 保留为Markdown格式
      保留原格式，在chunking时特殊处理

   C. 关键数据表 → 提取为结构化数据
      存储为JSON，便于精确查询

   ↓
4. 文本重组
   - 表格转描述
   - 添加表格摘要
   - 保留原表格（作为参考）
   ↓
5. Chunking
   - 表格chunk添加特殊metadata
   - 表格数据单独chunking（如果重要）
   ↓
6. 存储到RAG系统

# 阶段2：优化方案（可选）

使用专业OCR + 表格识别（PaddleOCR/Tabula）
"""

# ============================================================================
# 针对用户项目的具体建议
# ============================================================================

def get_project_specific_recommendations():
    """
    根据用户项目特点给出具体建议
    """
    return {
        "文档类型分析": {
            "中国通用航空2021": {
                "表格类型": "统计数据表格",
                "表格数量": "中等（10-20个）",
                "复杂度": "中等",
                "推荐处理": "方案A+B混合"
            },
            "交通运输统计公报": {
                "表格类型": "大量数据表",
                "表格数量": "很多（50+）",
                "复杂度": "高",
                "推荐处理": "方案B为主，方案C为辅"
            },
            "政策文件": {
                "表格类型": "条例/配置表",
                "表格数量": "少（5-10个）",
                "复杂度": "低",
                "推荐处理": "方案A"
            }
        },

        "实施优先级": {
            "P0_立即实施": {
                "方案": "方案3_表格转描述性文本",
                "原因": "实现简单，快速见效",
                "工作量": "1-2天"
            },
            "P1_短期优化": {
                "方案": "方案4_表格独立Chunk + Metadata标记",
                "原因": "提升检索精度",
                "工作量": "3-5天"
            },
            "P2_长期完善": {
                "方案": "方案1_表格检测 + Markdown保留",
                "原因": "最佳用户体验",
                "工作量": "5-10天"
            }
        },

        "不推荐": {
            "方案2_全部转结构化数据": {
                "原因": [
                    "schema定义复杂",
                    "表格格式不统一",
                    "维护成本高",
                    "过度工程"
                ]
            },
            "方案5_全套OCR": {
                "原因": [
                    "已有文本版PDF",
                    "GPU资源有限",
                    "处理时间过长",
                    "性价比不高"
                ]
            }
        }
    }


# ============================================================================
# 快速实现代码示例
# ============================================================================

QUICK_START_GUIDE = """
# 快速实现指南：表格转描述性文本

from rag_project.data_loader.table_processor import TableProcessor

# 1. 检测表格
processor = TableProcessor()
tables = processor.detect_tables("文档.txt")

# 2. 转换表格
for table in tables:
    if table['column_count'] <= 5:
        # 简单表格：转为描述
        description = processor.to_description(table)
    else:
        # 复杂表格：保留为Markdown
        markdown = processor.to_markdown(table)

# 3. 添加到文档
processor.replace_table_in_document(
    "文档.txt",
    table,
    description  # 或 markdown
)
"""
