"""
PDF表格处理器 - 为RAG系统优化表格数据
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from rag_project.utils.logger import logger


@dataclass
class TableRegion:
    """表格区域"""
    start_line: int
    end_line: int
    table_type: str  # 'data_table', 'list', 'simple_table'
    confidence: float
    headers: List[str]
    rows: List[List[str]]
    raw_text: str


class TableDetector:
    """
    PDF表格检测器

    检测PDF文本中的表格区域
    """

    # 表格标题关键词
    TABLE_KEYWORDS = [
        '年份', '指标', '数据', '统计', '总计', '合计',
        '单位', '数量', '金额', '比例', '增长率',
        'Year', 'Data', 'Total', 'Amount', 'Rate'
    ]

    def __init__(self):
        self.table_patterns = [
            # 管道符表格
            re.compile(r'^\|.*\|.*\|'),  # 至少3个|
            # 制表符分隔
            re.compile(r'.*\t.*\t.*'),  # 至少2个tab
            # 多个空格对齐（数字）
            re.compile(r'^\s*\d+.*\d+\s*$'),  # 开头和结尾都是数字
        ]

    def detect_tables(self, text: str) -> List[TableRegion]:
        """
        检测文本中的表格区域

        Args:
            text: 文本内容

        Returns:
            检测到的表格区域列表
        """
        lines = text.split('\n')
        tables = []

        # 策略1: 检测管道符表格
        pipe_table_lines = []
        in_table = False
        table_start = 0

        for i, line in enumerate(lines):
            # 检查是否是表格行
            is_table_line = False

            # 规则1: 管道符表格
            if '|' in line and line.count('|') >= 2:
                is_table_line = True
                table_type = 'pipe_table'

            # 规则2: 制表符分隔
            elif '\t' in line and line.count('\t') >= 1:
                is_table_line = True
                table_type = 'tab_table'

            # 规则3: 多个数字对齐
            elif self._is_aligned_number_row(line):
                is_table_line = True
                table_type = 'aligned_table'

            # 状态转换
            if is_table_line and not in_table:
                # 进入表格
                table_start = i
                in_table = True
                pipe_table_lines.append(i)

            elif not is_table_line and in_table:
                # 退出表格
                # 检查是否至少有2行（表头+1行数据）
                if len(pipe_table_lines) >= 2 and pipe_table_lines[-1] - table_start >= 1:
                    # 提取表格
                    table_lines = lines[table_start:pipe_table_lines[-1]+1]
                    table_text = '\n'.join(table_lines)

                    # 解析表格
                    headers, rows = self._parse_table(table_text, table_type)

                    if headers and rows:
                        tables.append(TableRegion(
                            start_line=table_start,
                            end_line=pipe_table_lines[-1],
                            table_type=table_type,
                            confidence=0.8,
                            headers=headers,
                            rows=rows,
                            raw_text=table_text
                        ))

                # 重置
                in_table = False
                pipe_table_lines = []

        logger.info(f"检测到 {len(tables)} 个表格区域")
        return tables

    def _is_aligned_number_row(self, line: str) -> bool:
        """
        检查是否是数字对齐的行

        Args:
            line: 文本行

        Returns:
            是否是数字对齐行
        """
        # 检查是否有3个以上数字
        numbers = re.findall(r'\d+', line)
        if len(numbers) < 3:
            return False

        # 检查数字是否用空格分隔
        parts = line.strip().split()
        if len(parts) < 3:
            return False

        # 检查大部分parts是否是数字
        digit_parts = sum(1 for p in parts if re.match(r'\d+', p))
        return digit_parts / len(parts) >= 0.7

    def _parse_table(
        self,
        table_text: str,
        table_type: str
    ) -> Tuple[List[str], List[List[str]]]:
        """
        解析表格文本

        Args:
            table_text: 表格文本
            table_type: 表格类型

        Returns:
            (表头列表, 数据行列表)
        """
        lines = table_text.strip().split('\n')
        if not lines:
            return [], []

        # 确定分隔符
        if table_type == 'pipe_table':
            separator = '|'
        elif table_type == 'tab_table':
            separator = '\t'
        else:  # aligned_table
            separator = None  # 需要智能检测

        # 解析第一行作为表头
        headers = self._parse_line(lines[0], separator)

        # 解析数据行
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue

            row = self._parse_line(line, separator)
            if row and len(row) == len(headers):
                rows.append(row)

        return headers, rows

    def _parse_line(self, line: str, separator: Optional[str] = None) -> List[str]:
        """
        解析单行表格数据

        Args:
            line: 表格行
            separator: 分隔符

        Returns:
            解析后的单元格列表
        """
        if separator:
            parts = line.split(separator)
        else:
            # 智能检测：尝试多个分隔符
            # 按空格分割，然后合并连续的空格
            parts = line.strip().split()
            merged = []
            current = []

            for part in parts:
                if not part:
                    if current:
                        merged.append(' '.join(current))
                        current = []
                else:
                    current.append(part)

            if current:
                merged.append(' '.join(current))

            parts = merged

        # 去除首尾空白
        return [p.strip() for p in parts if p.strip()]


class TableConverter:
    """
    表格转换器 - 将表格转换为不同格式
    """

    @staticmethod
    def to_description(
        table: TableRegion,
        context: str = ""
    ) -> str:
        """
        表格转描述性文本

        Args:
            table: 表格区域
            context: 上下文信息

        Returns:
            描述性文本
        """
        if not table.headers or not table.rows:
            return table.raw_text

        descriptions = []

        # 逐行转换为描述
        for row in table.rows[:10]:  # 限制前10行
            row_desc = []
            for i, cell in enumerate(row):
                if i < len(table.headers):
                    header = table.headers[i]
                    value = cell
                    if value:
                        row_desc.append(f"{header}为{value}")

            if row_desc:
                descriptions.append("，".join(row_desc) + "。")

        if descriptions:
            table_desc = "。".join(descriptions)
            if context:
                return f"根据{context}中的表格数据显示，{table_desc}"
            return table_desc

        return table.raw_text

    @staticmethod
    def to_markdown(table: TableRegion) -> str:
        """
        表格转Markdown格式

        Args:
            table: 表格区域

        Returns:
            Markdown表格
        """
        if not table.headers:
            return table.raw_text

        # 构建Markdown表格
        lines = []

        # 表头
        headers = "| " + " | ".join(table.headers) + " |"
        lines.append(headers)
        lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")

        # 数据行
        for row in table.rows:
            cells = " | ".join(row)
            lines.append(f"| {cells} |")

        return '\n'.join(lines)

    @staticmethod
    def to_structured(table: TableRegion) -> Dict:
        """
        表格转结构化数据

        Args:
            table: 表格区域

        Returns:
            结构化数据字典
        """
        if not table.headers:
            return {"raw": table.raw_text}

        # 转换为列表字典
        data = []
        for row in table.rows:
            if len(row) == len(table.headers):
                row_dict = {}
                for i, header in enumerate(table.headers):
                    if i < len(row):
                        row_dict[header] = row[i]
                data.append(row_dict)

        return {
            "headers": table.headers,
            "row_count": len(data),
            "data": data
        }


class TableProcessor:
    """
    表格处理器 - 完整的表格处理流程
    """

    def __init__(self):
        self.detector = TableDetector()
        self.converter = TableConverter()

    def process_document(
        self,
        text: str,
        document_name: str = ""
    ) -> Dict:
        """
        处理文档中的所有表格

        Args:
            text: 文档文本
            document_name: 文档名称

        Returns:
            处理结果
        """
        logger.info(f"开始处理文档: {document_name}")

        # 1. 检测表格
        tables = self.detector.detect_tables(text)

        if not tables:
            logger.info("未检测到表格")
            return {
                'tables_found': 0,
                'processed': 0,
                'text': text
            }

        logger.info(f"检测到 {len(tables)} 个表格")

        # 2. 处理每个表格
        processed_text = text
        replacements = []

        for table in tables:
            logger.info(f"处理表格 (行 {table.start_line}-{table.end_line})")

            # 转换策略选择
            if len(table.headers) <= 5 and len(table.rows) <= 10:
                # 小表格：转为描述
                description = self.converter.to_description(
                    table,
                    f"{document_name}第{table.start_line}页"
                )
                replacements.append((table.raw_text, description))
                logger.info(f"  → 转为描述性文本 ({len(table.rows)}行)")

            else:
                # 大表格：保留为Markdown
                markdown = self.converter.to_markdown(table)
                replacements.append((table.raw_text, markdown))
                logger.info(f"  → 保留为Markdown格式 ({len(table.rows)}行)")

        # 3. 替换表格
        for original, replacement in replacements:
            processed_text = processed_text.replace(original, replacement)

        # 4. 统计
        stats = {
            'tables_found': len(tables),
            'converted_to_description': sum(1 for _, r in replacements if '为' in r[:50]),
            'converted_to_markdown': sum(1 for _, r in replacements if '| ' in r[:5]),
            'tables_processed': len(replacements)
        }

        logger.info(f"表格处理完成: {stats}")
        logger.info(f"  → 描述性文本: {stats['converted_to_description']} 个")
        logger.info(f"  → Markdown格式: {stats['converted_to_markdown']} 个")

        return {
            'text': processed_text,
            'stats': stats,
            'tables': tables
        }

    def process_document_with_table_chunks(
        self,
        text: str,
        document_name: str = ""
    ) -> Tuple[str, List[Dict]]:
        """
        处理文档，将表格作为独立chunks

        Args:
            text: 文档文本
            document_name: 文档名称

        Returns:
            (处理后的文本, 表格chunks列表)
        """
        tables = self.detector.detect_tables(text)

        table_chunks = []
        for i, table in enumerate(tables):
            # 创建表格chunk
            chunk = {
                'content': table.raw_text,
                'metadata': {
                    'type': 'table',
                    'table_id': f"{document_name}_table_{i+1}",
                    'rows': len(table.rows),
                    'columns': len(table.headers),
                    'headers': table.headers,
                    'start_line': table.start_line,
                    'end_line': table.end_line,
                    'source': document_name
                }
            }
            table_chunks.append(chunk)

            logger.info(f"表格chunk: {len(table.rows)}行 × {len(table.headers)}列")

        return text, table_chunks


# 便捷函数
def process_tables_in_text(text: str, document_name: str = "") -> str:
    """处理文本中的表格（便捷函数）"""
    processor = TableProcessor()
    result = processor.process_document(text, document_name)
    return result['text']
