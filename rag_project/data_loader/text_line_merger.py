"""
PDF文本行合并器 - 合并"非段落换行"

解决PDF转换后的布局换行问题，将属于同一逻辑段落的行合并
"""
import re
from typing import List, Tuple
from dataclasses import dataclass
from rag_project.utils.logger import logger


@dataclass
class MergeConfig:
    """行合并配置"""
    # 段落结束标记（中文）
    cn_end_marks = ('。', '！', '？', '；', '：', '」', '』', '》')

    # 段落结束标记（英文）
    en_end_marks = ('.', '!', '?', ';', ':', '"', "'", ')', ']', '>')

    # 章节标题标记
    chapter_patterns = (
        r'^第[一二三四五六七八九十百千零\d]+章',
        r'^Chapter\s+[IVXLCDM\d]+',
        r'^[\d]+\.[\d]+\s+',  # 1.1
        r'^[一二三四五六七八九十]+、',
        r'^【.*】',
    )

    # 列表项标记
    list_patterns = (
        r'^\s*[\d]+\.\s+',  # 1. 2. 3.
        r'^\s*[一二三四五六七八九十]+、',  # 一、二、三、
        r'^\s*[-*•]\s+',  # - * •
        r'^\s*[（\(]\d+[）\)]\s+',  # (1) （1）
    )

    # 连字符（行尾连字符，表示单词断开）
    hyphen_pattern = r'-\s*$'

    # 最小行长度（短行可能是标题）
    min_line_length = 5

    # 缩进阈值（超过此缩进可能是新段落）
    indent_threshold = 4


class TextLineMerger:
    """
    文本行合并器

    功能：
    1. 识别并合并同一逻辑段落的行
    2. 保留真正的段落分隔
    3. 处理列表、标题等特殊格式
    """

    def __init__(self, config: MergeConfig = None):
        """
        初始化合并器

        Args:
            config: 合并配置
        """
        self.config = config or MergeConfig()

    def is_paragraph_end(self, line: str) -> bool:
        """
        判断一行是否是段落结束

        Args:
            line: 文本行

        Returns:
            是否是段落结束
        """
        line = line.rstrip()

        # 空行是段落分隔符
        if not line:
            return True

        # 检查段落结束标记
        if line[-1] in self.config.cn_end_marks:
            return True

        # 英文段落结束标记
        if line[-1] in self.config.en_end_marks:
            return True

        # 检查是否是章节标题
        for pattern in self.config.chapter_patterns:
            if re.match(pattern, line):
                return True

        # 检查是否是列表项
        for pattern in self.config.list_patterns:
            if re.match(pattern, line):
                return True

        return False

    def is_short_line_new_paragraph(self, line: str, prev_line: str) -> bool:
        """
        判断短行是否是新段落

        Args:
            line: 当前行
            prev_line: 上一行

        Returns:
            是否是新段落
        """
        line = line.rstrip()

        # 短行且没有标点，可能是标题
        if len(line) < self.config.min_line_length:
            # 检查上一行是否以段落结束标记结尾
            if prev_line and prev_line[-1] not in self.config.cn_end_marks + self.config.en_end_marks:
                return True

        return False

    def has_hyphen_break(self, line: str) -> bool:
        """
        检查行尾是否有连字符（单词断开）

        Args:
            line: 文本行

        Returns:
            是否有连字符断开
        """
        return bool(re.search(self.config.hyphen_pattern, line))

    def get_line_indent(self, line: str) -> int:
        """
        获取行缩进

        Args:
            line: 文本行

        Returns:
            缩进空格数
        """
        return len(line) - len(line.lstrip())

    def merge_text(self, text: str) -> str:
        """
        合并文本中的非段落换行

        Args:
            text: 原始文本

        Returns:
            合并后的文本
        """
        lines = text.split('\n')
        merged_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i].rstrip()
            prev_line = merged_lines[-1] if merged_lines else ""
            prev_indent = self.get_line_indent(prev_line) if prev_line else 0
            curr_indent = self.get_line_indent(current_line) if current_line else 0

            # 空行 - 保留作为段落分隔
            if not current_line:
                merged_lines.append("")
                i += 1
                continue

            # 检查是否需要与上一行合并
            should_merge = False

            # 条件1: 上一行没有段落结束标记
            if prev_line and not self.is_paragraph_end(prev_line):
                # 检查缩进变化（缩进变化大可能是新段落）
                if abs(curr_indent - prev_indent) <= self.config.indent_threshold:
                    should_merge = True

            # 条件2: 上一行有连字符（单词断开）
            if prev_line and self.has_hyphen_break(prev_line):
                should_merge = True

            # 条件3: 短行且上一行未结束，可能是标题（不合并）
            if self.is_short_line_new_paragraph(current_line, prev_line):
                should_merge = False

            # 条件4: 当前行以小写字母开头（未句首），可能需要合并
            if prev_line and current_line and not prev_line[-1] in self.config.cn_end_marks + self.config.en_end_marks:
                # 英文：小写字母开头
                # 中文：任何字符（中文不分大小写）
                if current_line[0].islower() or '\u4e00' <= current_line[0] <= '\u9fff':
                    # 但要排除列表项
                    is_list_item = any(re.match(p, current_line) for p in self.config.list_patterns)
                    if not is_list_item:
                        should_merge = True

            # 执行合并或保留
            if should_merge and merged_lines:
                # 合并到上一行
                if merged_lines[-1]:
                    # 移除连字符（如果有）
                    if self.has_hyphen_break(merged_lines[-1]):
                        merged_lines[-1] = re.sub(self.config.hyphen_pattern, '', merged_lines[-1])
                        merged_lines[-1] += current_line
                    else:
                        merged_lines[-1] += ' ' + current_line
                else:
                    merged_lines.append(current_line)

                logger.debug(f"合并: '{prev_line[-30:]}' + '{current_line[:30]}'")
            else:
                # 保留为新行
                merged_lines.append(current_line)

            i += 1

        # 重新组合文本
        merged_text = '\n'.join(merged_lines)

        # 清理多余的空行（保留段落间的一个空行）
        merged_text = re.sub(r'\n{3,}', '\n\n', merged_text)

        return merged_text

    def merge_with_stats(self, text: str) -> Tuple[str, dict]:
        """
        合并文本并返回统计信息

        Args:
            text: 原始文本

        Returns:
            (合并后的文本, 统计信息)
        """
        original_lines = text.split('\n')
        merged_text = self.merge_text(text)
        merged_lines = merged_text.split('\n')

        stats = {
            'original_lines': len(original_lines),
            'merged_lines': len(merged_lines),
            'lines_reduced': len(original_lines) - len(merged_lines),
            'reduction_ratio': (len(original_lines) - len(merged_lines)) / len(original_lines) if original_lines else 0,
            'original_chars': len(text),
            'merged_chars': len(merged_text),
        }

        logger.info(f"行合并: {stats['original_lines']} → {stats['merged_lines']} 行 "
                   f"({stats['lines_reduced']} 行减少, {stats['reduction_ratio']:.1%})")

        return merged_text, stats


def merge_pdf_text(
    text: str,
    config: MergeConfig = None
) -> str:
    """
    合并PDF文本中的非段落换行（便捷函数）

    Args:
        text: PDF转换后的文本
        config: 合并配置

    Returns:
        合并后的文本
    """
    merger = TextLineMerger(config)
    return merger.merge_text(text)


# 测试代码
if __name__ == "__main__":
    # 测试用例
    test_text = """
这是一个很长的句子，因为PDF版面宽度限制，
它被强制分成了多行，但实际上应该是一段完整的文本。

这是第二段，它有明确的段落结束标记。
所以不应该与上一段合并。

这是一个列表：
1. 第一项内容，
   继续在下一行。
2. 第二项内容。
3. 第三项内容，

这是普通段落。

第一章 概述
这是第一章的内容，
它被分成了多行。
"""

    print("="*80)
    print("文本行合并测试")
    print("="*80)

    print("\n原始文本:")
    print(test_text)

    merger = TextLineMerger()
    merged_text, stats = merger.merge_with_stats(test_text)

    print("\n合并后文本:")
    print(merged_text)

    print("\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
