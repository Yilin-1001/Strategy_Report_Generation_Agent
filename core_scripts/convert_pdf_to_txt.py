"""
使用PyMuPDF批量将PDF转换为TXT文件
转换后的TXT文件保存在PDF原文件的同一目录下
"""
from pathlib import Path
import json
from datetime import datetime
from rag_project.utils.logger import logger
import fitz  # PyMuPDF


class PDFConverter:
    """PDF转文本转换器（使用PyMuPDF）"""

    def __init__(self, knowledge_base_path: str):
        """
        初始化转换器

        Args:
            knowledge_base_path: 知识库目录路径
        """
        self.kb_path = Path(knowledge_base_path)

        # 转换日志保存在知识库根目录
        self.conversion_log_path = self.kb_path / "pdf_conversion_log.json"
        self.log_data = self._load_log()

        logger.info("="*80)
        logger.info("PDF转换器初始化")
        logger.info(f"知识库路径: {self.kb_path}")
        logger.info(f"转换策略: TXT文件保存在PDF原文件同目录")
        logger.info(f"转换日志: {self.conversion_log_path}")
        logger.info("="*80)

    def _load_log(self):
        """加载转换日志"""
        if self.conversion_log_path.exists():
            try:
                with open(self.conversion_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法加载日志文件: {e}")
                return {}
        return {}

    def _save_log(self):
        """保存转换日志"""
        try:
            with open(self.conversion_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存日志失败: {e}")

    def convert_pdf(self, pdf_path: Path) -> dict:
        """
        转换单个PDF文件

        Args:
            pdf_path: PDF文件路径

        Returns:
            转换结果字典
        """
        pdf_name = pdf_path.name

        # 输出TXT文件保存在PDF原文件的同一目录下
        output_path = pdf_path.parent / f"{pdf_path.stem}.txt"

        result = {
            "pdf_name": pdf_name,
            "pdf_path": str(pdf_path),
            "output_path": str(output_path),
            "relative_path": str(pdf_path.relative_to(self.kb_path)),
            "status": "pending",
            "pages": 0,
            "chars": 0,
            "error": None
        }

        try:
            logger.info(f"[转换中] {pdf_name}")

            # 打开PDF
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)  # 在close之前保存页数

            # 提取文本
            text_content = []

            for page_num in range(total_pages):
                page = doc[page_num]

                # 获取文本（使用text模式，更适合中文）
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)

                # 添加分页标记
                page_header = f"\n{'='*80}\n"
                page_header += f"第 {page_num + 1} 页 / 共 {total_pages} 页\n"
                page_header += f"{'='*80}\n\n"

                text_content.append(page_header + text)

            # 关闭文档
            doc.close()

            # 合并所有页面
            full_text = '\n'.join(text_content)

            # 保存到文件（与PDF在同一目录）
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)

            # 统计信息
            result.update({
                "status": "success",
                "pages": total_pages,
                "chars": len(full_text),
                "file_size_kb": output_path.stat().st_size / 1024
            })

            # 显示相对路径，更易读
            rel_output = output_path.relative_to(self.kb_path)
            logger.info(f"[完成] {pdf_name} -> {rel_output} ({total_pages}页, {len(full_text)}字符)")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[失败] {pdf_name}: {e}")

        return result

    def convert_all_pdfs(self, skip_existing: bool = True):
        """
        批量转换所有PDF文件

        Args:
            skip_existing: 是否跳过已转换的文件
        """
        # 递归查找所有PDF文件（包括子目录）
        pdf_files = list(self.kb_path.rglob("*.pdf"))

        if not pdf_files:
            logger.warning(f"在 {self.kb_path} 中未找到PDF文件")
            return {
                "total": 0,
                "converted": 0,
                "failed": 0,
                "skipped": 0,
                "total_pages": 0,
                "total_chars": 0
            }

        logger.info(f"\n找到 {len(pdf_files)} 个PDF文件")
        logger.info("-"*80)

        stats = {
            "total": len(pdf_files),
            "converted": 0,
            "failed": 0,
            "skipped": 0,
            "total_pages": 0,
            "total_chars": 0
        }

        # 转换每个PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            pdf_name = pdf_path.name

            # 检查是否已转换
            if skip_existing and pdf_name in self.log_data:
                old_status = self.log_data[pdf_name].get("status")
                if old_status == "success":
                    logger.info(f"[{i}/{len(pdf_files)}] [跳过] {pdf_name} - 已转换")
                    stats["skipped"] += 1
                    stats["total_pages"] += self.log_data[pdf_name].get("pages", 0)
                    stats["total_chars"] += self.log_data[pdf_name].get("chars", 0)
                    continue

            # 转换PDF
            result = self.convert_pdf(pdf_path)

            # 更新日志
            self.log_data[pdf_name] = {
                "converted_at": datetime.now().isoformat(),
                **result
            }

            # 定期保存日志
            if i % 5 == 0:
                self._save_log()

            # 更新统计
            if result["status"] == "success":
                stats["converted"] += 1
                stats["total_pages"] += result["pages"]
                stats["total_chars"] += result["chars"]
            elif result["status"] == "failed":
                stats["failed"] += 1

        # 保存最终日志
        self._save_log()

        # 打印总结
        self._print_summary(stats)

        return stats

    def _print_summary(self, stats: dict):
        """打印转换总结"""
        logger.info("\n" + "="*80)
        logger.info("转换完成总结")
        logger.info("="*80)
        logger.info(f"总PDF文件: {stats['total']}")
        logger.info(f"成功转换: {stats['converted']}")
        logger.info(f"转换失败: {stats['failed']}")
        logger.info(f"跳过文件: {stats['skipped']}")
        logger.info(f"总页数: {stats['total_pages']}")
        logger.info(f"总字符数: {stats['total_chars']:,}")
        logger.info(f"转换策略: TXT文件保存在PDF原文件同目录")
        logger.info(f"转换日志: {self.conversion_log_path}")
        logger.info("="*80)


def main():
    """主函数"""
    knowledge_base_path = "知识库/知识库"

    # 创建转换器（TXT将保存在PDF原文件同目录）
    converter = PDFConverter(knowledge_base_path=knowledge_base_path)

    # 批量转换所有PDF
    stats = converter.convert_all_pdfs(skip_existing=True)

    # 如果有失败的，打印失败文件列表
    if stats['failed'] > 0:
        logger.warning("\n转换失败的文件:")
        for pdf_name, info in converter.log_data.items():
            if info.get("status") == "failed":
                logger.warning(f"  - {pdf_name}: {info.get('error')}")

    return stats['failed'] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
