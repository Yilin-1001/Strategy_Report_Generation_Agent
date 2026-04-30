"""
恢复中国通��航空2021.txt为原始版本（保留分页标记）
"""
from pathlib import Path
import fitz  # PyMuPDF
from rag_project.utils.logger import logger


def restore_original_txt():
    """重新转换PDF，恢复分页标记"""
    kb_path = Path("知识库/知识库")

    # 查找PDF文件
    pdf_files = list(kb_path.rglob("*.pdf"))
    target_pdf = None

    for pdf_file in pdf_files:
        if "2021" in pdf_file.name:
            target_pdf = pdf_file
            break

    if not target_pdf:
        logger.error("找不到中国通用航空2021.pdf")
        return None

    logger.info(f"找到PDF: {target_pdf}")

    # 查找对应的txt文件
    txt_files = list(kb_path.rglob("*.txt"))
    target_txt = None

    for txt_file in txt_files:
        if "2021" in txt_file.name and len(txt_file.name) < 30:
            target_txt = txt_file
            break

    if not target_txt:
        logger.error("找不到对应的txt文件")
        return None

    logger.info(f"目标TXT: {target_txt}")

    # 备份清洗后的文件
    backup_file = target_txt.parent / f"{target_txt.stem}_cleaned_backup.txt"
    import shutil
    shutil.copy(target_txt, backup_file)
    logger.info(f"已备份清洗后的文件到: {backup_file}")

    # 重新转换PDF
    logger.info("开始重新转换PDF...")
    doc = fitz.open(str(target_pdf))
    total_pages = len(doc)

    text_content = []

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # 添加分页标记
        page_header = f"\n{'='*80}\n"
        page_header += f"第 {page_num + 1} 页 / 共 {total_pages} 页\n"
        page_header += f"{'='*80}\n\n"

        text_content.append(page_header + text)

    doc.close()

    # 合并所有页面
    full_text = '\n'.join(text_content)

    # 保存到txt文件
    with open(target_txt, 'w', encoding='utf-8') as f:
        f.write(full_text)

    logger.info(f"已保存原始版本到: {target_txt}")
    logger.info(f"总页数: {total_pages}")
    logger.info(f"总字符数: {len(full_text):,}")

    # 清除转换日志中的该文件记录，这样下次转换会重新处理
    conversion_log = kb_path / "pdf_conversion_log.json"
    if conversion_log.exists():
        import json
        with open(conversion_log, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        # 删除该文件的转换记录
        if target_pdf.name in log_data:
            del log_data[target_pdf.name]

        with open(conversion_log, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        logger.info("已清除转换日志记录")

    logger.info("\n" + "="*80)
    logger.info("恢复完成！")
    logger.info("="*80)
    logger.info(f"原始文件: {target_txt}")
    logger.info(f"备份文件: {backup_file}")
    logger.info("="*80)

    return {
        'pdf': str(target_pdf),
        'txt': str(target_txt),
        'backup': str(backup_file),
        'pages': total_pages,
        'chars': len(full_text)
    }


if __name__ == "__main__":
    result = restore_original_txt()
    if result:
        logger.info("\n恢复成功！")
    else:
        logger.error("\n恢复失败")
