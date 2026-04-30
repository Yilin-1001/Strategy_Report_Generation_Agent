"""
一键运行纯检索测试
1. 从chunks生成问题
2. 运行检索测试
3. 生成评估报告
"""
import subprocess
import sys
from pathlib import Path


def find_chunks_json():
    """查找chunks JSON文件"""
    # 可能的位置
    possible_paths = [
        "data/chunks_data.json",
        "data/all_chunks.json",
        "chunks.json"
    ]

    for path in possible_paths:
        if Path(path).exists():
            return path

    # 搜索当前目录
    for p in Path(".").rglob("*.json"):
        if "chunk" in p.name.lower():
            return str(p)

    return None


def main():
    print("="*80)
    print("RAG系统纯检索测试")
    print("="*80)

    # 1. 查找chunks文件
    print("\n[步骤1] 查找chunks数据文件...")
    chunks_path = find_chunks_json()

    if not chunks_path:
        print("错误: 未找到chunks JSON文件")
        print("请确保已经运行过chunking并保存了chunks数据")
        sys.exit(1)

    print(f"✓ 找到chunks文件: {chunks_path}")

    # 2. 生成测试问题
    print("\n[步骤2] 生成测试问题...")
    questions_path = "test_questions.json"

    try:
        subprocess.run([
            sys.executable,
            "question_generator.py",
            "--chunks", chunks_path,
            "--output", questions_path,
            "--num", "30"
        ], check=True)
        print(f"✓ 问题已生成: {questions_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ 问题生成失败: {e}")
        sys.exit(1)

    # 3. 运行检索测试
    print("\n[步骤3] 运行检索测试...")
    report_path = "retrieval_test_report.txt"

    try:
        subprocess.run([
            sys.executable,
            "test_retrieval.py",
            "--questions", questions_path,
            "--output", report_path,
            "--top-k", "5"
        ], check=True)
        print(f"✓ 测试完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 测试失败: {e}")
        sys.exit(1)

    # 4. 显示报告位置
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"测试问题: {questions_path}")
    print(f"测试报告: {report_path}")
    print("\n可以打开报告文件查看详细结果")


if __name__ == "__main__":
    main()
