"""
完整重新索引流程 - 方案A
1. 批量行合并所有文档
2. 清空Milvus
3. 重新chunking
4. 重新embedding
5. 重新插入Milvus
"""
import subprocess
import sys
import time
from pathlib import Path
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.logger import logger


def step_1_merge_lines():
    """步骤1: 批量行合并"""
    print("\n" + "="*80)
    print("步骤1: 批量行合并所有文档")
    print("="*80)

    try:
        subprocess.run([
            sys.executable,
            "batch_merge_lines.py",
            "--dir", "知识库/知识库"
        ], check=True)
        print("✓ 行合并完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 行合并失败: {e}")
        return False


def step_2_clear_milvus():
    """步骤2: 清空Milvus集合"""
    print("\n" + "="*80)
    print("步骤2: 清空Milvus向量数据库")
    print("="*80)

    try:
        manager = MilvusManager("config/milvus_config.yaml")

        # 获取当前状态
        stats_before = manager.get_collection_stats()
        num_before = stats_before.get("num_entities", 0)

        print(f"当前向量数量: {num_before}")

        if num_before > 0:
            # 删除所有数据
            confirm = input("\n确认要删除所有数据吗？(输入 yes 确认): ")

            if confirm.lower() == 'yes':
                manager.delete_all()
                print("✓ 已清空Milvus集合")
            else:
                print("✗ 取消清空操作")
                return False
        else:
            print("集合为空，无需清空")

        return True
    except Exception as e:
        print(f"✗ 清空失败: {e}")
        return False


def step_3_reindex():
    """步骤3: 重新索引所有文档"""
    print("\n" + "="*80)
    print("步骤3: 重新索引所有文档（Chunking + Embedding + 存储）")
    print("="*80)

    try:
        subprocess.run([
            sys.executable,
            "chunk_all_documents.py"
        ], check=True)
        print("✓ 重新索引完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 重新索引失败: {e}")
        return False


def verify_results():
    """验证结果"""
    print("\n" + "="*80)
    print("验证结果")
    print("="*80)

    try:
        manager = MilvusManager("config/milvus_config.yaml")
        stats = manager.get_collection_stats()

        num_entities = stats.get("num_entities", 0)
        print(f"\nMilvus向量数量: {num_entities}")

        # 检查chunks文件
        chunks_files = [
            "data/all_chunks.json",
            "data/knowledge_base_txt_chunks.json"
        ]

        for chunks_file in chunks_files:
            p = Path(chunks_file)
            if p.exists():
                import json
                with open(p, encoding='utf-8') as f:
                    chunks = json.load(f)
                print(f"{chunks_file}: {len(chunks)} chunks")

        return True
    except Exception as e:
        print(f"验证失败: {e}")
        return False


def main():
    """主流程"""
    print("="*80)
    print("完整重新索引流程 - 方案A")
    print("="*80)

    print("\n此流程将:")
    print("1. 对所有TXT文档应用行合并")
    print("2. 清空Milvus向量数据库")
    print("3. 重新进行chunking")
    print("4. 重新生成embeddings")
    print("5. 重新插入Milvus")
    print("\n预计总时间: 30-60分钟")

    confirm = input("\n确认开始？(输入 yes 继续): ")
    if confirm.lower() != 'yes':
        print("已取消")
        return

    start_time = time.time()
    step_times = {}

    # 步骤1: 行合并
    step_start = time.time()
    if not step_1_merge_lines():
        print("\n✗ 行合并失败，终止流程")
        return
    step_times['行合并'] = time.time() - step_start

    # 步骤2: 清空Milvus
    step_start = time.time()
    if not step_2_clear_milvus():
        print("\n✗ 清空Milvus失败，终止流程")
        return
    step_times['清空Milvus'] = time.time() - step_start

    # 步骤3: 重新索引
    step_start = time.time()
    if not step_3_reindex():
        print("\n✗ 重新索引失败，终止流程")
        return
    step_times['重新索引'] = time.time() - step_start

    # 验证
    verify_results()

    # 总结
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("流程完成")
    print("="*80)

    print("\n各步骤耗时:")
    for step, duration in step_times.items():
        print(f"  {step}: {duration/60:.1f}分钟")

    print(f"\n总耗时: {total_time/60:.1f}分钟")

    print("\n下一步:")
    print("1. 运行检索测试: python test_retrieval.py")
    print("2. 对比之前的检索质量")
    print("3. 如有LLM，进行完整问答测试")


if __name__ == "__main__":
    main()
