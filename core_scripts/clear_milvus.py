"""
清空Milvus向量数据库
"""
from rag_project.storage.milvus_manager import MilvusManager
from rag_project.utils.logger import logger
import sys


def clear_milvus(auto_confirm=False):
    """清空Milvus集合

    Args:
        auto_confirm: 是否自动确认，无需用户输入
    """
    print("="*80)
    print("清空Milvus向量数据库")
    print("="*80)

    manager = MilvusManager("config/milvus_config.yaml")

    # 获取当前状态
    stats_before = manager.get_collection_stats()
    num_before = stats_before.get("num_entities", 0)

    print(f"\n当前向量数量: {num_before}")

    if num_before > 0:
        should_delete = False

        if auto_confirm:
            print("\n自动确认模式: 将删除所有向量数据")
            should_delete = True
        else:
            print("\n警告: 此操作将删除所有向量数据！")
            print("请输入 'yes' 确认删除: ", end='')

            # 读取输入
            if sys.version_info[0] >= 3:
                confirm = input()
            else:
                confirm = raw_input()

            should_delete = (confirm.lower() == 'yes')

        if should_delete:
            # 删除集合
            manager.drop_collection()
            print("\n[OK] 已删除Milvus集合")

            # 重新创建空集合
            print("[INFO] 重新创建空集合...")
            manager.collection = manager._get_or_create_collection()

            # 验证
            stats_after = manager.get_collection_stats()
            num_after = stats_after.get("num_entities", 0)
            print(f"清空后向量数量: {num_after}")
            return True
        else:
            print("\n[CANCEL] 取消清空操作")
            return False
    else:
        print("集合为空，无需清空")
        return True


if __name__ == "__main__":
    # 检查命令行参数
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

    clear_milvus(auto_confirm=auto_confirm)
