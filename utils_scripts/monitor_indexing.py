"""
监控重新索引进度
"""
import time
import json
from pathlib import Path
from pymilvus import connections, Collection

def check_progress():
    """检查索引进度"""

    while True:
        print("\n" + "="*80)
        print("重新索引进度监控")
        print("="*80)

        # 1. 检查chunks文件
        chunks_file = Path("data/all_chunks.json")
        if chunks_file.exists():
            try:
                with open(chunks_file, encoding='utf-8') as f:
                    chunks = json.load(f)
                print(f"\nChunks文件: {len(chunks)} chunks")
                print(f"文件大小: {chunks_file.stat().st_size / 1024:.1f} KB")
                print(f"最后更新: {time.ctime(chunks_file.stat().st_mtime)}")
            except:
                print("Chunks文件: 正在生成...")
        else:
            print("\nChunks文件: 尚未生成")

        # 2. 检查Milvus
        try:
            connections.connect(alias='monitor', host='localhost', port=19530)
            collection = Collection('enterprise_docs')
            num_entities = collection.num_entities
            print(f"\nMilvus向量数: {num_entities}")
        except Exception as e:
            print(f"\nMilvus: {e}")

        # 3. 检查chunking进程
        import psutil
        found_process = False
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'chunk_all_documents.py' in ' '.join(cmdline):
                    found_process = True
                    print(f"\n索引进程: 运行中 (PID: {proc.pid})")
                    cpu = proc.cpu_percent()
                    mem = proc.memory_info().rss / 1024 / 1024
                    print(f"CPU: {cpu}%")
                    print(f"内存: {mem:.1f} MB")
                    break
            except:
                pass

        if not found_process:
            # 检查是否已完成
            if chunks_file.exists():
                print("\n索引进程: 已完成")
            else:
                print("\n索引进程: 未找到")

        print("\n" + "="*80)
        print("按Ctrl+C退出监控")
        print("30秒后刷新...")

        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break

if __name__ == "__main__":
    check_progress()
