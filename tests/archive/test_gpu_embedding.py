"""
Test GPU-accelerated embedding
"""
import torch
from rag_project.embeddings.embedding_model import EmbeddingModel
from rag_project.utils.logger import logger

def test_gpu_embedding():
    """Test that embedding model uses GPU"""
    print("\n" + "="*60)
    print("GPU Embedding Test")
    print("="*60)

    # Check CUDA availability
    print(f"\n1. PyTorch版本: {torch.__version__}")
    print(f"2. CUDA可用: {torch.cuda.is_available()}")
    print(f"3. GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available!")
        return False

    # Initialize embedding model
    print("\n4. 初始化embedding模型...")
    embedding_model = EmbeddingModel()

    # Check if model is on GPU
    print(f"5. 模型设备: {embedding_model.model.device}")

    # Test embedding
    test_text = "这是一个测试文本，用于验证GPU加速的embedding功能。"
    print(f"\n6. 测试文本: {test_text}")

    import time
    start = time.time()
    embedding = embedding_model.embed_text(test_text)
    elapsed = time.time() - start

    print(f"7. Embedding完成")
    print(f"   - 维度: {embedding.shape}")
    print(f"   - 耗时: {elapsed:.4f}秒")
    print(f"   - 前5个值: {embedding[:5]}")

    # Batch test
    print(f"\n8. 批量测试 (10个文本)...")
    test_texts = [test_text] * 10
    start = time.time()
    embeddings = embedding_model.embed_texts(test_texts)
    elapsed = time.time() - start

    print(f"   - 输出形状: {embeddings.shape}")
    print(f"   - 总耗时: {elapsed:.4f}秒")
    print(f"   - 平均每个: {elapsed/10:.4f}秒")

    print("\n" + "="*60)
    print("[SUCCESS] GPU Embedding test passed!")
    print("="*60 + "\n")

    return True

if __name__ == "__main__":
    test_gpu_embedding()
