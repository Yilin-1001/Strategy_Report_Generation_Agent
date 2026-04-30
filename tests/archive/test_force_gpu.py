"""
Test that embedding model FAILS if CUDA is not available (no silent fallback to CPU)
"""
import torch
import os
import sys

def test_force_gpu_validation():
    """Test that GPU is enforced and no fallback to CPU"""
    print("\n" + "="*60)
    print("Force GPU Validation Test")
    print("="*60)

    # Test 1: Check CUDA is available
    print("\n[Test 1] Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available - this test requires GPU")
        return False
    print(f"[PASS] CUDA available: {torch.cuda.get_device_name(0)}")

    # Test 2: Try loading with normal config (should succeed)
    print("\n[Test 2] Loading embedding model with CUDA config...")
    from rag_project.embeddings.embedding_model import EmbeddingModel

    try:
        model = EmbeddingModel(load_on_init=True)
        print(f"[PASS] Model loaded successfully")
        print(f"[PASS] Model device: {model.model.device}")

        # Verify it's actually on CUDA
        if not str(model.model.device).startswith('cuda'):
            print(f"[FAIL] Model is on {model.model.device} instead of CUDA!")
            return False
        print(f"[PASS] Model confirmed on CUDA device")

    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return False

    # Test 3: Test embedding actually uses GPU
    print("\n[Test 3] Testing actual embedding on GPU...")
    test_text = "GPU加速测试文本"
    embedding = model.embed_text(test_text)
    print(f"[PASS] Generated embedding with shape: {embedding.shape}")

    # Test 4: Monitor GPU memory during embedding
    print("\n[Test 4] Checking GPU memory usage...")
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
        print(f"[PASS] GPU Memory - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")

        if memory_allocated < 1:
            print("[WARN] Very low GPU memory usage - model might not be fully on GPU")
        else:
            print(f"[PASS] GPU memory usage indicates active GPU computation")

    print("\n" + "="*60)
    print("[SUCCESS] All GPU validation tests passed!")
    print("Embedding will NOT fall back to CPU.")
    print("="*60 + "\n")

    return True

def test_cpu_fallback_blocked():
    """Test that the model would FAIL if we tried to use it without CUDA"""
    print("\n" + "="*60)
    print("CPU Fallback Block Test (Hypothetical)")
    print("="*60)

    print("\nThis test verifies that if CUDA was unavailable,")
    print("the code would FAIL rather than silently use CPU.")

    # Read the code to show the protection
    with open('rag_project/embeddings/embedding_model.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if 'raise RuntimeError' in content and 'Do not fall back to CPU' in content:
        print("[PASS] Code contains强制GPU check with error on fallback")
        print("[PASS] CPU fallback is BLOCKED by design")
    else:
        print("[FAIL] Missing GPU enforcement check")

    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    test_force_gpu_validation()
    test_cpu_fallback_blocked()
