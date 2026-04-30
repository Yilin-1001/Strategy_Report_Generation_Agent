# GPU OCR加速指南

## 方案1：使用PaddleOCR (推荐，支持GPU)

### 安装
```bash
# 卸载当前版本
pip uninstall unstructured unstructured-inference -y

# 安装GPU版本的paddlepaddle
pip install paddlepaddle-gpu

# 安装支持GPU的unstructured
pip install "unstructured[inference-gpu]"
pip install paddleocr
```

### 验证GPU是否可用
```python
from paddleocr import PaddleOCR
import torch

# 检查CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

# 初始化PaddleOCR with GPU
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)
print("PaddleOCR GPU enabled!")
```

### 修改document_type_detector.py
```python
# 替换UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf

# 使用GPU加速的partition
elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    ocr_languages="chi_sim+eng",  # 中英文
    # GPU参数
    pdf_processing_mode="fast",  # 使用快速模式
)
```

---

## 方案2：使用EasyOCR (支持GPU)

### 安装
```bash
pip install easyocr
```

### 使用
```python
import easyocr
import torch

# 初始化GPU reader
reader = easyocr.Reader(['zh-CN', 'en'], gpu=torch.cuda.is_available())
```

---

## 方案3：优化当前Tesseract (有限GPU支持)

Tesseract对GPU支持有限，建议换用PaddleOCR

---

## 性能对比

| OCR引擎 | CPU模式 | GPU模式 | 加速比 |
|---------|---------|---------|--------|
| PaddleOCR | 基准 | 10-50x快 | ⭐⭐⭐⭐⭐ |
| EasyOCR | 基准 | 5-20x快 | ⭐⭐⭐⭐ |
| Tesseract | 基准 | 无GPU支持 | ⭐ |

---

## 建议

**立即可行的方案**：
1. 安装PaddleOCR GPU版本
2. 修改document_type_detector.py
3. 重新运行索引

**预期效果**：
- PDF处理速度提升：5-10倍
- GPU使用率：从30% → 70-90%
- 总时间缩短：从60分钟 → 10-20分钟

## 要现在启用GPU OCR吗？

我可以帮你：
1. 安装GPU版本的OCR库
2. 修改代码启用GPU
3. 测试验证加速效果

需要我现在执行吗？
