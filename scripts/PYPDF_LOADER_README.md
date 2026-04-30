# PyPDF PDF解析器 - 使用指南

## 简介

这是一个基于 `pypdf` (原 PyPDF2) 的PDF解析工具，提供了完整的PDF文本提取功能。

### 特点

✓ **纯Python实现** - 无需额外依赖，安装简单
✓ **轻量级** - 代码简洁，易于维护
✓ **跨平台** - Windows/Linux/macOS通用
✓ **支持中文** - 正确处理UTF-8编码
✓ **功能完整** - 提取文本、元数据、章节信息

### 与PyMuPDF对比

| 特性 | PyPDF | PyMuPDF (fitz) |
|------|-------|----------------|
| **安装** | `pip install pypdf` | `pip install pymupdf` |
| **依赖** | 无 (纯Python) | 需要编译的C库 |
| **文本提取** | ✓ 好 | ✓✓ 优秀 |
| **图像处理** | ✓ 基础 | ✓✓ 强大 (OCR) |
| **速度** | 中等 | 快 |
| **文件大小** | 小 | 较大 |
| **中文支持** | ✓ | ✓ |
| **推荐场景** | 简单PDF、快速原型 | 复杂PDF、OCR需求 |

---

## 安装

```bash
# 方法1: 安装pypdf (推荐)
pip install pypdf

# 方法2: 安装PyPDF2 (兼容旧代码)
pip install PyPDF2
```

---

## 快速开始

### 1. 转换单个PDF

```python
from rag_project.data_loader.pypdf_loader import convert_pdf_with_pypdf

# 转换PDF为TXT
result = convert_pdf_with_pypdf(
    pdf_path="path/to/file.pdf",
    output_path="path/to/output.txt",  # 可选
    add_page_markers=True  # 添加分页标记
)

if result['status'] == 'success':
    print(f"转换成功: {result['output_path']}")
    print(f"页数: {result['metadata']['total_pages']}")
    print(f"字符: {result['stats']['total_characters']}")
```

### 2. 批量转换

```python
from rag_project.data_loader.pypdf_loader import batch_convert_pdfs_with_pypdf

# 批量转换所有PDF
stats = batch_convert_pdfs_with_pypdf(
    kb_path="知识库/知识库",
    skip_existing=True
)

print(f"转换完成: {stats['converted']}/{stats['total']}")
```

### 3. 高级用法

```python
from rag_project.data_loader.pypdf_loader import PyPDFLoader

# 创建加载器
loader = PyPDFLoader("file.pdf")

# 加载PDF
reader = loader.load_document()

# 提取元数据
metadata = loader.extract_metadata(reader)
print(f"标题: {metadata.title}")
print(f"作者: {metadata.author}")

# 提取所有页面
pages = loader.extract_all_pages(reader)
for page_info in pages:
    print(f"第 {page_info.page_number} 页: {page_info.char_count} 字符")

# 提取章节标题
sections = loader.extract_sections(reader)
for page_num, pos, title in sections:
    print(f"第 {page_num} 页: {title}")

# 保存为TXT
loader.save_to_txt(reader, "output.txt")

# 获取统计信息
stats = loader.get_document_stats(reader)
print(f"统计: {stats}")
```

---

## 命令行使用

### 转换单个PDF

```bash
python pypdf_batch_converter.py file.pdf
```

### 指定输出路径

```bash
python pypdf_batch_converter.py file.pdf output.txt
```

### 批量转换所有PDF

```bash
python pypdf_batch_converter.py
```

---

## 输出格式

### 分页标记格式

与 `convert_pdf_to_txt.py` 兼容：

```
================================================================================
第 1 页 / 共 100 页
================================================================================

[页面内容...]
```

### 元数据结构

```python
{
    'status': 'success',
    'pdf_path': 'path/to/file.pdf',
    'output_path': 'path/to/file.txt',
    'metadata': {
        'title': '文档标题',
        'author': '作者',
        'total_pages': 100,
    },
    'stats': {
        'total_pages': 100,
        'total_characters': 50000,
        'empty_pages': 2,
        'pages_with_images': 10,
        'avg_chars_per_page': 500.0
    },
    'duration_seconds': 3.5
}
```

---

## API参考

### PyPDFLoader类

#### `__init__(pdf_path: str, extract_images: bool = False)`

初始化PyPDF加载器。

**参数:**
- `pdf_path`: PDF文件路径
- `extract_images`: 是否提取图像信息

#### `load_document() -> PdfReader`

加载PDF文档，返回PdfReader对象。

#### `extract_metadata(reader: PdfReader) -> PDFDocumentInfo`

提取PDF元数据（标题、作者、创建日期等）。

#### `extract_page_text(reader: PdfReader, page_num: int) -> str`

提取单页文本。

**参数:**
- `reader`: PdfReader对象
- `page_num`: 页码（从0开始）

#### `extract_all_pages(reader: PdfReader) -> List[PDFPageInfo]`

提取所有页面信息。

#### `extract_with_page_markers(reader: PdfReader) -> str`

提取文本并添加分页标记。

#### `extract_sections(reader: PdfReader) -> List[Tuple[int, int, str]]`

提取章节标题。

**返回:** [(页码, 位置, 标题), ...]

#### `save_to_txt(reader: PdfReader, output_path: Optional[str] = None) -> str`

保存为TXT文件。

#### `get_document_stats(reader: PdfReader) -> Dict`

获取文档统计信息。

---

## 功能对比

### PyPDF vs PyMuPDF

#### 文本提取质量

**PyPDF:**
- ✓ 文本提取准确率高
- ✓ 保留基本格式
- ⚠ 复杂布局可能丢失
- ⚠ 表格提取较弱

**PyMuPDF:**
- ✓✓ 文本提取优秀
- ✓✓ 保留复杂布局
- ✓✓ 表格提取好
- ✓✓ 支持OCR

#### 性能

**PyPDF:**
- 处理速度: 中等
- 内存占用: 低
- 启动速度: 快

**PyMuPDF:**
- 处理速度: 快
- 内存占用: 中等
- 启动速度: 中等

---

## 使用建议

### 推荐使用PyPDF的场景：

1. **快速原型开发**
   - 无需复杂依赖
   - 安装简单

2. **简单PDF文档**
   - 纯文本PDF
   - 标准格式
   - 无复杂布局

3. **资源受限环境**
   - 低内存
   - 无GPU
   - 容器化部署

4. **跨平台部署**
   - 避免编译问题
   - 减少依赖

### 推荐使用PyMuPDF的场景：

1. **生产环境**
   - 需要最佳质量
   - 处理大量PDF

2. **复杂PDF**
   - 扫描PDF (需要OCR)
   - 多栏布局
   - 图像密集

3. **高级功能**
   - 图像提取
   - 表格解析
   - PDF生成

---

## 故障排除

### 问题1: ImportError: No module named 'pypdf'

**解决:**
```bash
pip install pypdf
```

### 问题2: 提取的文本为空

**原因:** PDF是图像扫描版，需要OCR

**解决:** 使用PyMuPDF + OCR
```python
# 使用PyMuPDF的GPU OCR加载器
from rag_project.data_loader.gpu_pdf_loader import GPUPDFLoader
```

### 问题3: 中文乱码

**原因:** PDF编码不是UTF-8

**解决:** PyPDF会自动检测编码，如果仍有问题：
```python
# 尝试指定编码
loader = PyPDFLoader(pdf_path)
# 在提取后转换编码
```

### 问题4: 某些页面提取失败

**原因:** 页面可能损坏或加密

**解决:**
```python
try:
    text = loader.extract_page_text(reader, page_num)
except Exception as e:
    logger.warning(f"页面 {page_num} 提取失败: {e}")
```

---

## 示例项目

### 示例1: 检查PDF质量

```python
from rag_project.data_loader.pypdf_loader import PyPDFLoader

loader = PyPDFLoader("file.pdf")
reader = loader.load_document()

# 获取统计
stats = loader.get_document_stats(reader)

# 检查空页面比例
empty_ratio = stats['empty_pages'] / stats['total_pages']
if empty_ratio > 0.5:
    print("警告: 超过50%页面为空，可能是图像PDF")
```

### 示例2: 提取目录

```python
loader = PyPDFLoader("file.pdf")
reader = loader.load_document()

# 提取章节
sections = loader.extract_sections(reader)

# 打印目录
print("目录:")
for page_num, pos, title in sections:
    print(f"  第 {page_num} 页: {title}")
```

### 示例3: 分页处理

```python
loader = PyPDFLoader("file.pdf")
reader = loader.load_document()

# 只处理特定页面范围
for i in range(10, 20):  # 第11-20页
    text = loader.extract_page_text(reader, i)
    print(f"第 {i+1} 页: {len(text)} 字符")
```

---

## 集成到RAG系统

### 与现有Chunking系统集成

```python
from rag_project.pipeline import RAGPipeline

# 现在可以直接使用转换后的TXT文件
pipeline = RAGPipeline()

# PyPDF转换的TXT文件可以正常chunking
chunks = pipeline.index_documents([
    "converted_file1.txt",
    "converted_file2.txt"
])
```

### 自定义处理流程

```python
from rag_project.data_loader.pypdf_loader import PyPDFLoader
from rag_project.data_loader.configurable_splitter import ConfigurableChunker

# 1. 使用PyPDF加载
loader = PyPDFLoader("file.pdf")
reader = loader.load_document()

# 2. 提取带分页标记的文本
text = loader.extract_with_page_markers(reader)

# 3. 保存为临时文件
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(text)
    temp_txt = f.name

# 4. 进行chunking
chunker = ConfigurableChunker()
# ... (后续处理)
```

---

## 更新日志

### v1.0.0 (2025-02-08)
- ✓ 初始版本发布
- ✓ 基础PDF文本提取
- ✓ 元数据提取
- ✓ 章节识别
- ✓ 分页标记支持
- ✓ 批量转换功能
- ✓ 与现有系统集成

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请联系项目维护者。
