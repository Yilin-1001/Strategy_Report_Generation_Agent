# RAG项目架构与脚本说明文档

生成时间: 2026-02-09
项目: RAG问答系统

---

## 目录

1. [项目架构概述](#一项目架构概述)
2. [核心代码模块](#二核心代码模块)
3. [主要脚本说明](#三主要脚本说明)
4. [数据流与调用关系](#四数据流与调用关系)
5. [完整工作流程](#五完整工作流程)
6. [配置文件说明](#六配置文件说明)

---

## 一、项目架构概述

### 1.1 整体架构

```
RAG Project/
├── rag_project/              # 核心代码包
│   ├── data_loader/         # 数据加载与处理
│   ├── embeddings/          # 向量嵌入
│   ├── storage/             # 向量数据库管理
│   ├── utils/               # 工具类
│   └── pipeline.py          # RAG Pipeline主入口
│
├── config/                   # 配置文件
│   ├── chunking_config.yaml # Chunking配置
│   └── milvus_config.yaml   # Milvus配置
│
├── data/                     # 数据目录
│   ├── all_chunks.json      # 所有chunks (5.8MB, 3,426个)
│   └── models/              # BGE-M3模型缓存
│
├── logs/                     # 日志目录
│
├── 知识库/知识库/           # 原始文档库
│   ├── 监管报告/
│   ├── 政策文献/
│   ├── 学术论文/
│   └── ...
│
└── [功能脚本]                # 核心运行脚本
    ├── convert_pdf_to_txt.py
    ├── batch_merge_lines.py
    ├── chunk_all_documents.py
    ├── question_generator.py
    └── test_retrieval.py
```

### 1.2 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| **文档处理** | PyMuPDF, LangChain | PDF解析、文档加载 |
| **文本分割** | LangChain TextSplitter | 按文档类型分割 |
| **嵌入模型** | BAAI/bge-m3 | 1024维向量，GPU加速 |
| **向量数据库** | Milvus (Docker) | HNSW索引 |
| **相似度计算** | Inner Product | 内积计算 |
| **编程语言** | Python 3.10+ | 类型注解完整 |

---

## 二、核心代码模块

### 2.1 RAG Pipeline (`rag_project/pipeline.py`)

**作用**: RAG系统的主入口，协调整个处理流程

**核心类**: `RAGPipeline`

**主要方法**:

```python
class RAGPipeline:
    def __init__(
        chunking_config_path: str,
        milvus_config_path: str,
        chunks_storage_path: Optional[str] = None
    )
    # 初始化所有组件

    def index_documents(file_paths: List[str]) -> int
    # 完整索引流程: 加载 → 分块 → 嵌入 → 存储

    def search(query: str, top_k: int = 5) -> List[Dict]
    # 检索相关文档

    def _load_and_chunk_file(file_path: str) -> List[Document]
    # 加载单个文件并分块
```

**处理流程**:
```
index_documents():
  1. 加载文档 → 2. 分割成chunks → 3. 生成embeddings → 4. 插入Milvus
```

---

### 2.2 数据加载模块 (`rag_project/data_loader/`)

#### 2.2.1 文档类型检测 (`document_type_detector.py`)

```python
def detect_doc_type(file_path: str) -> str
# 检测文档类型: 'policy', 'report', 'academic', 'news', 'other'

def get_loader_for_file(file_path: str)
# 根据文件类型返回对应的LangChain Loader
```

#### 2.2.2 可配置分割器 (`configurable_splitter.py`)

```python
class ConfigurableChunker:
    def __init__(config_path: str)
    # 从YAML加载分割配置

    def split_document(doc: Document) -> List[Document]
    # 根据文档类型使用不同分割策略
```

**支持的分割策略** (YAML配置):
```yaml
chunking:
  policies:
    chunk_size: 800
    chunk_overlap: 100
    separator: "\n\n"

  reports:
    chunk_size: 600
    chunk_overlap: 50
    separator: "\n"

  academic:
    chunk_size: 1000
    chunk_overlap: 150
```

#### 2.2.3 元数据提取器 (`metadata_extractor.py`)

```python
class MetadataExtractor:
    def extract(file_path: str, doc: Document) -> Dict
    # 提取文档元数据: source, doc_type, page_number等
```

**提取的元数据**:
- `source`: 文件名
- `doc_type`: 文档类型 (policy/report/academic/news)
- `page_number`: 页码
- `total_pages`: 总页数
- `char_count`: 字符数

#### 2.2.4 行合并模块 (`text_line_merger.py`)

**作用**: 合并非段落换行，修复PDF布局问题

```python
class TextLineMerger:
    def merge_text(text: str) -> str
    # 合并非段落换行

    def merge_with_stats(text: str) -> Tuple[str, Dict]
    # 合并并返回统计信息
```

**合并规则**:
- 识别非段落换行（行尾无句号、问号、感叹号）
- 将短行合并到上一行
- 保留段落结构

**效果**: 平均减少68.6%的行数

#### 2.2.5 Chunks存储 (`chunk_storage.py`)

```python
class ChunkStorage:
    def save_chunks_to_json(chunks: List[Document], path: str)
    # 保存chunks到JSON文件

    def load_chunks_from_json(path: str) -> List[Document]
    # 从JSON加载chunks
```

**JSON格式**:
```json
[
  {
    "text": "chunk内容...",
    "metadata": {
      "source": "文件名.txt",
      "doc_type": "report",
      "page_number": 1,
      "char_count": 450
    }
  }
]
```

---

### 2.3 嵌入模型模块 (`rag_project/embeddings/embedding_model.py`)

**作用**: 生成文本向量嵌入

```python
class EmbeddingModel:
    def __init__(config_path: str, load_on_init: bool = False)
    # 初始化BGE-M3模型

    def embed_documents(docs: List[Document]) -> np.ndarray
    # 批量生成文档embeddings (GPU加速)

    def embed_query(text: str) -> np.ndarray
    # 生成查询embedding
```

**模型信息**:
- 模型: `BAAI/bge-m3`
- 维度: 1024
- 设备: CUDA (GPU)
- 缓存路径: `data/models/`

---

### 2.4 向量存储模块 (`rag_project/storage/milvus_manager.py`)

**作用**: 管理Milvus向量数据库

```python
class MilvusManager:
    def __init__(config_path: str)
    # 连接Milvus服务器

    def create_collection(collection_name: str)
    # 创建Collection

    def insert_data(data: List[Dict])
    # 插入向量数据

    def search(query_vector: np.ndarray, top_k: int) -> List[Dict]
    # 向量检索

    def clear_collection(collection_name: str)
    # 清空Collection
```

**Collection结构**:
```
Collection: rag_documents
  - id (INT64): 主键
  - vector (FLOAT_VECTOR, dim=1024): 向量
  - text (VARCHAR): 文本内容
  - source (VARCHAR): 来源文件
  - doc_type (VARCHAR): 文档类型
  - page_number (INT): 页码
```

**索引**: HNSW, metric=INNER_PRODUCT

---

## 三、主要脚本说明

### 3.1 PDF转TXT (`convert_pdf_to_txt.py`)

**作用**: 批量将PDF文档转换为TXT文本

**功能**:
- 递归扫描 `知识库/知识库/` 目录
- 使用PyMuPDF解析PDF
- 保留页码标记 `[[Page N]]`
- 跳过已转换的文件

**输出**:
- 位置: 原文件所在目录
- 命名: `原文件名.txt`
- 数量: 10个PDF → 10个TXT

**数据流**:
```
PDF文件 → PyMuPDF解析 → 提取文本 → 插入页码标记 → 保存TXT
```

**使用场景**:
- 初期文档处理
- 新增PDF文档时

---

### 3.2 批量行合并 (`batch_merge_lines.py`)

**作用**: 对所有TXT文档应用行合并优化

**功能**:
- 扫描所有TXT文件（>10KB）
- 跳过已合并文件 (`*_merged.txt`)
- 备份原始文件到 `backup_original/`
- 生成 `*_merged.txt` 文件

**核心逻辑**:
```python
for txt_file in txt_files:
    1. 读取文本
    2. 调用 TextLineMerger.merge_with_stats()
    3. 保存为 *_merged.txt
    4. 记录统计信息
```

**输出示例**:
```
[1/21] 中国交通运输2021.txt
  行数: 2,548 → 802 (-68.5%)
```

**数据流**:
```
原始TXT → 行合并处理 → 优化TXT → 备份原始
```

**使用场景**:
- 提升文档质量
- 修复PDF布局问题
- Chunking前的预处理

---

### 3.3 批量Chunking (`chunk_all_documents.py`)

**作用**: 批量处理所有文档，生成chunks并索引到Milvus

**核心功能**:
```python
def get_all_txt_files(kb_path: str) -> List[str]
    # 优先使用merged版本
    # 过滤backup/metadata文件

def chunk_all_documents(
    kb_path: str = "知识库/知识库",
    chunks_output_path: str = "data/all_chunks.json"
) -> Dict
```

**处理逻辑**:
```
1. 扫描知识库 → 获取181个TXT文件
2. 过滤 → 优先使用*_merged.txt
3. 创建 RAGPipeline
4. 调用 pipeline.index_documents()
5. 保存chunks到JSON
6. 生成处理日志
```

**输出**:
- `data/all_chunks.json`: 3,426个chunks
- Milvus Collection: 3,426个向量
- `chunking_log.txt`: 处理日志

**数据流**:
```
TXT文件 → RAGPipeline.index_documents()
  → 加载文档 → 分割成chunks → 生成embeddings → 插入Milvus
  → 保存chunks到JSON
```

**使用场景**:
- 首次索引建立
- 重新索引
- 文档更新后

---

### 3.4 问题生成器 (`question_generator.py`)

**作用**: 基于文档chunks生成测试问题集

**核心类**: `QuestionGenerator`

**生成策略** (规则驱动):
```python
策略1: 年份 + 实体 → "统计数据"问题
  例: "2021年交通运输的投资情况如何？"

策略2: 机构名 + 职责关键词 → "机构职责"问题
  例: "某港口集团的主要工作是什么？"

策略3: 政策关键词 + 领域 → "政策措施"问题
  例: "如何加强安全生产管理？"

策略4: 多实体 → "综合概述"问题
  例: "关于交通运输的概况介绍"
```

**使用方式**:
```bash
python question_generator.py \
    --chunks data/all_chunks.json \
    --output test_questions_v2.json \
    --num 30
```

**输出格式**:
```json
[
  {
    "question": "2021年交通运输的数据情况如何？",
    "type": "统计数据",
    "source_chunk": "chunk内容...",
    "source_metadata": {
      "source": "中国交通运输2021.txt",
      "doc_type": "report",
      "page_number": 5
    }
  }
]
```

**数据流**:
```
chunks JSON → 随机采样 → 提取实体/数字 → 应用规则生成问题 → 保存JSON
```

**使用场景**:
- 检索质量测试
- 评估RAG系统性能
- 无人工标注问题时

---

### 3.5 检索测试 (`test_retrieval.py`)

**作用**: 测试RAG系统的检索质量

**核心类**: `RetrievalTester`

**测试流程**:
```python
1. 加载测试问题 (test_questions.json)
2. 初始化 RAGPipeline
3. 对每个问题:
   - 调用 pipeline.search(query, top_k=5)
   - 记录检索结果、评分、耗时
4. 分析结果:
   - 成功率
   - 平均评分
   - 评分分布
   - 按类型统计
5. 生成详细报告
```

**评估指标**:
- **检索成功率**: 找到结果的问题比例
- **平均Top-1评分**: 最高匹配分数的平均值
- **评分分布**: 高分(≥0.8)、中分[0.6-0.8)、低分(<0.6)
- **响应时间**: 平均/最快/最慢
- **按类型统计**: 各问题类型的平均分

**使用方式**:
```bash
python test_retrieval.py \
    --questions test_questions_v2.json \
    --output retrieval_test_report_v2.txt \
    --top-k 5
```

**输出**:
- 控制台: 实时测试进度
- `retrieval_test_report_v2.txt`: 详细报告

**数据流**:
```
问题JSON → RAGPipeline.search()
  → Milvus向量检索 → 返回Top-K结果
  → 计算指标 → 生成报告
```

**使用场景**:
- 优化前后对比
- 参数调优验证
- 系统性能监控

---

### 3.6 清空Milvus (`clear_milvus.py`)

**作用**: 清空Milvus Collection（重新索引前使用）

**简单直接**:

```python
from rag_project.storage.milvus_manager import MilvusManager

manager = MilvusManager("../config/milvus_config.yaml")
manager.clear_collection("rag_documents")
```

**使用场景**:
- 重新索引前
- 测试后清理
- 数据迁移

---

### 3.7 搜索演示 (`demo_search.py`)

**作用**: 交互式搜索演示

**功能**:
- 实时输入问题
- 显示Top-3检索结果
- 展示评分、来源、内容

**使用场景**:
- 手动测试检索效果
- 演示系统功能
- 调试查询问题

---

## 四、数据流与调用关系

### 4.1 完整数据流

```
┌──────────────────────────────────────────────────────────────┐
│                      原始文档                                  │
│  知识库/知识库/{监管报告, 政策文献, 学术论文}/                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 1: PDF → TXT 转换 (convert_pdf_to_txt.py)               │
│  - PyMuPDF解析                                                │
│  - 保留页码标记                                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 2: 行合并优化 (batch_merge_lines.py)                    │
│  - TextLineMerger处理                                         │
│  - 生成 *_merged.txt                                          │
│  - 备份原始文件                                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: Chunking + 索引 (chunk_all_documents.py)             │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ RAGPipeline.index_documents()                         │   │
│  │                                                        │   │
│  │  3.1 文档加载                                           │   │
│  │   ├─ document_type_detector.py                        │   │
│  │   └─ LangChain Loaders                                │   │
│  │                                                        │   │
│  │  3.2 分割Chunks                                        │   │
│  │   ├─ configurable_splitter.py (按类型)                │   │
│  │   └─ metadata_extractor.py                            │   │
│  │                                                        │   │
│  │  3.3 生成Embeddings                                    │   │
│  │   └─ embedding_model.py (BGE-M3 GPU)                  │   │
│  │                                                        │   │
│  │  3.4 插入Milvus                                        │   │
│  │   └─ milvus_manager.py                                │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  输出:                                                         │
│  - data/all_chunks.json (3,426 chunks)                        │
│  - Milvus Collection (3,426 vectors)                          │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 4: 问题生成 (question_generator.py)                     │
│  - 加载 chunks JSON                                           │
│  - 随机采样chunks                                             │
│  - 提取实体、数字                                             │
│  - 应用规则生成问题                                           │
│                                                               │
│  输出: test_questions_v2.json (30个问题)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 5: 检索测试 (test_retrieval.py)                         │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ for question in questions:                             │   │
│  │   results = pipeline.search(question, top_k=5)         │   │
│  │   # 记录评分、耗时                                      │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  输出: retrieval_test_report_v2.txt                           │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 模块调用关系图

```
convert_pdf_to_txt.py
  └─> pymupdf (外部库)

batch_merge_lines.py
  └─> rag_project.data_loader.text_line_merger.TextLineMerger
       └─> re (正则表达式)

chunk_all_documents.py
  └─> rag_project.pipeline.RAGPipeline
       ├─> rag_project.data_loader.configurable_splitter.ConfigurableChunker
       │    └─> rag_project.data_loader.document_type_detector
       │         └─> LangChain Loaders
       ├─> rag_project.data_loader.metadata_extractor.MetadataExtractor
       ├─> rag_project.embeddings.embedding_model.EmbeddingModel
       │    └─> sentence_transformers (BGE-M3)
       └─> rag_project.storage.milvus_manager.MilvusManager
            └─> pymilvus (Milvus SDK)

question_generator.py
  └─> [读取] data/all_chunks.json
       └─> re, random, json

test_retrieval.py
  └─> rag_project.pipeline.RAGPipeline
       ├─> rag_project.embeddings.embedding_model.EmbeddingModel
       └─> rag_project.storage.milvus_manager.MilvusManager
```

### 4.3 数据依赖关系

```
原始文档 (PDF)
    │
    ├─> [convert_pdf_to_txt.py]
    │     └─> TXT文件
    │
    ├─> [batch_merge_lines.py]
    │     └─> *_merged.txt
    │
    └─> [chunk_all_documents.py]
          ├─> data/all_chunks.json
          └─> Milvus Collection
                │
                ├─> [question_generator.py]
                │     └─> test_questions_v2.json
                │
                └─> [test_retrieval.py]
                      └─> retrieval_test_report_v2.txt
```

---

## 五、完整工作流程

### 5.1 首次索引流程

**目标**: 从零建立RAG系统索引

```
1. 转换PDF为TXT
   python convert_pdf_to_txt.py
   # 输出: 10个TXT文件

2. 行合并优化 (可选，推荐)
   python batch_merge_lines.py
   # 输出: 21个 *_merged.txt 文件
   # 效果: 行数减少68.6%

3. 批量Chunking + 索引
   python chunk_all_documents.py
   # 输出: 3,426个chunks + Milvus索引

4. 验证索引
   python demo_search.py
   # 手动测试检索效果
```

### 5.2 测试与评估流程

**目标**: 评估系统检索质量

```
1. 生成测试问题
   python question_generator.py \
       --chunks data/all_chunks.json \
       --output test_questions_v2.json \
       --num 30
   # 输出: 30个测试问题

2. 运行检索测试
   python test_retrieval.py \
       --questions test_questions_v2.json \
       --output retrieval_test_report_v2.txt
   # 输出: 详细测试报告

3. 查看报告
   cat retrieval_test_report_v2.txt
   # 评估: 平均评分、成功率、各类型表现
```

### 5.3 重新索引流程

**目标**: 数据优化后重新索引

```
1. 清空Milvus
   python clear_milvus.py

2. 执行首次索引流程步骤2-3
   python batch_merge_lines.py  # (如果需要)
   python chunk_all_documents.py

3. 重新测试
   python test_retrieval.py

4. 对比优化效果
   # 比较 retrieval_test_report.txt (优化前)
   # 和 retrieval_test_report_v2.txt (优化后)
```

### 5.4 新增文档流程

**目标**: 添加新文档到系统

```
1. 放置PDF到知识库
   知识库/知识库/新文档.pdf

2. 转换为TXT
   python convert_pdf_to_txt.py
   # 自动处理新增PDF

3. 行合并 (可选)
   python batch_merge_lines.py
   # 生成 *_merged.txt

4. 重新Chunking (全部)
   python chunk_all_documents.py
   # 自动使用merged版本

5. 验证新文档
   python demo_search.py
   # 搜索新文档内容
```

---

## 六、配置文件说明

### 6.1 Chunking配置 (`config/chunking_config.yaml`)

```yaml
chunking:
  # 政策文献 - 大chunk, 保持政策连贯性
  policies:
    chunk_size: 800
    chunk_overlap: 100
    separator: "\n\n"
    keep_separator: true

  # 监管报告 - 中等chunk
  reports:
    chunk_size: 600
    chunk_overlap: 50
    separator: "\n"
    keep_separator: false

  # 学术论文 - 大chunk, 保持论证完整
  academic:
    chunk_size: 1000
    chunk_overlap: 150
    separator: "\n\n"
    keep_separator: true

  # 新闻 - 小chunk, 快速定位
  news:
    chunk_size: 400
    chunk_overlap: 50
    separator: "\n"
    keep_separator: false

  # 默认配置
  default:
    chunk_size: 600
    chunk_overlap: 50
    separator: "\n"
    keep_separator: false
```

**参数说明**:
- `chunk_size`: 最大字符数
- `chunk_overlap`: chunk间重叠字符数
- `separator`: 分隔符
- `keep_separator`: 是否保留分隔符

### 6.2 Milvus配置 (`config/milvus_config.yaml`)

```yaml
milvus:
  host: localhost
  port: 19530

  collection_name: rag_documents

  # 向量维度 (BGE-M3)
  embedding_dim: 1024

  # 索引配置
  index_type: HNSW
  metric_type: INNER_PRODUCT
  index_params:
    M: 16
    efConstruction: 256

  # 搜索参数
  search_params:
    ef: 64

  # Embedding模型配置
  embedding:
    model_name: BAAI/bge-m3
    device: cuda  # GPU加速
    cache_dir: ./data/models
    batch_size: 32
```

**参数说明**:
- `host/port`: Milvus服务器地址
- `collection_name`: Collection名称
- `embedding_dim`: 向量维度
- `index_type`: 索引类型 (HNSW)
- `metric_type`: 相似度度量 (INNER_PRODUCT)
- `device`: 模型运行设备 (cuda/cpu)

---

## 七、关键性能指标

### 7.1 当前系统性能 (优化后)

| 指标 | 数值 |
|------|------|
| **文档数量** | 181个 |
| **Chunks数量** | 3,426个 |
| **向量维度** | 1024 |
| **索引类型** | HNSW (INNER_PRODUCT) |
| **嵌入模型** | BGE-M3 (GPU) |
| **检索成功率** | 100% |
| **平均评分** | 0.5776 |
| **最高评分** | 0.7788 |
| **平均响应时间** | 0.486秒 |
| **首次查询** | 12.160秒 (含模型加载) |
| **后续查询** | 0.066秒 |

### 7.2 优化效果对比

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **文档行数** | 42,094 | 13,224 | **-68.6%** |
| **Chunks数量** | 2,548 | 3,426 | +34.4% |
| **平均评分** | 0.5680 | 0.5776 | +1.7% |
| **最高评分** | 0.6873 | 0.7788 | **+13.3%** |
| **平均响应时间** | 0.446秒 | 0.486秒 | +9.0% |

---

## 八、常见问题与解决

### Q1: Chunking后Milvus为空？

**原因**: Milvus Collection未正确创建或插入失败

**解决**:
```bash
# 1. 检查Milvus服务
docker ps | grep milvus

# 2. 清空并重建
python clear_milvus.py
python chunk_all_documents.py

# 3. 查看日志
tail -f logs/rag_*.log
```

### Q2: GPU内存不足？

**现象**: `CUDA out of memory`

**解决**:
```yaml
# 修改 config/milvus_config.yaml
embedding:
  batch_size: 16  # 从32降到16
  device: cpu     # 或改用CPU
```

### Q3: 评分都很低 (<0.6)？

**可能原因**:
1. Chunks质量差 → 执行行合并
2. 查询表述不清晰 → 使用更具体的问题
3. 文档不匹配 → 检查知识库覆盖度

**优化方向**:
- 增加Reranker
- 调整chunking参数
- 查询扩展

### Q4: 检索速度慢？

**检查**:
```python
# 1. 确认GPU加速
embedding_model.py: device: cuda

# 2. 检查Milvus索引
milvus_config.yaml: index_type: HNSW

# 3. 减少Top-K
test_retrieval.py --top-k 3  # 从5降到3
```

---

## 九、下一步优化建议

### 9.1 短期优化 (1-2天)

**优先级1: 检索参数调优**
```python
# 调整 config/milvus_config.yaml
search_params:
  ef: 128  # 从64增加到128 (更精确)

# 调整 top_k
pipeline.search(query, top_k=10)  # 从5增加到10
```

**优先级2: 添加Reranker**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
# 重新排序Top-K结果
```

### 9.2 中期优化 (3-7天)

**集成DeepSeek LLM**
- 端到端问答测试
- 答案生成质量评估
- 添加引用溯源

**问题生成优化**
- 使用LLM生成更高质量问题
- 人工审核问题集
- 增加问题类型覆盖

### 9.3 长期优化 (持续)

**收集用户反馈**
- 真实查询日志分析
- 失败案例分析
- 迭代优化策略

**元数据增强**
- 添加时间、来源、类别等元数据
- 支持元数据过滤
- 提升检索精确度

---

## 十、文档版本

**版本**: v1.0
**生成时间**: 2026-02-09
**作者**: Claude (Sonnet 4.5)
**项目**: RAG问答系统

**更新记录**:
- v1.0 (2026-02-09): 初始版本

---

## 附录: 快速参考

### 核心命令速查

```bash
# PDF转TXT
python convert_pdf_to_txt.py

# 行合并优化
python batch_merge_lines.py

# 批量Chunking
python chunk_all_documents.py

# 生成测试问题
python question_generator.py --chunks data/all_chunks.json --output test_questions_v2.json --num 30

# 检索测试
python test_retrieval.py --questions test_questions_v2.json --output retrieval_test_report_v2.txt

# 清空Milvus
python clear_milvus.py

# 搜索演示
python demo_search.py
```

### 重要文件路径

```
配置:
  - config/chunking_config.yaml
  - config/milvus_config.yaml

数据:
  - data/all_chunks.json
  - data/models/

日志:
  - logs/rag_*.log

测试:
  - test_questions_v2.json
  - retrieval_test_report_v2.txt
```

---

**文档结束**
