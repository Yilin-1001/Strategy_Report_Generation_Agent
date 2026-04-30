# 基于RAG的检索型AI智能体 - 完整项目架构文档

**项目名称**：基于RAG的检索型AI智能体的设计及原型实现
**文档版本**：v1.0
**生成日期**：2026-03-10
**用途**：中期答辩PPT参考资料

---

## 目录

1. [系统架构概览](#系统架构概览)
2. [分层架构详解](#分层架构详解)
3. [核心模块说明](#核心模块说明)
4. [脚本依赖关系](#脚本依赖关系)
5. [数据流向](#数据流向)
6. [调用链路](#调用链路)
7. [运行指南](#运行指南)

---

## 系统架构概览

本项目采用**四层架构设计**，从上到下分别为：

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application)                      │
│              用户交互 & 批处理脚本                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      核心层 (Core)                           │
│                 RAG Pipeline 主控制逻辑                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    服务层 (Service)                          │
│           文档处理 | 向量化 | 存储                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 基础设施层 (Infrastructure)                   │
│           BGE-M3 | Milvus | 配置 | 数据存储                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 分层架构详解

### 第1层：应用层 (Application Layer)

**职责**：提供用户交互接口和批处理能力

| 脚本文件 | 功能描述 | 调用对象 |
|---------|---------|---------|
| **demo_search.py** | 交互式检索演示 | RAGPipeline.search() |
| **test_retrieval.py** | 自动化检索测试 | RAGPipeline.search() |
| **chunk_all_documents.py** | 批量索引主入口 ⭐ | RAGPipeline.index_documents() |
| **clear_milvus.py** | 数据库清理 | MilvusManager.clear_collection() |

**代码示例**：
```python
# demo_search.py - 交互式检索
from rag_project.pipeline import RAGPipeline

pipeline = RAGPipeline()
results = pipeline.search("江西省交通投资集团的主要业务", top_k=3)

# chunk_all_documents.py - 批量索引
file_paths = get_all_txt_files("知识库/知识库")
count = pipeline.index_documents(file_paths)
print(f"已索引 {count} 个chunks")
```

---

### 第2层：核心层 (Core Layer)

**职责**：RAG Pipeline核心控制逻辑

**主代码**：`rag_project/pipeline.py`

**核心类**：`RAGPipeline`

```python
class RAGPipeline:
    """RAG处理管道核心类"""

    def __init__(self):
        # 初始化三个核心服务
        self.chunker = ConfigurableChunker()      # 文档分块
        self.embedding_model = EmbeddingModel()    # 向量化
        self.milvus_manager = MilvusManager()      # 向量存储

    def index_documents(self, file_paths: List[str]) -> int:
        """
        索引文档流程
        输入: 文件路径列表
        输出: 索引的chunk数量
        """
        # 1. 加载+分块
        all_chunks = []
        for file_path in file_paths:
            chunks = self._load_and_chunk_file(file_path)
            all_chunks.extend(chunks)

        # 2. 向量化
        embeddings = self.embedding_model.embed_documents(all_chunks)

        # 3. 存储到Milvus
        milvus_data = self._prepare_milvus_data(all_chunks, embeddings)
        self.milvus_manager.insert_data(milvus_data)

        return len(all_chunks)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索流程
        输入: 查询文本, 返回数量
        输出: [{score, text, metadata}, ...]
        """
        # 1. 查询向量化
        query_vector = self.embedding_model.embed_text(query)

        # 2. Milvus检索
        results = self.milvus_manager.search(query_vector, top_k)

        return results
```

**关键方法**：
- `index_documents()`: 完整索引流程
- `search()`: 向量检索
- `_load_and_chunk_file()`: 单文件处理

---

### 第3层：服务层 (Service Layer)

**职责**：提供具体的功能服务模块

#### 3.1 文档处理模块

**目录**：`rag_project/data_loader/`

| 文件 | 类/函数 | 功能 |
|------|--------|------|
| **document_type_detector.py** | `detect_doc_type()` | 识别文档类型(news/report/regulation/tender) |
| **configurable_splitter.py** | `ConfigurableChunker` | 类型感知分块器 |
| **metadata_builder_v2.py** | `MetadataBuilderV2` | 增强元数据生成器 |
| **text_line_merger.py** | `TextLineMerger` | 行合并优化器 |

**文档类型检测**：
```python
# document_type_detector.py
def detect_doc_type(file_path: str) -> str:
    """检测文档类型"""
    keywords = {
        'news': ['新闻', '消息', '快讯'],
        'report': ['报告', '研报', '年报'],
        'regulation': ['规范', '规定', '办法'],
        'tender': ['招标', '投标']
    }
    # 返回: 'news' | 'report' | 'regulation' | 'tender' | 'default'
```

**类型感知分块**：
```python
# configurable_splitter.py
class ConfigurableChunker:
    def split_documents(self, docs, doc_type):
        """根据文档类型选择对应的chunker"""

        chunkers = {
            'news': NewsChunker(500, 50),          # 新闻: 500字
            'report': ReportChunker(1000, 200),    # 报告: 1000字
            'regulation': RegulationChunker(1500), # 规范: 1500字
            'tender': TenderChunker(1200, 150),    # 招标: 1200字
            'default': DefaultChunker(800, 100)    # 默认: 800字
        }

        return chunkers[doc_type].split(docs)
```

**元数据生成**：
```python
# metadata_builder_v2.py
class MetadataBuilderV2:
    def build(self, doc, source):
        """生成增强元数据"""
        return {
            'doc_type': self._detect_type(doc),
            'source': source,
            'title': self._extract_title(doc),
            'publish_date': self._extract_date(source),
            'tags': self._extract_tags(doc),
            'page_number': doc.metadata.get('page_number'),
            'char_count': len(doc.page_content)
        }
```

---

#### 3.2 向量化模块

**目录**：`rag_project/embeddings/`

| 文件 | 类 | 功能 |
|------|-----|------|
| **embedding_model.py** | `EmbeddingModel` | BGE-M3模型封装 |

**模型配置**：
```python
# embedding_model.py
class EmbeddingModel:
    def __init__(self):
        self.model_name = "BAAI/bge-m3"  # 中文优化模型
        self.dimension = 1024             # 向量维度
        self.device = "cuda"              # GPU加速
        self.batch_size = 32              # 批处理大小

    def embed_documents(self, docs):
        """批量向量化文档"""
        # 输入: List[Document]
        # 输出: np.ndarray (shape: [n, 1024])
        # 耗时: ~8.5分钟 (3,422个chunks, 107 batches)

    def embed_text(self, text):
        """单个文本向量化"""
        # 输入: str
        # 输出: np.ndarray (shape: [1024])
        # 耗时: ~0.06秒 (后续查询)
```

**性能指标**：
- 首次加载：~10秒
- 批量处理：32 chunks/batch
- 单个查询：0.06秒（已加载模型）

---

#### 3.3 存储模块

**目录**：`rag_project/storage/`

| 文件 | 类 | 功能 |
|------|-----|------|
| **milvus_manager.py** | `MilvusManager` | Milvus数据库管理 |

**Milvus集合结构**：
```python
# milvus_manager.py
class MilvusManager:
    def __init__(self):
        self.collection_name = "enterprise_docs"
        self.index_type = "HNSW"           # 高性能索引
        self.metric_type = "INNER_PRODUCT" # 相似度度量

    def create_collection(self):
        """创建集合"""
        schema = [
            FieldSchema("id", VARCHAR, max_length=100, is_primary=True),
            FieldSchema("vector", FLOAT_VECTOR, dim=1024),
            FieldSchema("text", VARCHAR, max_length=65535),
            FieldSchema("doc_type", VARCHAR, max_length=50),
            FieldSchema("source", VARCHAR, max_length=255),
            FieldSchema("publish_date", INT64),
            FieldSchema("page_number", INT64),
            FieldSchema("title", VARCHAR, max_length=255)
        ]
        # 创建HNSW索引

    def insert_data(self, data):
        """插入向量数据"""
        # 输入: List[Dict] with id, vector, text, metadata
        # self.collection.insert(data)

    def search(self, vector, top_k):
        """向量检索"""
        # 输出: [{score, text, metadata}, ...]
```

**集合信息**：
- 集合名：`enterprise_docs`
- 向量数量：3,422
- 索引类型：HNSW（层次化小世界图）
- 相似度度量：内积（INNER_PRODUCT）

---

#### 3.4 工具模块

**目录**：`rag_project/utils/`

| 文件 | 功能 |
|------|------|
| **config_loader.py** | YAML配置加载器 |
| **logger.py** | 结构化日志输出 |
| **chunk_storage.py** | Chunks JSON存储 |

**配置文件**：
```yaml
# config/chunking_config.yaml
chunking:
  news:
    chunk_size: 500
    chunk_overlap: 50
  report:
    chunk_size: 1000
    chunk_overlap: 200
  regulation:
    chunk_size: 1500
    chunk_overlap: 0
  default:
    chunk_size: 800
    chunk_overlap: 100

# config/milvus_config.yaml
milvus:
  host: localhost
  port: 19530
  collection:
    name: enterprise_docs
    dimension: 1024
    index_type: HNSW
```

---

### 第4层：基础设施层 (Infrastructure Layer)

**职责**：外部依赖和数据存储

| 组件 | 类型 | 说明 |
|------|------|------|
| **BGE-M3模型** | Embedding模型 | BAAI/bge-m3, 1024维, 中文优化 |
| **Milvus** | 向量数据库 | Docker部署, HNSW索引 |
| **知识库** | 原始数据 | 180个TXT文档 |
| **chunks.json** | 数据备份 | 3,422个chunks + 元数据 |

**Docker服务**：
```yaml
# docker-compose.yml
services:
  milvus-standalone:
    image: milvusdb/milvus:latest
    ports: "19530:19530"

  etcd:
    image: quay.io/coreos/etcd:latest

  minio:
    image: minio/minio:latest
```

**数据目录**：
```
data/
├── models/              # BGE-M3模型缓存 (~2.2GB)
├── all_chunks.json      # 3,422个chunks备份
└── logs/                # 运行日志

知识库/知识库/
├── 中华人民共和国交通运输部/
├── 江西省交通投资集团有限责任公司/
├── 供应链上下游企业/
├── 相关研报/
└── 相关论文/
```

---

## 脚本依赖关系

### 完整依赖图

```
convert_pdf_to_txt.py
    │
    ├─ 依赖: pymupdf/fitz
    ├─ 输入: 知识库/**/*.pdf
    └─ 输出: 知识库/**/*.txt
         │
         ▼
batch_merge_lines.py
    │
    ├─ 依赖: text_line_merger.py
    ├─ 输入: 所有*.txt (过滤backup/元数据)
    └─ 输出: *_merged.txt (减少68.6%行数)
         │
         ▼
chunk_all_documents.py ⭐主入口
    │
    ├─ 依赖: pipeline.py
    ├─ 输入: 180个TXT文档 (优先merged版本)
    └─ 调用: RAGPipeline.index_documents()
         │
         ▼
pipeline.py (RAGPipeline)
    │
    ├─ document_type_detector.py ─→ 识别文档类型
    ├─ configurable_splitter.py ───→ 类型感知分块
    ├─ metadata_builder_v2.py ────→ 生成元数据
    ├─ embedding_model.py ────────→ BGE-M3向量化
    └─ milvus_manager.py ──────────→ Milvus存储
         │
         ▼
demo_search.py / test_retrieval.py
    │
    └─ 调用: RAGPipeline.search(query, top_k)
         │
         ▼
检索结果 (score + text + metadata)
```

### 依赖关系表

| 脚本 | 依赖模块 | 输出 | 被依赖 |
|------|---------|------|--------|
| **convert_pdf_to_txt.py** | pymupdf | .txt文件 | batch_merge_lines.py |
| **batch_merge_lines.py** | text_line_merger.py | *_merged.txt | chunk_all_documents.py |
| **clear_milvus.py** | milvus_manager.py | 空集合 | chunk_all_documents.py |
| **chunk_all_documents.py** | pipeline.py | all_chunks.json | demo_search.py |
| **demo_search.py** | pipeline.py | 检索结果 | - |
| **test_retrieval.py** | pipeline.py | 测试报告 | - |
| **pipeline.py** | 所有服务模块 | - | chunk_all_documents.py |

---

## 数据流向

### 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     阶段1: 文档预处理                       │
└─────────────────────────────────────────────────────────────┘

原始PDF文档 (10个)
    │
    ├─ [convert_pdf_to_txt.py] PyMuPDF解析
    │  └─ 提取文本 + 插入[[Page N]]标记
    │
    ▼
TXT文件 (197个)
    │
    ├─ [batch_merge_lines.py] 行合并优化
    │  ├─ TextLineMerger合并非段落换行
    │  ├─ 减少行数68.6% (42,094 → 13,224)
    │  └─ 生成 *_merged.txt (21个)
    │
    ▼
优化后TXT (180个去重)
    │
    └─ 优先使用 *_merged.txt 版本


┌─────────────────────────────────────────────────────────────┐
│                     阶段2: 文档分块                         │
└─────────────────────────────────────────────────────────────┘

TXT文件 (180个)
    │
    ├─ [chunk_all_documents.py] 扫描知识库
    │  ├─ 过滤backup/元数据文件
    │  ├─ 去重处理 (17个重复文件)
    │  └─ 最终处理180个文件
    │
    ▼
[RAGPipeline.index_documents()]
    │
    ├─ document_type_detector → 识别类型
    ├─ LangChain Loader → 加载文档
    ├─ MetadataBuilderV2 → 生成元数据
    │
    ▼
[ConfigurableChunker.split_documents()]
    │
    ├─ NewsChunker (500字) → 新闻类
    ├─ ReportChunker (1000字) → 报告类
    ├─ RegulationChunker (1500字) → 规范类
    ├─ TenderChunker (1200字) → 招标类
    └─ DefaultChunker (800字) → 其他
    │
    ▼
3,422个Chunks
    │
    ├─ 保存: data/all_chunks.json
    └─ 传递给向量化模块


┌─────────────────────────────────────────────────────────────┐
│                     阶段3: 向量化                           │
└─────────────────────────────────────────────────────────────┘

3,422个Chunks
    │
    ├─ [EmbeddingModel.embed_documents()]
    │  ├─ 模型: BAAI/bge-m3
    │  ├─ 设备: NVIDIA RTX 3050 Ti (GPU)
    │  ├─ 批处理: 32 chunks/batch
    │  ├─ 总批次: 107 batches
    │  └─ 耗时: ~8.5分钟
    │
    ▼
3,422个向量 (1024维)


┌─────────────────────────────────────────────────────────────┐
│                     阶段4: 向量存储                         │
└─────────────────────────────────────────────────────────────┘

3,422个向量 + 元数据
    │
    ├─ [MilvusManager.insert_data()]
    │  ├─ 集合: enterprise_docs
    │  ├─ 索引: HNSW (INNER_PRODUCT)
    │  └─ 存储向量+元数据
    │
    ▼
Milvus数据库
    ├─ 向量数量: 3,422
    ├─ 向量维度: 1024
    ├─ 索引类型: HNSW
    └─ 元数据字段: 8个


┌─────────────────────────────────────────────────────────────┐
│                     阶段5: 检索服务                         │
└─────────────────────────────────────────────────────────────┘

用户查询
    │
    ├─ [demo_search.py] 或 [test_retrieval.py]
    │
    ▼
[RAGPipeline.search(query, top_k=5)]
    │
    ├─ [EmbeddingModel.embed_text(query)]
    │  └─ 生成查询向量 (1024维)
    │     耗时: ~0.06秒
    │
    ▼
[MilvusManager.search(vector, top_k=5)]
    │
    ├─ HNSW向量检索
    │  └─ 相似度计算 (内积)
    │     耗时: ~0.4秒
    │
    ▼
Top-K结果
    │
    └─ 返回: [{score, text, metadata}, ...]
       ├─ score: 相似度分数
       ├─ text: chunk内容
       └─ metadata: {doc_type, source, page_number, ...}


总响应时间: ~0.5秒 (首次查询12秒含模型加载)
```

---

## 调用链路

### 索引流程调用链

```python
# 1. 用户触发索引
chunk_all_documents.py
    │
    ├─► get_all_txt_files("知识库/知识库")
    │   └─ 返回: 180个文件路径 (优先merged版本)
    │
    ├─► RAGPipeline.__init__()
    │   ├─ ConfigurableChunker(config_path)
    │   ├─ EmbeddingModel(config_path)
    │   └─ MilvusManager(config_path)
    │
    ├─► RAGPipeline.index_documents(file_paths)
    │   │
    │   ├─► 遍历文件
    │   │   └─► _load_and_chunk_file(file)
    │   │       │
    │   │       ├─► detect_doc_type(file)
    │   │       │   └─ 返回: 'news' | 'report' | 'regulation' | 'tender'
    │   │       │
    │   │       ├─► get_loader_for_file(file)
    │   │       │   └─ LangChain Loader.load()
    │   │       │       └─ 返回: List[Document]
    │   │       │
    │   │       ├─► MetadataBuilderV2.build(doc, source)
    │   │       │   └─ 返回: {doc_type, title, publish_date, tags, ...}
    │   │       │
    │   │       └─► ConfigurableChunker.split_documents(docs, doc_type)
    │   │           ├─ NewsChunker.split()
    │   │           ├─ ReportChunker.split()
    │   │           ├─ RegulationChunker.split()
    │   │           ├─ TenderChunker.split()
    │   │           └─ DefaultChunker.split()
    │   │               └─ 返回: List[Document] (chunks)
    │   │
    │   ├─► EmbeddingModel.embed_documents(all_chunks)
    │   │   ├─ 批处理: 32 chunks/batch
    │   │   ├─ 总批次: 107 batches
    │   │   └─ 返回: np.ndarray (3422, 1024)
    │   │
    │   └─► MilvusManager.insert_data(milvus_data)
    │       ├─ 准备数据: {id, vector, text, metadata}
    │       └─ 插入集合: enterprise_docs
    │
    └─► 返回: 3422 (索引的chunk数量)


输出结果:
    ├─ data/all_chunks.json (3,422个chunks)
    ├─ Milvus: enterprise_docs (3,422个向量)
    └─ chunking_log.txt (处理日志)
```

### 检索流程调用链

```python
# 2. 用户发起查询
demo_search.py
    │
    ├─► RAGPipeline.__init__()
    │   └─ (同上，初始化三个服务)
    │
    ├─► RAGPipeline.search(query, top_k=5)
    │   │
    │   ├─► EmbeddingModel.embed_text(query)
    │   │   ├─ 输入: "江西省交通投资集团的主要业务"
    │   │   ├─ 模型: BGE-M3
    │   │   └─ 返回: np.ndarray (1024,) 查询向量
    │   │
    │   ├─► MilvusManager.search(query_vector, top_k=5)
    │   │   ├─ 输入: 查询向量 (1024维)
    │   │   ├─ 检索: HNSW索引 + 内积计算
    │   │   └─ 返回: List[Dict]
    │   │       ├─ score: 0.85 (相似度分数)
    │   │       ├─ text: "chunk内容..."
    │   │       └─ metadata: {doc_type, source, page_number, ...}
    │   │
    │   └─► 返回结果给用户
    │
    └─► 显示检索结果
        ├─ 结果1: score=0.85, 江西省交通投资集团...
        ├─ 结果2: score=0.78, 集团业务范围...
        ├─ 结果3: score=0.72, 主要职能...
        ├─ 结果4: score=0.65, 投资方向...
        └─ 结果5: score=0.60, 发展规划...


响应时间:
    ├─ 首次查询: ~12秒 (含模型加载10秒)
    └─ 后续查询: ~0.5秒
```

---

## 运行指南

### 环境准备

```bash
# 1. 启动Milvus (Docker)
docker-compose up -d

# 2. 检查服务状态
docker ps
# 应该看到: milvus-standalone, etcd, minio

# 3. 安装Python依赖
pip install -r requirements.txt

# 主要依赖:
# - pymilvus (Milvus客户端)
# - transformers (BGE-M3模型)
# - torch (PyTorch + CUDA)
# - langchain (RAG框架)
# - pymupdf (PDF解析)
```

### 完整运行流程

#### 步骤1: 文档预处理 (可选，如已有TXT可跳过)

```bash
# 1.1 PDF转TXT
python convert_pdf_to_txt.py

# 功能:
# - 扫描 知识库/知识库/ 下的所有PDF文件
# - 使用PyMuPDF解析PDF
# - 提取文本并插入[[Page N]]页码标记
# - 保存为同名的.txt文件

# 输出示例:
# [1/10] 处理中: 中国交通运输2021.pdf
#   → 输出: 中国交通运输2021.txt
# [2/10] 处理中: ...
```

```bash
# 1.2 行合并优化 (强烈推荐)
python batch_merge_lines.py

# 功能:
# - 扫描所有TXT文件 (>10KB)
# - 跳过已合并文件 (*_merged.txt)
# - 调用TextLineMerger合并非段落换行
# - 备份原始文件到backup_original/
# - 生成 *_merged.txt 文件

# 优化效果:
# - 减少行数: 68.6% (42,094 → 13,224行)
# - 提升chunk质量
# - 减少检索噪音

# 输出示例:
# [1/21] 中国交通运输2021.txt
#   行数: 2,548 → 802 (-68.5%)
#   保存: *_merged.txt
#   备份: backup_original/
```

#### 步骤2: 建立向量索引

```bash
# 2.1 清空旧数据 (如需要重新索引)
python clear_milvus.py --yes

# 功能:
# - 清空enterprise_docs集合
# - 删除所有向量数据
# - 重新创建空集合

# 输出:
# ✓ 删除旧数据: 3426 条记录
# ✓ 重新创建空集合
# ✓ 当前向量数量: 0
```

```bash
# 2.2 批量分块+索引 (核心步骤) ⭐
python chunk_all_documents.py

# 功能:
# - 扫描知识库目录 (197个文件)
# - 去重处理 (保留merged版本)
# - 调用RAGPipeline.index_documents()
# - 完整流程: 加载→���块→向量化→存储

# 处理流程:
# [1/180] 江西省交通投资集团.txt
#   类型: report
#   Chunks: 45
# [2/180] ...
# ...
# ✓ 扫描文件总数: 197
# ✓ 去重后文件数: 180
# ✓ 生成chunks数: 3,422
# ✓ 处理时间: 524.34秒 (~8.7分钟)

# 输出文件:
# - data/all_chunks.json (3,422个chunks)
# - chunking_log.txt (处理日志)
# - Milvus: enterprise_docs (3,422个向量)
```

#### 步骤3: 验证检索功能

```bash
# 3.1 交互式检索测试
python demo_search.py

# 功能:
# - 实时输入查询问题
# - 显示Top-3检索结果
# - 展示评分、来源、内容

# 使用示例:
# 输入查询: 江西省交通投资集团的主要业务是什么？
#
# 结果1: score=0.6363
# 来源: 江西省交通投资集团有限责任公司.txt
# 内容: 省交投集团主要承担省内...
#
# 结果2: score=0.5899
# 来源: ...
#
# 输入查询: exit (退出)
```

```bash
# 3.2 自动化检索测试
python test_retrieval.py \
    --questions test_questions_v2.json \
    --output retrieval_test_report_v2.txt \
    --top-k 5

# 功能:
# - 加载测试问题集
# - 逐个执行检索
# - 计算评估指标
# - 生成详细报告

# 测试结果:
# ✓ 测试问题数: 30
# ✓ 检索成功率: 100%
# ✓ 平均Top-1分数: 0.5776
# ✓ 平均响应时间: 0.486秒
# ✓ 最高分数: 0.7788
```

### 运行决策树

```
开始
  │
  ├─ 有新的PDF文档？
  │   ├─ 是 → 运行 convert_pdf_to_txt.py
  │   │       ↓
  │   │   运行 batch_merge_lines.py
  │   │       ↓
  │   └─ 运行 chunk_all_documents.py
  │
  ├─ 需要重新索引？
  │   ├─ 是 → 运行 clear_milvus.py --yes
  │   │       ↓
  │   └─ 运行 chunk_all_documents.py
  │
  ├─ 测试检索功能？
  │   └─ 运行 demo_search.py (交互式)
  │       或
  │       运行 test_retrieval.py (自动化)
  │
  └─ 查看系统状态？
      └─ 检查 data/all_chunks.json
          检查 chunking_log.txt
          连接Milvus查看向量数量
```

---

## 项目数据统计

### 处理能力

| 指标 | 数值 | 说明 |
|------|------|------|
| **文档总数** | 180个 | 去重后 |
| **Chunks数量** | 3,422个 | 平均~19个/文档 |
| **向量维度** | 1024维 | BGE-M3模型 |
| **索引类型** | HNSW | 高性能索引 |
| **处理时间** | 8.7分钟 | 180个文档 |
| **检索响应** | 0.5秒 | 平均响应时间 |

### 优化效果

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **文档行数** | 42,094 | 13,224 | -68.6% |
| **Chunks数量** | 2,548 | 3,422 | +34.4% |
| **平均评分** | 0.5680 | 0.5776 | +1.7% |
| **最高评分** | 0.6873 | 0.7788 | +13.3% |

### 测试覆盖

| 模块 | 测试数 | 状态 |
|------|--------|------|
| Configuration Loader | 3 | ✅ 全部通过 |
| Document Type Detector | 7 | ✅ 全部通过 |
| Configurable Splitter | 5 | ✅ 全部通过 |
| Metadata Extractor | 5 | ✅ 全部通过 |
| Chunk Storage | 3 | ✅ 全部通过 |
| Embedding Model | 5 | ✅ 全部通过 |
| Milvus Manager | 4 | ✅ 全部通过 |
| RAG Pipeline | 3 | ✅ 全部通过 |
| **总计** | **35** | **✅ 100%通过** |

---

## 技术栈总览

### 核心技术

| 层级 | 技术选型 | 版本 | 说明 |
|------|---------|------|------|
| **Embedding** | BAAI/bge-m3 | Latest | 1024维, 中文优化, GPU加速 |
| **向量数据库** | Milvus | Latest | HNSW索引, Docker部署 |
| **深度学习框架** | PyTorch | 2.x | CUDA 11.x |
| **RAG框架** | LangChain | Latest | 文档处理、分块 |
| **PDF解析** | PyMuPDF | Latest | 高性能PDF处理 |

### Python依赖

```txt
# requirements.txt
pymilvus>=2.3.0
transformers>=4.30.0
torch>=2.0.0
langchain>=0.1.0
pymupdf>=1.23.0
numpy>=1.24.0
pandas>=2.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0
```

### 开发工具

| 工具 | 用途 |
|------|------|
| **Git** | 版本控制 |
| **Docker** | Milvus容器化部署 |
| **VS Code** | 开发环境 |
| **pytest** | 单元测试 |

---

## 后续计划：AI Agent

### Agent架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Agent 架构 (规划中)                     │
└─────────────────────────────────────────────────────────────┘

用户查询
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent核心 (ReAct模式)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Thought: 我需要查询江西省交通投资集团的业务信息             │
│    │                                                        │
│    ▼                                                        │
│  Action: 调用检索工具                                        │
│    │                                                        │
│    ├─► RAGPipeline.search("江西省交通投资集团 业务")       │
│    │   └─ 返回: Top-K相关文档                               │
│    │                                                        │
│    ▼                                                        │
│  Observation: 检索到5条相关结果，主要涉及...                 │
│    │                                                        │
│    ▼                                                        │
│  Thought: 我需要综合这些信息生成答案                         │
│    │                                                        │
│    ▼                                                        │
│  Action: 调用LLM生成答案                                     │
│    │                                                        │
│    ├─► DeepSeek LLM.generate(context, query)               │
│    │   ├─ 输入: 检索结果 + 用户查询                          │
│    │   └─ 输出: 结构化答案                                   │
│    │                                                        │
│    ▼                                                        │
│  Answer: 江西省交通投资集团有限责任公司的主要业务包括...      │
│         (含引用溯源)                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

组件构成:
  1. 推理引擎 (Reasoning Engine)
     - LangChain Agent
     - ReAct模式

  2. 工具集 (Tools)
     - retrieval_tool: RAG检索
     - query_expansion: 查询扩展
     - metadata_filter: 元数据过滤

  3. 记忆系统 (Memory)
     - 短期记忆: 对话上下文
     - 长期记忆: 检索历史

  4. LLM接口
     - DeepSeek-V3
     - 答案生成
     - 摘要生成
```

### 集成计划

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| **第1阶段** | ReAct Agent框架搭建 | 1周 |
| **第2阶段** | DeepSeek LLM集成 | 1周 |
| **第3阶段** | 工具调用与记忆系统 | 1周 |
| **第4阶段** | 答案溯源与展示 | 1周 |
| **第5阶段** | 集成测试与评估 | 1周 |

---

## 文档版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-03-10 | 初始版本，完整架构文档 |

---

## 联系与支持

- **项目名称**：基于RAG的检索型AI智能体的设计及原型实现
- **文档用途**：中期答辩PPT参考资料
- **生成工具**：Claude Code (Sonnet 4.5)

---

**文档结束**
