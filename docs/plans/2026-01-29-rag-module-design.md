# RAG模块详细设计文档

**项目名称**：基于RAG的检索型AI智能体设计与原型实现
**模块名称**：RAG检索增强生成模块
**设计日期**：2026-01-29
**设计版本**：v1.0

---

## 1. 项目概述

### 1.1 项目背景
为江西省高速公路投资集团开发智能体，自动生成"行业总体性分析报告"。解决人工检索效率低、信息覆盖不全、难以追溯等痛点问题。

### 1.2 核心目标
- 构建高效的RAG检索系统，支持政策解读和行业趋势分析
- 实现完整的Pre-Retrieval和Post-Retrieval优化策略
- 提供可靠的来源追溯机制
- 支持多轮对话和结构化报告生成

### 1.3 应用场景
1. **政策解读与影响分析**：自动检索相关政策法规，分析对行业的影响
2. **行业发展趋势分析**：基于历史数据和报告，分析行业发展态势
3. **企业动态监测**：跟踪企业新闻公告，监控关键动态

---

## 2. 需求分析

### 2.1 数据源类型
- **政策法规文件**：国家和地方的交通政策、法规、条例等
- **行业分析报告**：已有的行业研究报告、市场分析、趋势分析
- **新闻公告文档**：企业新闻、项目公告、通知通告等

### 2.2 功能需求

#### 核心功能
1. **智能文档检索**
   - 语义理解用户查询意图
   - 从大量文档中快速定位相关内容
   - 支持多维度过滤（时间、类型、来源）

2. **检索优化**
   - Pre-Retrieval：文档分块优化、元数据增强、查询预处理
   - Post-Retrieval：结果重排序、去重、相关性过滤

3. **来源追溯**
   - 记录每个文档块的完整来源链路
   - 提供原文定位和引用信息
   - 评估文档可信度

4. **多轮对话支持**
   - 维护对话上下文
   - 支持追问和细化查询

### 2.3 非功能需求
- **性能**：单次检索响应时间 < 3秒
- **准确率**：Top-10结果相关性 > 0.85
- **可扩展性**：支持百万级文档规模
- **可维护性**：模块化设计，便于测试和升级

---

## 3. 技术选型

### 3.1 核心技术栈
| 技术组件 | 选型方案 | 说明 |
|---------|---------|------|
| RAG框架 | **LangChain** | 生态完善，功能丰富，社区活跃 |
| 向量数据库 | **Milvus** | 高性能、可扩展、支持多种索引 |
| 嵌入模型 | **BGE-M3** (本地) | 中文效果好，支持多语言，零成本 |
| 重排序模型 | **BGE-Reranker** (本地) | 精细排序，提升相关性 |
| 文档处理 | LangChain DocumentLoaders | 支持多种格式 |
| 向量存储 | Milvus Python SDK | 官方支持 |

### 3.2 技术选型理由

**选择LangChain**
- 提供完整的RAG工具链
- 丰富的检索器实现
- 易于扩展和定制
- 适合快速原型开发

**选择Milvus**
- 开源免费，性能优秀
- 支持分布式部署
- 原生支持标量过滤
- 提供多种索引类型（IVF、HNSW等）
- 与LangChain集成良好

**选择本地嵌入模型**
- 零API调用成本
- 数据隐私保护
- 可离线部署
- BGE-M3在中文任务上表现优异

---

## 4. 系统架构设计

### 4.1 整体架构

系统采用**三层架构**设计：

```
┌─────────────────────────────────────────────────┐
│              Agent Layer (Agent层)               │
│  - 多轮对话管理  - 报告生成  - 用户交互          │
└──────────────────┬──────────────────────────────┘
                   │ RAG Service Interface
┌──────────────────▼──────────────────────────────┐
│            Service Layer (服务层)                │
│  - RAG引擎  - 来源追溯  - 上下文管理             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           Retrieval Layer (检索层)              │
│  ┌──────────────┬──────────────┬──────────────┐ │
│  │Pre-Retrieval │  Retrieval   │Post-Retrieval│ │
│  │   优化策略   │   核心检索   │   优化策略   │ │
│  └──────────────┴──────────────┴──────────────┘ │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            Data Layer (数据层)                   │
│  ┌──────────────┬──────────────┬──────────────┐ │
│  │  文档加载    │   向量化     │ Milvus存储   │ │
│  │  元数据提取  │   嵌入生成   │  向量索引    │ │
│  └──────────────┴──────────────┴──────────────┘ │
└─────────────────────────────────────────────────┘
```

### 4.2 架构分层说明

#### 4.2.1 数据层 (Data Layer)
**职责**：文档处理、向量化、存储

- **文档加载**：支持PDF、Word、TXT、HTML等格式
- **智能分块**：根据文档类型采用不同分块策略
- **元数据提取**：自动提取文档结构化信息
- **向量化**：使用本地嵌入模型生成向量
- **Milvus存储**：向量 + 元数据的持久化存储

#### 4.2.2 检索层 (Retrieval Layer)
**职责**：检索优化和核心检索逻辑

- **Pre-Retrieval**：查询优化、元数据过滤
- **Retrieval**：语义检索 + 标量过滤
- **Post-Retrieval**：重排序、去重、相关性过滤

#### 4.2.3 服务层 (Service Layer)
**职责**：对Agent提供统一的RAG服务接口

- **RAG引擎**：协调整个检索流程
- **来源追溯**：管理文档块的来源信息
- **上下文管理**：维护多轮对话状态

---

## 5. 核心模块设计

### 5.1 目录结构

```
rag_project/
├── data_loader/                 # 数据加载模块
│   ├── __init__.py
│   ├── document_loader.py       # 多格式文档加载
│   ├── text_splitter.py         # 智能分块器
│   └── metadata_extractor.py    # 元数据提取器
│
├── embeddings/                   # 向量化模块
│   ├── __init__.py
│   ├── embedding_model.py       # 本地嵌入模型封装
│   └── milvus_client.py         # Milvus客户端封装
│
├── retrieval/                    # 检索模块
│   ├── __init__.py
│   ├── pre_retrieval.py         # Pre-Retrieval策略
│   ├── retriever.py             # 核心检索器
│   └── post_retrieval.py        # Post-Retrieval优化
│
├── rag_service/                  # RAG服务层
│   ├── __init__.py
│   ├── rag_engine.py            # RAG主引擎
│   ├── source_tracker.py        # 来源追溯
│   └── context_manager.py       # 多轮对话上下文
│
├── models/                       # 数据模型
│   ├── __init__.py
│   ├── document.py              # 文档数据模型
│   ├── retrieval_result.py      # 检索结果模型
│   └── source_info.py           # 来源信息模型
│
├── config/                       # 配置文件
│   ├── milvus_config.yaml       # Milvus配置
│   ├── model_config.yaml        # 模型配置
│   └── retrieval_config.yaml    # 检索参数配置
│
├── utils/                        # 工具函数
│   ├── __init__.py
│   ├── logger.py                # 日志工具
│   └── helpers.py               # 辅助函数
│
├── tests/                        # 测试代码
│   ├── test_pre_retrieval.py
│   ├── test_retriever.py
│   └── test_post_retrieval.py
│
├── main.py                       # 主入口
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

### 5.2 核心类设计

#### 5.2.1 RAGEngine（RAG主引擎）

```python
class RAGEngine:
    """RAG检索主引擎，协调各模块完成检索任务"""

    def __init__(self, config: Dict):
        self.pre_retrieval = PreRetrievalStrategy(config)
        self.retriever = MilvusRetriever(config)
        self.post_retrieval = PostRetrievalOptimizer(config)
        self.source_tracker = SourceTracker()

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> RetrievalResult:
        """执行标准检索"""
        # 1. Pre-Retrieval优化
        optimized_query = self.pre_retrieval.optimize_query(query)

        # 2. 执行检索
        raw_results = self.retriever.retrieve(
            optimized_query,
            filters=filters,
            top_k=top_k * 5  # 获取更多候选用于重排序
        )

        # 3. Post-Retrieval优化
        final_chunks = self.post_retrieval.optimize(
            raw_results,
            query,
            top_k=top_k
        )

        # 4. 来源追溯
        self.source_tracker.track_sources(final_chunks)

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            sources=self.source_tracker.get_source_map()
        )

    def retrieve_for_report(
        self,
        query: str,
        report_type: str,
        time_range: Optional[Tuple] = None
    ) -> RetrievalResult:
        """为报告生成专用检索"""
        filters = self._build_report_filters(report_type, time_range)
        return self.retrieve(query, filters)
```

#### 5.2.2 PreRetrievalStrategy（检索前策略）

```python
class PreRetrievalStrategy:
    """Pre-Retrieval优化策略"""

    def __init__(self, config: Dict):
        self.query_expander = QueryExpander()
        self.keyword_extractor = KeywordExtractor()

    def optimize_query(self, query: str) -> str:
        """查询优化"""
        # 1. 提取关键词
        keywords = self.keyword_extractor.extract(query)

        # 2. 查询扩展（基于领域词典）
        expanded_query = self.query_expander.expand(query)

        # 3. 构建优化后的查询
        optimized = self._combine_queries(query, expanded_query, keywords)

        return optimized

    def apply_filters(
        self,
        filters: Dict
    ) -> Dict:
        """构建Milvus标量过滤表达式"""
        # 支持时间范围、文档类型等过滤
        pass
```

#### 5.2.3 PostRetrievalOptimizer（检索后优化）

```python
class PostRetrievalOptimizer:
    """Post-Retrieval优化策略"""

    def __init__(self, config: Dict):
        self.reranker = BGEReranker(config['reranker_model'])
        self.deduplicator = Deduplicator()
        self.threshold = config.get('similarity_threshold', 0.7)

    def optimize(
        self,
        chunks: List[DocumentChunk],
        query: str,
        top_k: int
    ) -> List[DocumentChunk]:
        """执行完整的Post-Retrieval优化"""

        # 1. 相关性过滤
        filtered = self._filter_by_relevance(chunks, self.threshold)

        # 2. 重排序
        reranked = self.reranker.rerank(query, filtered)

        # 3. 去重
        deduped = self.deduplicator.deduplicate(reranked)

        # 4. 多样性控制
        diverse = self._ensure_diversity(deduped, top_k)

        return diverse[:top_k]

    def _filter_by_relevance(
        self,
        chunks: List[DocumentChunk],
        threshold: float
    ) -> List[DocumentChunk]:
        """基于相似度阈值过滤"""
        return [c for c in chunks if c.similarity_score >= threshold]
```

#### 5.2.4 SourceTracker（来源追溯）

```python
class SourceTracker:
    """文档来源追溯管理"""

    def __init__(self):
        self.source_map: Dict[str, SourceInfo] = {}
        self.citation_index: List[Citation] = []

    def track_sources(
        self,
        chunks: List[DocumentChunk]
    ) -> None:
        """为文档块建立来源追踪"""
        for chunk in chunks:
            source_info = SourceInfo(
                chunk_id=chunk.id,
                source_doc=chunk.metadata['source_doc'],
                page_number=chunk.metadata.get('page_number'),
                section_title=chunk.metadata.get('section_title'),
                publish_date=chunk.metadata.get('publish_date'),
                doc_type=chunk.metadata.get('doc_type'),
                confidence_score=chunk.similarity_score
            )
            self.source_map[chunk.id] = source_info

    def get_source_map(self) -> Dict[str, SourceInfo]:
        """获取完整的来源映射"""
        return self.source_map

    def get_citation(self, chunk_id: str) -> Citation:
        """获取引用信息（用于报告）"""
        source = self.source_map.get(chunk_id)
        if not source:
            return None
        return Citation(
            doc_name=source.source_doc,
            page=source.page_number,
            section=source.section_title,
            date=source.publish_date,
            snippet=source.text_snippet
        )
```

---

## 6. 详细设计

### 6.1 Pre-Retrieval优化策略

#### 6.1.1 智能文档分块

针对不同文档类型采用差异化分块策略：

| 文档类型 | 分块策略 | 块大小 | 重叠 | 说明 |
|---------|---------|--------|------|------|
| 政策法规 | 按条款/章节 | 200 token | 50 token | 保持条款完整性 |
| 行业报告 | 按章节 | 500 token | 100 token | 保留上下文 |
| 新闻公告 | 按段落 | 150-300 token | 30 token | 保持语义完整 |

实现要点：
```python
class AdaptiveTextSplitter:
    """自适应文本分块器"""

    def split_document(self, doc: Document) -> List[DocumentChunk]:
        doc_type = doc.metadata.get('doc_type')

        if doc_type == 'policy':
            return self._split_by_clause(doc)
        elif doc_type == 'report':
            return self._split_by_chapter(doc)
        elif doc_type == 'news':
            return self._split_by_paragraph(doc)
        else:
            return self._split_by_token_size(doc, chunk_size=400)
```

#### 6.1.2 元数据增强

为每个文档块添加结构化元数据：

```python
metadata_schema = {
    "chunk_id": "str (UUID)",
    "doc_type": "str (policy/report/news)",
    "source_doc": "str (文档名称)",
    "publish_date": "datetime",
    "keywords": "List[str]",
    "importance_score": "float (0-1)",
    "page_number": "int",
    "section_title": "str",
    "author": "str (可选)",
    "category": "str (行业分类)"
}
```

重要性评分计算：
- 政策文件：基础分0.8 + 发布时效性
- 行业报告：基础分0.7 + 数据完整度
- 新闻公告：基础分0.6 + 热度指标

#### 6.1.3 查询预处理

```python
class QueryPreprocessor:
    """查询预处理器"""

    def process(self, query: str) -> ProcessedQuery:
        # 1. 关键词提取
        keywords = self._extract_keywords(query)

        # 2. 查询扩展（同义词、相关词）
        expanded = self._expand_query(query, keywords)

        # 3. 意图识别
        intent = self._classify_intent(query)  # 政策/趋势/数据

        # 4. 时间范围提取
        time_range = self._extract_time_range(query)

        return ProcessedQuery(
            original=query,
            optimized=expanded,
            keywords=keywords,
            intent=intent,
            time_range=time_range
        )
```

### 6.2 Post-Retrieval优化策略

#### 6.2.1 混合重排序流程

```
初始检索 (Top-50)
    ↓
Cross-Encoder重排序 (基于BGE-Reranker)
    ↓
业务逻辑排序
    - 政策文件优先
    - 最新发布优先
    - 高重要性优先
    ↓
返回Top-10
```

#### 6.2.2 去重策略

**内容去重**：
```python
def deduplicate_content(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """基于文本相似度去重"""
    unique_chunks = []
    seen_hashes = set()

    for chunk in chunks:
        # 使用MinHash计算文本指纹
        fingerprint = compute_minhash(chunk.text)

        # 检查是否重复
        if not is_similar_to_existing(fingerprint, seen_hashes):
            unique_chunks.append(chunk)
            seen_hashes.add(fingerprint)

    return unique_chunks
```

**来源多样性**：
```python
def ensure_source_diversity(
    chunks: List[DocumentChunk],
    max_per_doc: int = 3
) -> List[DocumentChunk]:
    """确保结果不集中在单一文档"""
    source_count = defaultdict(int)
    diverse_chunks = []

    for chunk in chunks:
        source = chunk.metadata['source_doc']
        if source_count[source] < max_per_doc:
            diverse_chunks.append(chunk)
            source_count[source] += 1

    return diverse_chunks
```

#### 6.2.3 相关性过滤

```python
class RelevanceFilter:
    """相关性过滤器"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def filter(
        self,
        chunks: List[DocumentChunk],
        query: str
    ) -> List[DocumentChunk]:
        """多维度相关性过滤"""

        filtered = []
        for chunk in chunks:
            # 1. 向量相似度过滤
            if chunk.similarity_score < self.threshold:
                continue

            # 2. 关键词匹配度
            keyword_score = self._compute_keyword_match(chunk, query)
            if keyword_score < 0.3:
                continue

            # 3. 意图相关性（根据查询意图调整权重）
            intent_score = self._compute_intent_relevance(chunk, query)

            # 综合评分
            final_score = (
                chunk.similarity_score * 0.6 +
                keyword_score * 0.3 +
                intent_score * 0.1
            )

            if final_score >= self.threshold:
                filtered.append(chunk)

        return filtered
```

### 6.3 来源追溯机制

#### 6.3.1 追溯数据结构

```python
@dataclass
class SourceInfo:
    """文档块来源信息"""
    chunk_id: str
    source_doc: str
    page_number: Optional[int]
    section_title: Optional[str]
    paragraph_index: Optional[int]
    publish_date: datetime
    doc_type: str
    confidence_score: float
    text_snippet: str  # 原文片段（用于展示）

@dataclass
class Citation:
    """引用信息（用于报告）"""
    doc_name: str
    page: Optional[int]
    section: Optional[str]
    date: datetime
    snippet: str
    url: Optional[str]  # 如果是线上文档
```

#### 6.3.2 可信度评分

```python
def compute_credibility_score(source: SourceInfo) -> float:
    """计算文档块可信度评分"""

    # 1. 来源权威性 (0-0.4)
    authority_scores = {
        'policy': 0.4,      # 政策文件
        'report': 0.3,      # 行业报告
        'news': 0.2         # 新闻公告
    }
    authority = authority_scores.get(source.doc_type, 0.1)

    # 2. 时效性 (0-0.3)
    days_old = (datetime.now() - source.publish_date).days
    recency = max(0, 0.3 - (days_old / 3650))  # 10年衰减

    # 3. 相关性 (0-0.3)
    relevance = source.confidence_score * 0.3

    return authority + recency + relevance
```

#### 6.3.3 引用链追踪

```python
class CitationChain:
    """引用链管理"""

    def __init__(self):
        self.chain: Dict[str, CitationNode] = {}

    def build_chain(
        self,
        chunk: DocumentChunk,
        generated_text: str
    ) -> CitationPath:
        """构建从生成结果到原文的引用路径"""

        path = CitationPath(
            generated_text=generated_text,
            cited_chunks=[chunk],
            source_doc=chunk.metadata['source_doc'],
            retrieval_confidence=chunk.similarity_score
        )

        # 记录引用关系
        self.chain[generated_text[:50]] = CitationNode(
            chunk_id=chunk.id,
            source_doc=chunk.metadata['source_doc'],
            location=f"Page {chunk.metadata.get('page_number')}",
            confidence=chunk.similarity_score
        )

        return path
```

---

## 7. 数据流设计

### 7.1 文档索引流程

```
原始文档 (PDF/Word/TXT)
    ↓
DocumentLoader (加载文档)
    ↓
MetadataExtractor (提取元数据)
    ↓
AdaptiveTextSplitter (智能分块)
    ↓
EmbeddingModel (生成向量)
    ↓
MilvusClient (存储向量+元数据)
    ↓
索引完成
```

### 7.2 检索流程

```
用户查询
    ↓
QueryPreprocessor (查询预处理)
    - 提取关键词
    - 查询扩展
    - 意图识别
    ↓
PreRetrievalStrategy (应用策略)
    - 构建过滤条件
    - 优化查询表达式
    ↓
MilvusRetriever (向量检索)
    - 语义检索
    - 标量过滤
    - 返回Top-50
    ↓
PostRetrievalOptimizer (优化结果)
    - 相关性过滤
    - 重排序 (Reranker)
    - 去重
    - 多样性控制
    - 返回Top-10
    ↓
SourceTracker (来源追溯)
    - 建立来源映射
    - 计算可信度
    ↓
返回结构化结果
```

### 7.3 多轮对话检索流程

```
用户查询 (第N轮)
    ↓
ContextManager (加载历史上下文)
    - 前N-1轮查询
    - 前N-1轮检索结果
    ↓
ContextAwareQueryBuilder (构建上下文感知查询)
    - 合并当前查询和历史
    - 识别引用关系
    ↓
执行标准检索流程
    ↓
ContextManager (更新上下文)
    - 保存当前查询和结果
    ↓
返回结果
```

---

## 8. Agent集成接口

### 8.1 RAG服务接口

```python
class RAGServiceInterface:
    """RAG服务接口，供Agent调用"""

    def __init__(self, config_path: str):
        self.rag_engine = RAGEngine(config_path)
        self.context_manager = ContextManager()

    def retrieve_for_report(
        self,
        query: str,
        report_type: str = "policy_analysis",
        top_k: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> RetrievalResult:
        """
        为报告生成检索相关内容

        Args:
            query: 用户查询
            report_type: 报告类型 (policy_analysis/trend_analysis)
            top_k: 返回结果数量
            time_range: 时间范围过滤

        Returns:
            RetrievalResult: 包含文档块和来源信息的结果对象
        """
        filters = self._build_report_filters(report_type, time_range)
        return self.rag_engine.retrieve(query, filters, top_k)

    def retrieve_with_context(
        self,
        query: str,
        session_id: str
    ) -> RetrievalResult:
        """
        多轮对话检索（考虑历史上下文）

        Args:
            query: 当前查询
            session_id: 会话ID

        Returns:
            RetrievalResult: 检索结果
        """
        # 获取历史上下文
        context = self.context_manager.get_context(session_id)

        # 构建上下文感知查询
        context_query = self._build_context_query(query, context)

        # 执行检索
        result = self.rag_engine.retrieve(context_query)

        # 更新上下文
        self.context_manager.update_context(session_id, query, result)

        return result

    def get_source_details(
        self,
        chunk_ids: List[str]
    ) -> List[SourceDetail]:
        """
        获取文档块的详细来源信息（用于报告引用）

        Args:
            chunk_ids: 文档块ID列表

        Returns:
            List[SourceDetail]: 详细来源信息列表
        """
        return self.rag_engine.source_tracker.get_details(chunk_ids)

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> Dict[str, RetrievalResult]:
        """
        批量检索（用于报告多个章节的检索）

        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数

        Returns:
            Dict[str, RetrievalResult]: 查询到结果的映射
        """
        results = {}
        for query in queries:
            results[query] = self.rag_engine.retrieve(query, top_k=top_k)
        return results
```

### 8.2 数据模型

```python
@dataclass
class RetrievalResult:
    """检索结果"""
    query: str                              # 原始查询
    chunks: List[DocumentChunk]             # 检索到的文档块
    sources: Dict[str, SourceInfo]          # 来源信息映射
    summary: ResultSummary                  # 结果摘要

@dataclass
class DocumentChunk:
    """文档块"""
    id: str                                 # 唯一标识
    text: str                               # 文本内容
    embedding: np.ndarray                   # 向量表示
    metadata: Dict[str, Any]                # 元数据
    similarity_score: float                 # 相似度分数

@dataclass
class ResultSummary:
    """结果摘要"""
    total_chunks: int                       # 总块数
    doc_types_distribution: Dict[str, int]  # 文档类型分布
    time_range: Tuple[datetime, datetime]   # 时间跨度
    avg_confidence: float                   # 平均置信度
    top_sources: List[str]                  # 主要来源
```

---

## 9. 配置文件

### 9.1 Milvus配置

```yaml
# config/milvus_config.yaml

milvus:
  # 连接配置
  host: "localhost"
  port: 19530
  alias: "default"

  # Collection配置
  collection:
    name: "enterprise_docs"
    description: "企业文档向量库"
    dimension: 1024  # BGE-M3维度

  # 索引配置
  index:
    type: "HNSW"  # 层次化小世界图索引
    metric_type: "IP"  # 内积相似度
    params:
      M: 16  # HNSW图的连接数
      efConstruction: 256  # 构建时的搜索深度

  # 搜索配置
  search:
    top_k: 50  # 初检返回数量
    ef: 128  # 搜索时的深度
    round_decimal: 4  # 结果精度

  # 字段配置
  fields:
    - name: "id"
      type: "VARCHAR"
      max_length: 100
      primary_key: true

    - name: "vector"
      type: "FLOAT_VECTOR"
      dimension: 1024

    - name: "text"
      type: "VARCHAR"
      max_length: 65535

    - name: "doc_type"
      type: "VARCHAR"
      max_length: 50

    - name: "publish_date"
      type: "INT64"  # Unix时间戳

    - name: "source_doc"
      type: "VARCHAR"
      max_length: 512

    - name: "importance_score"
      type: "FLOAT"
```

### 9.2 模型配置

```yaml
# config/model_config.yaml

embedding:
  # 嵌入模型配置
  model_name: "BAAI/bge-m3"
  device: "cuda"  # cuda/cpu
  batch_size: 32
  max_length: 8192
  normalize_embeddings: true

  # 本地缓存路径
  cache_dir: "./models/embeddings"

reranker:
  # 重排序模型配置
  model_name: "BAAI/bge-reranker-v2-m3"
  device: "cuda"
  batch_size: 16
  max_length: 512
  top_k: 10

  # 缓存路径
  cache_dir: "./models/reranker"

text_splitter:
  # 分块配置
  chunk_size:
    policy: 200
    report: 500
    news: 200
    default: 400

  chunk_overlap:
    policy: 50
    report: 100
    news: 30
    default: 50

  separators: ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
```

### 9.3 检索配置

```yaml
# config/retrieval_config.yaml

pre_retrieval:
  # Pre-Retrieval配置
  query_expansion:
    enabled: true
    method: "synonym"  # synonym/llm
    top_k: 5

  keyword_extraction:
    enabled: true
    method: "jieba"
    top_k: 10

  metadata_filtering:
    enabled: true
    default_filters:
      - field: "publish_date"
        operator: ">="
        value: "2020-01-01"

retrieval:
  # 核心检索配置
  top_k_pre_rerank: 50
  top_k_final: 10
  enable_hybrid_search: false  # 是否启用混合检索（向量+关键词）

post_retrieval:
  # Post-Retrieval配置
  reranking:
    enabled: true
    model_weight: 0.7  # 模型评分权重
    metadata_weight: 0.3  # 元数据权重

  deduplication:
    enabled: true
    method: "minhash"  # minhash/simhash
    threshold: 0.85

  diversity:
    enabled: true
    max_chunks_per_doc: 3  # 每个文档最多返回块数
    time_balance: true  # 时间平衡

  relevance_filter:
    enabled: true
    similarity_threshold: 0.7
    keyword_match_threshold: 0.3

source_tracking:
  # 来源追溯配置
  enable_credibility_scoring: true
  enable_citation_chain: true
  text_snippet_length: 200  # 引用片段长度
```

---

## 10. 实现计划

### 10.1 开发阶段

**Phase 1: 基础设施搭建** (预计3-4天)
- [ ] 项目结构初始化
- [ ] 依赖包安装和配置
- [ ] Milvus数据库部署和连接测试
- [ ] 本地嵌入模型下载和测试
- [ ] 日志和监控框架搭建

**Phase 2: 数据层开发** (预计4-5天)
- [ ] DocumentLoader实现（支持PDF/Word/TXT）
- [ ] AdaptiveTextSplitter实现
- [ ] MetadataExtractor实现
- [ ] EmbeddingModel封装
- [ ] MilvusClient封装
- [ ] 文档索引Pipeline测试

**Phase 3: 检索层开发** (预计5-6天)
- [ ] PreRetrievalStrategy实现
  - [ ] QueryPreprocessor
  - [ ] KeywordExtractor
  - [ ] QueryExpander
- [ ] MilvusRetriever实现
- [ ] PostRetrievalOptimizer实现
  - [ ] BGEReranker集成
  - [ ] Deduplicator
  - [ ] RelevanceFilter
- [ ] 检索流程端到端测试

**Phase 4: 服务层开发** (预计3-4天)
- [ ] RAGEngine主引擎实现
- [ ] SourceTracker实现
- [ ] ContextManager实现
- [ ] RAGServiceInterface接口实现
- [ ] 单元测试和集成测试

**Phase 5: 测试与优化** (预计3-4天)
- [ ] 性能测试（检索速度、准确率）
- [ ] 边界情况测试
- [ ] 参数调优
- [ ] 错误处理完善
- [ ] 文档编写

**Phase 6: 演示与文档** (预计2-3天)
- [ ] 准备演示数据集
- [ ] 编写使用示例
- [ ] 性能基准测试报告
- [ ] API文档生成
- [ ] 毕设答辩准备

### 10.2 关键里程碑

| 里程碑 | 交付物 | 预计时间 |
|-------|--------|---------|
| M1: 基础设施完成 | Milvus运行、模型可用、框架搭建 | Day 4 |
| M2: 数据索引完成 | 能成功索引和检索文档 | Day 9 |
| M3: 检索优化完成 | Pre/Post-Retrieval策略工作正常 | Day 15 |
| M4: 服务集成完成 | 完整RAG服务可用 | Day 19 |
| M5: 测试通过 | 单元测试、集成测试通过 | Day 23 |
| M6: 答辩准备 | 演示系统、文档、报告 | Day 26 |

### 10.3 验收标准

**功能验收**
- [ ] 能成功加载PDF/Word文档并提取元数据
- [ ] 能根据文档类型智能分块
- [ ] 能使用本地模型生成向量并存储到Milvus
- [ ] 能执行语义检索并返回相关结果
- [ ] Pre-Retrieval策略有效（查询优化、过滤）
- [ ] Post-Retrieval策略有效（重排序、去重）
- [ ] 能完整追溯每个文档块的来源
- [ ] 支持多轮对话的上下文检索
- [ ] 接口清晰，易于Agent集成

**性能验收**
- [ ] 单次检索响应时间 < 3秒
- [ ] Top-10结果平均相关性 > 0.85
- [ ] 支持至少10万文档块的索引
- [ ] 来源追溯准确率 100%

**质量验收**
- [ ] 单元测试覆盖率 > 80%
- [ ] 代码符合PEP8规范
- [ ] 关键模块有完整文档字符串
- [ ] 有清晰的使用示例和API文档

---

## 11. 技术风险与应对

### 11.1 潜在风险

| 风险项 | 影响 | 概率 | 应对措施 |
|-------|------|------|---------|
| 本地模型性能不足 | 高 | 中 | 预留切换到商业API的方案 |
| Milvus部署困难 | 中 | 低 | 使用Docker快速部署 |
| 文档格式兼容性问题 | 中 | 中 | 提前测试各种格式，准备备用方案 |
| 检索准确率不达标 | 高 | 中 | 预留充足的调优时间 |
| 时间进度紧张 | 高 | 中 | 采用MVP方式，优先核心功能 |

### 11.2 优化空间

**短期优化**（毕业设计阶段）
- 尝试不同的分块策略
- 调优重排序模型参数
- 优化元数据过滤规则

**长期优化**（生产环境）
- 引入查询改写（Query Rewriting）
- 实现HyDE（Hypothetical Document Embeddings）
- 添加知识图谱增强
- 实现增量索引更新
- 分布式部署支持

---

## 12. 总结

本设计文档详细描述了基于RAG的检索模块的设计方案，核心特点包括：

1. **完整的检索优化**：包含Pre-Retrieval和Post-Retrieval深度优化
2. **可靠的来源追溯**：为Agent提供准确的文档来源信息
3. **灵活的架构设计**：模块化、可扩展、易维护
4. **实用的技术选型**：LangChain + Milvus + 本地模型，平衡效果与成本
5. **清晰的实现路径**：分阶段开发，里程碑明确

该设计能够有效支撑毕业设计的技术深度要求，同时具备实际应用价值。

---

**文档版本历史**

| 版本 | 日期 | 修改人 | 说明 |
|-----|------|--------|------|
| v1.0 | 2026-01-29 | Claude | 初始版本 |

