"""
Milvus Hybrid Manager — Dense Vector + BM25 Full-Text Search.

Uses Milvus 2.6+ built-in BM25 Function to auto-generate sparse vectors
from raw text. Combines dense (semantic) and sparse (keyword) search via
hybrid_search() with RRFRanker.
"""

from typing import List, Dict, Optional
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
    WeightedRanker,
    utility,
)
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger


class MilvusHybridManager:
    """Manage hybrid (dense + BM25) search in Milvus."""

    def __init__(
        self,
        config_path: str = "config/milvus_config.yaml",
        collection_name: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.milvus_config = self.config.get("milvus", {})

        self._connect()

        self.collection_name = (
            collection_name
            or self.milvus_config["collection"]["name"] + "_hybrid"
        )
        self.collection = self._get_or_create_collection()

    # ── Connection ─────────────────────────────────────────────────────

    def _connect(self):
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        alias = self.milvus_config.get("alias", "default")
        connections.connect(alias=alias, host=host, port=port)
        logger.info(f"Hybrid manager connected to Milvus at {host}:{port}")

    # ── Schema ─────────────────────────────────────────────────────────

    def _create_hybrid_schema(self) -> CollectionSchema:
        dimension = self.milvus_config["collection"]["dimension"]

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                auto_id=False,
            ),
            # Dense vector field (BGE-M3 embeddings)
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
            ),
            # Text field — enable BM25 analyzer for full-text search
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": "standard"},
            ),
            # Sparse vector field — auto-populated by BM25 Function
            FieldSchema(
                name="sparse",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            # Metadata fields
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="publish_date", dtype=DataType.INT64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Hybrid dense + BM25 collection",
            enable_dynamic_field=False,
        )

        # BM25 Function: text → sparse
        bm25_fn = Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_fn)

        logger.info("Created hybrid schema with BM25 function")
        return schema

    # ── Collection & Index ─────────────────────────────────────────────

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(self.collection_name):
            logger.info(f"Using existing hybrid collection: {self.collection_name}")
            return Collection(self.collection_name)

        logger.info(f"Creating new hybrid collection: {self.collection_name}")
        schema = self._create_hybrid_schema()
        collection = Collection(name=self.collection_name, schema=schema)
        self._create_indexes(collection)
        return collection

    def _create_indexes(self, collection: Collection):
        # Dense vector index (HNSW, same as original)
        index_config = self.milvus_config.get("index", {})
        dense_index = {
            "index_type": index_config.get("type", "HNSW"),
            "metric_type": index_config.get("metric_type", "IP"),
            "params": index_config.get("params", {}),
        }
        collection.create_index(field_name="vector", index_params=dense_index)
        logger.info(f"Created dense index: {dense_index['index_type']}")

        # Sparse vector index (BM25)
        sparse_index = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        }
        collection.create_index(field_name="sparse", index_params=sparse_index)
        logger.info("Created sparse BM25 index")

    # ── Insert ──────────────────────────────────────────────────────────

    def insert_data(self, data: List[Dict]) -> int:
        """Insert chunks. Sparse vectors auto-generated by BM25 Function."""
        insert_batch = {
            "id": [item["id"] for item in data],
            "vector": [item["vector"] for item in data],
            "text": [item["text"] for item in data],
            # sparse field is auto-generated, do NOT include it
            "doc_type": [item.get("doc_type", "unknown") for item in data],
            "source": [item.get("source", "") for item in data],
            "publish_date": [item.get("publish_date") or 0 for item in data],
            "page_number": [item.get("page_number", 0) for item in data],
            "title": [item.get("title", "") for item in data],
        }

        self.collection.insert([
            insert_batch[field.name]
            for field in self.collection.schema.fields
            if field.name in insert_batch
        ])
        self.collection.flush()
        logger.info(f"Inserted {len(data)} records into {self.collection_name}")
        return len(data)

    # ── Hybrid Search ───────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        ranker: str = "rrf",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[Dict]:
        """Hybrid search combining dense vector + BM25 text search.

        Args:
            query_vector: Dense embedding of the query.
            query_text: Raw query text for BM25.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            ranker: "rrf" (Reciprocal Rank Fusion) or "weighted".
            dense_weight: Weight for dense search (only for weighted ranker).
            sparse_weight: Weight for BM25 search (only for weighted ranker).

        Returns:
            List of search results with text, score, and metadata.
        """
        search_config = self.milvus_config.get("search", {})
        output_fields = [
            "text", "doc_type", "source", "publish_date",
            "page_number", "title",
        ]
        expr = self._build_filter_expression(filters) if filters else None

        # Dense search request
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={
                "metric_type": self.milvus_config["index"]["metric_type"],
                "params": {"ef": search_config.get("ef", 128)},
            },
            limit=top_k,
            expr=expr,
        )

        # Sparse (BM25) search request — pass raw text
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
            expr=expr,
        )

        # Ranker
        if ranker == "weighted":
            ranker_obj = WeightedRanker(dense_weight, sparse_weight)
        else:
            ranker_obj = RRFRanker()

        results = self.collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker_obj,
            limit=top_k,
            output_fields=output_fields,
        )

        formatted = []
        for hit in results[0]:
            formatted.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "metadata": {
                    "doc_type": hit.entity.get("doc_type"),
                    "source": hit.entity.get("source"),
                    "publish_date": hit.entity.get("publish_date"),
                    "page_number": hit.entity.get("page_number"),
                    "title": hit.entity.get("title"),
                },
            })
        return formatted

    def search_dense_only(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Dense-only search (same as original MilvusManager)."""
        search_config = self.milvus_config.get("search", {})
        search_params = {
            "metric_type": self.milvus_config["index"]["metric_type"],
            "params": {"ef": search_config.get("ef", 128)},
        }
        expr = self._build_filter_expression(filters) if filters else None

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "text", "doc_type", "source",
                "publish_date", "page_number", "title",
            ],
        )

        formatted = []
        for hit in results[0]:
            formatted.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "metadata": {
                    "doc_type": hit.entity.get("doc_type"),
                    "source": hit.entity.get("source"),
                    "publish_date": hit.entity.get("publish_date"),
                    "page_number": hit.entity.get("page_number"),
                    "title": hit.entity.get("title"),
                },
            })
        return formatted

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_filter_expression(self, filters: Dict) -> Optional[str]:
        parts = []
        if "doc_type" in filters:
            types = ", ".join([f'"{dt}"' for dt in filters["doc_type"]])
            parts.append(f"doc_type in [{types}]")
        if "start_date" in filters:
            parts.append(f"publish_date >= {filters['start_date']}")
        if "end_date" in filters:
            parts.append(f"publish_date <= {filters['end_date']}")
        return " and ".join(parts) if parts else None

    def get_collection_stats(self) -> Dict:
        self.collection.load()
        return {
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
        }

    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped hybrid collection: {self.collection_name}")
