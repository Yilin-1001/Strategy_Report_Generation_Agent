import uuid
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from pymilvus import (  # type: ignore
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger

class MilvusManager:
    """Manage Milvus vector database operations"""

    def __init__(
        self,
        config_path: str = "config/milvus_config.yaml",
        collection_name: Optional[str] = None
    ):
        """
        Initialize Milvus manager

        Args:
            config_path: Path to Milvus configuration
            collection_name: Collection name (overrides config)
        """
        self.config = load_config(config_path)
        self.milvus_config = self.config.get('milvus', {})

        # Connect to Milvus
        self._connect()

        # Get or create collection
        collection_name = collection_name or self.milvus_config['collection']['name']
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _connect(self):
        """Connect to Milvus server"""
        host = self.milvus_config.get('host', 'localhost')
        port = self.milvus_config.get('port', 19530)
        alias = self.milvus_config.get('alias', 'default')

        try:
            connections.connect(alias=alias, host=host, port=port)
            logger.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _create_collection_schema(self) -> CollectionSchema:
        """Create collection schema"""
        dimension = self.milvus_config['collection']['dimension']

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="doc_type",
                dtype=DataType.VARCHAR,
                max_length=20,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="publish_date",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="page_number",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=self.milvus_config['collection'].get('description', ''),
            enable_dynamic_field=False,
        )

        return schema

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one"""
        if utility.has_collection(self.collection_name):
            logger.info(f"Using existing collection: {self.collection_name}")
            collection = Collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: {self.collection_name}")

            schema = self._create_collection_schema()
            collection = Collection(
                name=self.collection_name,
                schema=schema,
            )

            # Create index
            self._create_index(collection)

        return collection

    def _create_index(self, collection: Collection):
        """Create index for vector field"""
        index_config = self.milvus_config.get('index', {})

        index_params = {
            "index_type": index_config.get('type', 'HNSW'),
            "metric_type": index_config.get('metric_type', 'IP'),
            "params": index_config.get('params', {}),
        }

        collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info(f"Created index: {index_params['index_type']}")

    def insert_data(self, data: List[Dict]) -> int:
        """
        Insert data into collection

        Args:
            data: List of dictionaries with keys: id, vector, text, metadata fields

        Returns:
            Number of inserted records
        """
        # Prepare data by field
        insert_data = {
            "id": [item["id"] for item in data],
            "vector": [item["vector"] for item in data],
            "text": [item["text"] for item in data],
            "doc_type": [item.get("doc_type", "unknown") for item in data],
            "source": [item.get("source", "") for item in data],
            "publish_date": [self._to_timestamp(item.get("publish_date")) for item in data],
            "page_number": [item.get("page_number", 0) for item in data],
            "title": [item.get("title", "") for item in data],
        }

        # Insert
        insert_result = self.collection.insert([insert_data[field.name] for field in self.collection.schema.fields])

        # Flush
        self.collection.flush()

        logger.info(f"Inserted {len(data)} records into {self.collection_name}")

        return len(data)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search similar vectors

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional filters (e.g., {"doc_type": ["news"]})

        Returns:
            List of search results with text, score, and metadata
        """
        # Note: Collection must be loaded before search
        self.collection.load()

        # Search parameters
        search_config = self.milvus_config.get('search', {})
        search_params = {
            "metric_type": self.milvus_config['index']['metric_type'],
            "params": {"ef": search_config.get('ef', 128)},
        }

        # Build filter expression
        expr = self._build_filter_expression(filters) if filters else None

        # Search
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "doc_type", "source", "publish_date", "page_number", "title"],
        )

        # Format results
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "text": hit.entity.get("text"),
                "score": hit.score,
                "metadata": {
                    "doc_type": hit.entity.get("doc_type"),
                    "source": hit.entity.get("source"),
                    "publish_date": hit.entity.get("publish_date"),
                    "page_number": hit.entity.get("page_number"),
                    "title": hit.entity.get("title"),
                }
            })

        return formatted_results

    def _build_filter_expression(self, filters: Dict) -> Optional[str]:
        """Build Milvus filter expression from filters dictionary"""
        expr_parts = []

        # Document type filter
        if "doc_type" in filters:
            doc_types = ", ".join([f'"{dt}"' for dt in filters["doc_type"]])
            expr_parts.append(f"doc_type in [{doc_types}]")

        # Date range filter
        if "start_date" in filters:
            start_ts = self._to_timestamp(filters["start_date"])
            expr_parts.append(f"publish_date >= {start_ts}")

        if "end_date" in filters:
            end_ts = self._to_timestamp(filters["end_date"])
            expr_parts.append(f"publish_date <= {end_ts}")

        return " and ".join(expr_parts) if expr_parts else None

    def _to_timestamp(self, date_str) -> int:
        """Convert date string to Unix timestamp"""
        if not date_str:
            return 0

        if isinstance(date_str, int):
            return date_str

        try:
            dt = datetime.fromisoformat(str(date_str).replace("T", " "))
            return int(dt.timestamp())
        except:
            return 0

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        self.collection.load()

        num_entities = self.collection.num_entities

        return {
            "name": self.collection_name,
            "num_entities": num_entities,
        }

    def drop_collection(self):
        """Drop the collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped collection: {self.collection_name}")
