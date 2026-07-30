from typing import List, Optional, TypedDict
from dataclasses import dataclass

class SparseVector(TypedDict):
    indices: List[int]
    values: List[float]


@dataclass
class PineconeConfig:
    """
    Configuration for Pinecone connection.

    .. deprecated::
        Replaced by :class:`QdrantConfig` hybrid collections.
        Retained for reference only — no longer used at runtime.

    Dual-index mode (enforced):
    - Separate indexes for patent-level and claim-level data
    - Automatically routes queries to the appropriate index based on level
    """
    api_key: str
    patent_index_host: str
    claim_index_host: str
    namespace: str = "default"

    def __post_init__(self):
        """Validate that both index hosts are provided."""
        if not self.patent_index_host:
            raise ValueError("'patent_index_host' is required for dual-index mode")
        if not self.claim_index_host:
            raise ValueError("'claim_index_host' is required for dual-index mode")


@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch Cloud connection.

    .. deprecated::
        Replaced by :class:`QdrantConfig`. Retained for backwards compatibility.
    """
    api_key: str
    cloud_id: str = ""
    hosts: Optional[List[str]] = None
    index_name: str = "patents"


@dataclass
class QdrantConfig:
    """Configuration for Qdrant hybrid-vector store (sparse + dense).

    Manages two collections:
    - ``patent_collection_name``: holds patent-level points with both a dense
      (OpenAI) vector and a BM25 sparse vector.
    - ``claim_collection_name``: holds claim-level points with a dense vector
      only.

    Environment variables:
        QDRANT_URL                     e.g. https://<cluster>.cloud.qdrant.io:6333
        QDRANT_API_KEY                 Qdrant Cloud API key
        QDRANT_PATENT_COLLECTION_NAME  (default: patents_hybrid)
        QDRANT_CLAIM_COLLECTION_NAME   (default: claims_hybrid)
        QDRANT_DENSE_VECTOR_SIZE       (default: 1536, matches text-embedding-3-small)
        QDRANT_TIMEOUT                 (default: 30.0 seconds)
        QDRANT_UPLOAD_BATCH_SIZE       (default: 100 points per batch)
    """
    # Required
    url: str                                    # e.g. https://<cluster-id>.gcp.cloud.qdrant.io:6333
    api_key: str                                # Qdrant API key (required for Qdrant Cloud)

    # Collections
    patent_collection_name: str = "patents_hybrid"  # dense + BM25 sparse
    claim_collection_name: str = "claims_hybrid"    # dense only

    # Vector shape
    dense_vector_size: int = 1536               # must match the OpenAI embedding model dimension

    # Performance
    timeout: float = 30.0                       # seconds
    upload_batch_size: int = 100                # points per upsert batch
