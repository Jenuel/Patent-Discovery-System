from typing import List, Optional, TypedDict
from dataclasses import dataclass

class SparseVector(TypedDict):
    indices: List[int]
    values: List[float]


@dataclass
class PineconeConfig:
    """
    Configuration for Pinecone connection.
    
    Supports two modes:
    1. Single index mode: Use index_host for both patent and claim levels
    2. Dual index mode: Use patent_index_host and claim_index_host for separate indexes
    """
    api_key: str
    # Single index mode (backward compatible)
    index_host: str = ""
    # Dual index mode (patent-level and claim-level)
    patent_index_host: str = ""
    claim_index_host: str = ""
    namespace: str = "default"
    
    def __post_init__(self):
        """Validate that at least one index configuration is provided."""
        if not self.index_host and not (self.patent_index_host and self.claim_index_host):
            raise ValueError(
                "Either 'index_host' (single index) or both 'patent_index_host' "
                "and 'claim_index_host' (dual index) must be provided"
            )


@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch Cloud connection."""
    api_key: str
    cloud_id: str = ""
    hosts: Optional[List[str]] = None
    index_name: str = "patents"
