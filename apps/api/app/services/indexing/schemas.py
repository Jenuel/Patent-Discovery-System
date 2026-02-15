from typing import List, Optional, TypedDict
from dataclasses import dataclass

class SparseVector(TypedDict):
    indices: List[int]
    values: List[float]


@dataclass
class PineconeConfig:
    """
    Configuration for Pinecone connection.
    
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
    """Configuration for Elasticsearch Cloud connection."""
    api_key: str
    cloud_id: str = ""
    hosts: Optional[List[str]] = None
    index_name: str = "patents"
