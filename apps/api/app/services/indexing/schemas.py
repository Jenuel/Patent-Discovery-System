from typing import List, Optional, TypedDict
from dataclasses import dataclass

class SparseVector(TypedDict):
    indices: List[int]
    values: List[float]


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str
    index_host: str
    namespace: str = "default"


@dataclass(frozen=True)
class ElasticsearchConfig:
    """Configuration for Elasticsearch Cloud connection."""
    api_key: str
    cloud_id: str = ""
    hosts: Optional[List[str]] = None
    index_name: str = "patents"
