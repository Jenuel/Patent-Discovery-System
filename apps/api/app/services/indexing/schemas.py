from typing import List, TypedDict
from dataclasses import dataclass

class SparseVector(TypedDict):
    indices: List[int]
    values: List[float]


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str
    index_host: str
    namespace: str = "default"
