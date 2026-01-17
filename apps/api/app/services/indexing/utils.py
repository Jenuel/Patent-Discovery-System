from typing import List, Optional, Sequence, Tuple
from .schemas import SparseVector

def apply_alpha_to_query(
    dense: Sequence[float],
    sparse: Optional[SparseVector],
    alpha: float,
) -> Tuple[List[float], Optional[SparseVector]]:
    """
    Hybrid weighting by scaling:
      dense_scaled = alpha * dense
      sparse_scaled.values = (1-alpha) * sparse.values
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    dense_scaled = [float(v) * alpha for v in dense]
    if sparse is None:
        return dense_scaled, None
    sparse_scaled: SparseVector = {
        "indices": list(sparse["indices"]),
        "values": [float(v) * (1.0 - alpha) for v in sparse["values"]],
    }
    return dense_scaled, sparse_scaled
