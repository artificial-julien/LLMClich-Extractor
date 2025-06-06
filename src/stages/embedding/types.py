from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.execution import Execution

@dataclass(kw_only=True)
class EmbeddingExecution(Execution):
    """
    Represents a single item with its computed embedding.
    is_second_dimension: If True, the embedding is the second dimension for the matrix of distances.
    """
    item: str
    embedding: List[float]
    is_second_dimension: bool

    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_embedding_item': self.item,
            '_embedding_dimensions': len(self.embedding),
            '_embedding_vector': self.embedding
        }

@dataclass(kw_only=True)
class DistanceMatrixExecution(Execution):
    """
    Represents a distance matrix computed from embeddings.
    Can be either square (n*n) or rectangular (n*m) when second dimension items are specified.
    """
    items: List[str]
    second_dimension_items: List[str]
    distance_matrix: List[List[float]]
    distance_metric: str

    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_matrix_items': self.items,
            '_matrix_size': len(self.items),
            '_matrix_second_dimension_items': self.second_dimension_items,
            '_matrix_second_dimension_size': len(self.second_dimension_items) if self.second_dimension_items else len(self.items),
            '_matrix_distance_metric': self.distance_metric,
            '_matrix_data': self.distance_matrix
        } 