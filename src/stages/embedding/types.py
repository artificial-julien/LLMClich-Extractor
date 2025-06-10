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
class DistanceExecution(Execution):
    """
    Represents a distance measurement between two items.
    """
    item1: str
    item2: str
    distance_value: float
    distance_metric: str

    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_distance_item1': self.item1,
            '_distance_item2': self.item2,
            '_distance_value': self.distance_value,
            '_distance_metric': self.distance_metric
        } 