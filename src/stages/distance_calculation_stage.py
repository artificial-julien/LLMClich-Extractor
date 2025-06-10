from typing import List, Dict, Any, Type, Iterator
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from src.stages.embedding.types import EmbeddingExecution, DistanceExecution
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock, chebyshev
import math
from typing import Literal
from itertools import groupby
from operator import attrgetter

class DistanceCalculationStage(Stage):
    """
    Stage that computes pairwise distances from embedding executions.
    
    Processes Embedding executions and calculates pairwise distances between items,
    yielding individual Distance executions for each pair.
    """
    
    SUPPORTED_METRICS = {
        'cosine': lambda x, y: cosine(x, y),
        'euclidean': lambda x, y: euclidean(x, y),
        'manhattan': lambda x, y: cityblock(x, y),
        'chebyshev': lambda x, y: chebyshev(x, y),
        'dot_product': lambda x, y: -np.dot(x, y),  # Negative for distance interpretation
    }
    
    def __init__(self, distance_metric: Literal["cosine", "euclidean", "manhattan", "chebyshev", "dot_product"] = "cosine"):
        """
        Initialize the distance calculation stage.
        
        Args:
            distance_metric: Distance metric to use ('cosine', 'euclidean', 'manhattan', 'chebyshev', 'dot_product')
        """
        
        self.distance_metric = distance_metric
        self.type_filter = [EmbeddingExecution]
    
    def _calculate_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate distance between two embedding vectors.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Distance value as float
        """
        try:
            distance_func = self.SUPPORTED_METRICS[self.distance_metric]
            
            # Handle special case for identical vectors in cosine distance
            if self.distance_metric == 'cosine' and np.array_equal(embedding1, embedding2):
                return 0.0
                
            # Normalize vectors for cosine distance
            if self.distance_metric == 'cosine':
                embedding1 = np.array(embedding1)
                embedding2 = np.array(embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                if norm1 > 0 and norm2 > 0:
                    embedding1 = embedding1 / norm1
                    embedding2 = embedding2 / norm2
                
            distance = distance_func(embedding1, embedding2)
            
            # Handle NaN values that might occur with cosine distance
            if math.isnan(distance):
                return 0.0 if np.array_equal(embedding1, embedding2) else 1.0
                
            return float(distance)
            
        except Exception as e:
            # Return a high distance value for calculation errors
            return float('inf')

    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process Embedding executions lazily and create Distance executions.
        
        Calculates pairwise distances between all Embedding executions with the same
        embedding model. If some embeddings are marked as second dimension, creates
        distances only between first dimension and second dimension items.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            Distance executions and any non-Embedding executions
        """
        # Collect all executions
        executions_list = list(executions)
        
        # Filter executions by type using list comprehension
        embedding_executions: List[EmbeddingExecution] = [execution for execution in executions_list 
                                if any(isinstance(execution, t) for t in self.type_filter)]
        other_executions: List[Execution] = [execution for execution in executions_list 
                            if not any(isinstance(execution, t) for t in self.type_filter)]
        
        # Yield all original executions first
        for execution in executions_list:
            yield execution

        if not embedding_executions:
            return
        
        # Group by model
        model_groups = {
            model_key: list(group) 
            for model_key, group in groupby(embedding_executions, 
                                            key=lambda x: x.embedding_model_config)
        }
        
        for embedding_model, group_executions in model_groups.items():
            # Extract embeddings and items
            embeddings: Dict[str, List[float]] = {}
            embeddings_second_dimension: Dict[str, List[float]] = {}
            base_execution = Execution(embedding_model_config=embedding_model)
            
            for embedding_exec in group_executions:
                if embedding_exec.is_second_dimension:
                    embeddings_second_dimension[embedding_exec.item] = embedding_exec.embedding
                else:
                    embeddings[embedding_exec.item] = embedding_exec.embedding

            if len(embeddings_second_dimension) == 0:
                embeddings_second_dimension = embeddings
            
            if len(embeddings) < 2 and len(embeddings_second_dimension) == len(embeddings):
                # Need at least 2 items for distance calculation in symmetric case
                error_execution = base_execution.copy()
                error_execution.set_error(f"Need at least 2 items for distance calculation, got {len(embeddings)}")
                yield error_execution
                continue
            
            # Calculate distances between all pairs
            items = list(embeddings.keys())
            second_dimension_items = list(embeddings_second_dimension.keys())
            
            for item1 in items:
                for item2 in second_dimension_items:
                    # Skip same item pairs in symmetric case to avoid duplicates and self-distances
                    if embeddings_second_dimension is embeddings and item1 >= item2:
                        continue
                        
                    distance_value = self._calculate_distance(
                        embeddings[item1], 
                        embeddings_second_dimension[item2]
                    )
                    
                    distance_execution = DistanceExecution(
                        embedding_model_config=base_execution.embedding_model_config,
                        item1=item1,
                        item2=item2,
                        distance_value=distance_value,
                        distance_metric=self.distance_metric,
                    )
                    
                    distance_execution.import_variables_from(base_execution)
                    
                    yield distance_execution
        
        for other_exec in other_executions:
            yield other_exec
