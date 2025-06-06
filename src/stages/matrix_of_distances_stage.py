from typing import List, Dict, Any, Type
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from src.stages.embedding.types import EmbeddingExecution, DistanceMatrixExecution
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock, chebyshev
import math
from typing import Literal
from itertools import groupby
from operator import attrgetter

class MatrixOfDistancesStage(Stage):
    """
    Stage that computes distance matrices from embedding executions.
    
    Processes Embedding executions and calculates pairwise distances between items,
    creating a single DistanceMatrix execution with the computed matrix.
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
        Initialize the matrix of distances stage.
        
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
    
    def _build_distance_matrix(self, embeddings: Dict[str, List[float]], embeddings_second_dimension: Dict[str, List[float]]) -> List[List[float]]:
        """
        Build a distance matrix from embeddings.
        
        Args:
            embeddings: Dictionary mapping item names to their embedding vectors
            second_dimension_items: Optional list of items that should be used as the second dimension.
                                  If None, creates a symmetric n*n matrix by using the same items for both dimensions.
            
        Returns:
            2D list representing the distance matrix
        """
        items = list(embeddings.keys())
        second_dimension_items = list(embeddings_second_dimension.keys())
        
        n = len(items)
        m = len(second_dimension_items)
        matrix = [[0.0 for _ in range(m)] for _ in range(n)]
        
        # Calculate distances between all pairs
        for i in range(n):
            for j in range(m):
                distance = self._calculate_distance(embeddings[items[i]], embeddings_second_dimension[second_dimension_items[j]])
                matrix[i][j] = distance
                
                # For symmetric case, copy to the other triangle
                if second_dimension_items is items and i != j:
                    matrix[j][i] = distance
        
        return matrix
    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process Embedding executions and create a DistanceMatrix execution.
        
        Aggregates all Embedding executions with the same embedding model into
        a single distance matrix calculation. If some embeddings are marked as
        second dimension, creates a rectangular matrix instead of a square one.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: List of input executions
            
        Returns:
            List containing DistanceMatrix executions and any non-Embedding executions
        """
        result_executions : List[Execution] = []
        
        # Filter executions by type using list comprehension
        embedding_executions : List[EmbeddingExecution] = [execution for execution in executions 
                              if any(isinstance(execution, t) for t in self.type_filter)]
        other_executions : List[Execution] = [execution for execution in executions 
                          if not any(isinstance(execution, t) for t in self.type_filter)]
        
        result_executions.extend(embedding_executions)
        result_executions.extend(other_executions)

        if not embedding_executions:
            return executions
        
        # Group by model
        model_groups = {
            model_key: list(group) 
            for model_key, group in groupby(embedding_executions, 
                                          key=lambda x: x.embedding_model_config)
        }
        
        for embedding_model, group_executions in model_groups.items():
            try:
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
                
                if len(embeddings) < 2:
                    # Need at least 2 items for distance matrix
                    error_execution = base_execution.copy()
                    error_execution.set_error(f"Need at least 2 items for distance matrix, got {len(embeddings)}")
                    result_executions.append(error_execution)
                    continue
                
                # Build distance matrix
                items = list(embeddings.keys())
                second_dimension_items = list(embeddings_second_dimension.keys())
                distance_matrix = self._build_distance_matrix(
                    embeddings,
                    embeddings_second_dimension
                )
                
                # Create DistanceMatrix execution
                matrix_execution = DistanceMatrixExecution(
                    embedding_model_config=base_execution.embedding_model_config,
                    items=items,
                    second_dimension_items=second_dimension_items,
                    distance_matrix=distance_matrix,
                    distance_metric=self.distance_metric,
                )
                
                # Import variables from base execution
                matrix_execution.import_variables_from(base_execution)
                
                result_executions.append(matrix_execution)
                
            except Exception as e:
                error_execution = group_executions[0].copy()
                error_execution.set_error(f"Failed to compute distance matrix: {str(e)}")
                result_executions.append(error_execution)
        
        return result_executions 