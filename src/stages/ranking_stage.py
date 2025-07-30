from typing import List, Dict, Any, Iterator, Literal
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from src.stages.pivot_stage import MatrixExecution
from dataclasses import dataclass
import numpy as np

@dataclass(kw_only=True)
class RankingExecution(Execution):
    """
    Represents a ranking result with ranked data and ranking information.
    """
    original_matrix_data: List[List[float]]
    ranked_matrix_data: List[List[float]]
    ranking_matrix: List[List[int]]
    row_labels: List[str]
    column_labels: List[str]
    ranking_direction: Literal["rows", "columns"]
    sort_order: Literal["asc", "desc"]
    
    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_ranking_original_matrix': self.original_matrix_data,
            '_ranking_ranked_matrix': self.ranked_matrix_data,
            '_ranking_matrix': self.ranking_matrix,
            '_ranking_row_labels': self.row_labels,
            '_ranking_column_labels': self.column_labels,
            '_ranking_direction': self.ranking_direction,
            '_ranking_sort_order': self.sort_order,
            '_ranking_rows': len(self.row_labels),
            '_ranking_columns': len(self.column_labels)
        }

class RankingStage(Stage):
    """
    Stage that applies ranking with tie-breaking to matrix data.
    
    Takes MatrixExecution objects from the pivot stage and applies ranking
    with tie-breaking functionality. Supports ranking by rows or columns
    in ascending or descending order.
    """
    
    def __init__(self, 
                 direction: Literal["rows", "columns"] = "rows",
                 sort_order: Literal["asc", "desc"] = "asc"):
        """
        Initialize the ranking stage.
        
        Args:
            direction: Whether to rank by "rows" or "columns" (default: "rows")
            sort_order: Whether to sort in "asc"ending or "desc"ending order (default: "asc")
        """
        self.direction = direction
        self.sort_order = sort_order
    
    def _apply_tie_breaking_ranking(self, values: List[float]) -> List[int]:
        """
        Apply tie-breaking ranking to a list of values.
        
        For example: [1, 1, 2, 3, 3, 5] becomes [1, 1, 3, 4, 4, 6]
        
        Args:
            values: List of values to rank
            
        Returns:
            List of ranks with tie-breaking
        """
        # Create list of (value, index) pairs
        indexed_values = [(value, i) for i, value in enumerate(values)]
        
        # Sort by value (ascending or descending)
        reverse = self.sort_order == "desc"
        indexed_values.sort(key=lambda x: x[0], reverse=reverse)
        
        # Apply tie-breaking ranking
        ranks = [0] * len(values)
        current_rank = 1
        
        for i, (value, original_index) in enumerate(indexed_values):
            # If this is the first item or the value is different from the previous
            if i == 0 or indexed_values[i-1][0] != value:
                current_rank = i + 1
            
            ranks[original_index] = current_rank
        
        return ranks
    
    def _rank_matrix_by_rows(self, matrix_data: List[List[float]]) -> tuple[List[List[float]], List[List[int]]]:
        """
        Rank matrix by rows with tie-breaking.
        
        Args:
            matrix_data: Original matrix data
            
        Returns:
            Tuple of (ranked_matrix_data, ranking_matrix)
        """
        ranked_data = []
        ranking_matrix = []
        
        for row in matrix_data:
            # Apply ranking to this row
            ranks = self._apply_tie_breaking_ranking(row)
            ranking_matrix.append(ranks)
            
            # Create ranked row (values sorted by rank)
            ranked_row = [row[i] for i in np.argsort(ranks)]
            ranked_data.append(ranked_row)
        
        return ranked_data, ranking_matrix
    
    def _rank_matrix_by_columns(self, matrix_data: List[List[float]]) -> tuple[List[List[float]], List[List[int]]]:
        """
        Rank matrix by columns with tie-breaking.
        
        Args:
            matrix_data: Original matrix data
            
        Returns:
            Tuple of (ranked_matrix_data, ranking_matrix)
        """
        # Transpose matrix to work with columns as rows
        transposed = list(zip(*matrix_data))
        
        ranked_transposed = []
        ranking_transposed = []
        
        for col in transposed:
            # Apply ranking to this column
            ranks = self._apply_tie_breaking_ranking(list(col))
            ranking_transposed.append(ranks)
            
            # Create ranked column (values sorted by rank)
            ranked_col = [col[i] for i in np.argsort(ranks)]
            ranked_transposed.append(ranked_col)
        
        # Transpose back to original format
        ranked_data = list(zip(*ranked_transposed))
        ranking_matrix = list(zip(*ranking_transposed))
        
        return [list(row) for row in ranked_data], [list(row) for row in ranking_matrix]
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process MatrixExecution objects by applying ranking with tie-breaking.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions (should be MatrixExecution objects)
            
        Yields:
            RankingExecution objects with ranking results
        """
        for execution in executions:
            # Check if this is a MatrixExecution
            if not isinstance(execution, MatrixExecution):
                # Skip non-matrix executions but preserve them
                yield execution
                continue
            
            # Apply ranking based on direction
            if self.direction == "rows":
                ranked_data, ranking_matrix = self._rank_matrix_by_rows(execution.matrix_data)
            else:  # columns
                ranked_data, ranking_matrix = self._rank_matrix_by_columns(execution.matrix_data)
            
            # Create ranking execution
            ranking_execution = RankingExecution(
                original_matrix_data=execution.matrix_data,
                ranked_matrix_data=ranked_data,
                ranking_matrix=ranking_matrix,
                row_labels=execution.row_labels,
                column_labels=execution.column_labels,
                ranking_direction=self.direction,
                sort_order=self.sort_order
            )
            
            # Import variables from the original execution
            ranking_execution.import_variables_from(execution)
            
            yield ranking_execution 