from typing import List, Dict, Any, Type, Iterator, Optional, Union
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from dataclasses import dataclass
from src.stages.embedding.types import DistanceExecution

@dataclass(kw_only=True)
class MatrixExecution(Execution):
    """
    Represents a matrix created from pivoted data.
    """
    row_labels: List[str]
    column_labels: List[str]
    matrix_data: List[List[float]]
    row_key: str
    column_key: str
    value_key: str
    
    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_matrix_row_labels': self.row_labels,
            '_matrix_column_labels': self.column_labels,
            '_matrix_data': self.matrix_data,
            '_matrix_rows': len(self.row_labels),
            '_matrix_columns': len(self.column_labels),
            '_matrix_row_key': self.row_key,
            '_matrix_column_key': self.column_key,
            '_matrix_value_key': self.value_key
        }

class PivotStage(Stage):
    """
    Stage that pivots executions into matrix format.
    
    Takes any execution and creates MatrixExecution objects by pivoting
    based on row, column, and value keys from the execution variables.
    """
    
    def __init__(self, rows: str, columns: str, values: str, fill_value: Any = None):
        """
        Initialize the pivot stage.
        
        Args:
            rows: Variable name to use for matrix rows
            columns: Variable name to use for matrix columns  
            values: Variable name to use for matrix values
            fill_value: Value to use for missing data points
        """
        self.rows = rows
        self.columns = columns
        self.values = values
        self.fill_value = fill_value
    
    def _create_matrix_from_executions(self, executions: List[Execution]) -> tuple[List[str], List[str], List[List[str]]]:
        """
        Create a matrix from a list of executions.
        
        Args:
            executions: List of executions to pivot
            
        Returns:
            Tuple of (row_labels, column_labels, matrix_data)
        """
        # Collect all unique row and column values
        row_values = set()
        column_values = set()
        data_points = {}
        
        for execution in executions:
            # Get values from execution variables
            row_val = execution.get_variable(self.rows)
            col_val = execution.get_variable(self.columns)
            value_val = execution.get_variable(self.values)
            
            if row_val is not None and col_val is not None and value_val is not None:
                row_values.add(row_val)
                column_values.add(col_val)
                
                # Check if key already exists
                key = (row_val, col_val)
                if key in data_points:
                    raise ValueError(f"Duplicate data point found for row '{row_val}' and column '{col_val}'")
                    
                data_points[key] = value_val
        
        # Sort labels for consistent ordering
        row_labels = sorted(list(row_values))
        column_labels = sorted(list(column_values))
        
        # Create matrix
        matrix_data = []
        for row_label in row_labels:
            row_data = []
            for col_label in column_labels:
                value = data_points.get((row_label, col_label), self.fill_value)
                row_data.append(value)
            matrix_data.append(row_data)
        
        return row_labels, column_labels, matrix_data
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process executions by pivoting them into MatrixExecution objects.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            MatrixExecution objects and any executions that couldn't be pivoted
        """
        # Collect executions
        executions_list = list(executions)
        
        # Create matrix from the executions
        row_labels, column_labels, matrix_data = self._create_matrix_from_executions(executions_list)
        
        # Only create matrix if we have data
        if row_labels and column_labels and matrix_data:
            matrix_execution = MatrixExecution(
                row_labels=row_labels,
                column_labels=column_labels,
                matrix_data=matrix_data,
                row_key=self.rows,
                column_key=self.columns,
                value_key=self.values
            )
            
            # Import variables from first execution if available
            if executions_list:
                matrix_execution.import_variables_from(executions_list[0])
            
            yield matrix_execution
        else:
            # No valid data to pivot
            error_execution = Execution()
            error_execution.set_error(f"No valid data to pivot from executions")
            yield error_execution