import src.common.script_runner as script_runner
from typing import List, Dict, Any, Optional, Type
import pandas as pd
from pathlib import Path
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from src.stages.embedding.types import DistanceMatrixExecution

class ExportMatrixToCsvStage(Stage):
    """
    Stage that exports distance matrices to CSV files.
    
    Uses type filtering to process only DistanceMatrix executions and exports
    each matrix as a CSV file with items as row and column headers.
    """
    
    DEFAULT_OUTPUT_FILE_PREFIX = "distance_matrix"
    
    def __init__(self, output_file_prefix: Optional[str] = None):
        """
        Initialize the export matrix to CSV stage.
        
        Args:
            output_file: Path to the output CSV file. If not provided, defaults to "distance_matrix.csv"
        """
        self.output_file_prefix = output_file_prefix or self.DEFAULT_OUTPUT_FILE_PREFIX
    
    def _export_matrix_to_csv(self, matrix_execution: DistanceMatrixExecution, output_path: Path) -> None:
        """
        Export a single distance matrix to CSV format.
        
        Args:
            matrix_execution: DistanceMatrix execution to export
            output_path: Path where to save the CSV file
        """
        # Create DataFrame with items as both index and columns
        df = pd.DataFrame(
            matrix_execution.distance_matrix,
            index=matrix_execution.items,
            columns=matrix_execution.second_dimension_items
        )
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with index and header
        df.to_csv(str(output_path), float_format='%.6f')
    
    def _generate_output_filename(self, matrix_execution: DistanceMatrixExecution, filename_prefix: str) -> str:
        """
        Generate a unique output filename for a matrix execution.
        Follows the format:
        <distance>_<matrix>_<model>_<metric>_<size>.csv
        
        Args:
            matrix_execution: DistanceMatrix execution
            filename_prefix: Filename prefix
            
        Returns:
            Generated filename with metadata
        """
        
        # Add metadata to filename
        metadata_parts = [
            f"model_{matrix_execution.embedding_model_config.name.replace('/', '_').replace('-', '_')}",
            f"metric_{matrix_execution.distance_metric}",
            f"size_{len(matrix_execution.items)}"
        ]
        
        metadata_suffix = "_".join(metadata_parts)
        return f"{filename_prefix}_{metadata_suffix}.csv"
    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions by filtering DistanceMatrix types and exporting them to CSV.
        
        Args:
            executions: List of input executions
            
        Returns:
            The same list of executions (this stage doesn't modify executions)
        """
        matrix_executions = []
        
        for execution in executions:
            if isinstance(execution, DistanceMatrixExecution):
                matrix_executions.append(execution)
        
        # Export each matrix execution
        for i, matrix_execution in enumerate(matrix_executions):
            try:
                output_filename = self._generate_output_filename(matrix_execution, self.output_file_prefix)
                
                # Resolve output path relative to output folder if provided
                if script_runner.global_config.output_dir:
                    output_path = Path(script_runner.global_config.output_dir) / output_filename
                else:
                    # Fallback to current directory if no output_dir specified
                    output_path = Path(output_filename)
                
                # Export the matrix
                self._export_matrix_to_csv(matrix_execution, output_path)
                
                if script_runner.global_config.verbose:
                    print(f"Exported distance matrix to: {output_path}")
                    print(f"  - Items: {len(matrix_execution.items)}")
                    print(f"  - Embedding model: {matrix_execution.embedding_model}")
                    print(f"  - Distance metric: {matrix_execution.distance_metric}")
                
            except Exception as e:
                if pipeline_config.verbose:
                    print(f"Failed to export matrix execution {i}: {str(e)}")
                # Note: We don't modify the execution's error state since this is an export stage
        
        return executions