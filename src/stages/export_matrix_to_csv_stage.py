import src.common.script_runner as script_runner
from typing import List, Dict, Any, Optional, Type, Iterator
import pandas as pd
from pathlib import Path
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from src.stages.pivot_stage import MatrixExecution

class ExportMatrixToCsvStage(Stage):
    """
    Stage that exports matrices to CSV files.
    
    Uses type filtering to process only Matrix executions and exports
    each matrix as a CSV file with row and column labels as headers.
    """
    
    DEFAULT_OUTPUT_FILE_PREFIX = "matrix"
    
    def __init__(self, output_file_prefix: Optional[str] = None):
        """
        Initialize the export matrix to CSV stage.
        
        Args:
            output_file_prefix: Prefix for the output CSV file. If not provided, defaults to "matrix"
        """
        self.output_file_prefix = output_file_prefix or self.DEFAULT_OUTPUT_FILE_PREFIX
    
    def _export_matrix_to_csv(self, matrix_execution: MatrixExecution, output_path: Path) -> None:
        """
        Export a single matrix to CSV format.
        
        Args:
            matrix_execution: Matrix execution to export
            output_path: Path where to save the CSV file
        """
        # Create DataFrame with row and column labels
        df = pd.DataFrame(
            matrix_execution.matrix_data,
            index=matrix_execution.row_labels,
            columns=matrix_execution.column_labels
        )
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with index and header
        df.to_csv(str(output_path), float_format='%.6f')
    
    def _generate_output_filename(self, matrix_execution: MatrixExecution, filename_prefix: str) -> str:
        """
        Generate a unique output filename for a matrix execution.
        Follows the format:
        <prefix>_<model>_<size>.csv
        
        Args:
            matrix_execution: Matrix execution
            filename_prefix: Filename prefix
            
        Returns:
            Generated filename with metadata
        """
        
        # Add metadata to filename
        metadata_parts = []
        
        if matrix_execution.embedding_model_config:
            metadata_parts.append(f"model_{matrix_execution.embedding_model_config.name.replace('/', '_').replace('-', '_')}")
        
        metadata_parts.append(f"size_{len(matrix_execution.row_labels)}x{len(matrix_execution.column_labels)}")
        
        metadata_suffix = "_".join(metadata_parts)
        return f"{filename_prefix}_{metadata_suffix}.csv"
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process input executions lazily by filtering Matrix types and exporting them to CSV.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            The same executions that were processed (this stage doesn't modify executions)
        """
        matrix_count = 0
        
        for execution in executions:

            # Always yield the execution first
            yield execution
            
            # Process if it's a matrix execution
            if isinstance(execution, MatrixExecution) and not execution.has_error():
                try:
                    output_filename = self._generate_output_filename(execution, self.output_file_prefix)
                    
                    # Resolve output path relative to output folder if provided
                    if script_runner.global_config.output_dir:
                        output_path = Path(script_runner.global_config.output_dir) / output_filename
                    else:
                        # Fallback to current directory if no output_dir specified
                        output_path = Path(output_filename)
                    
                    # Export the matrix
                    self._export_matrix_to_csv(execution, output_path)
                    matrix_count += 1
                    
                    if script_runner.global_config.verbose:
                        print(f"Exported matrix to: {output_path}")
                        print(f"  - Rows: {len(execution.row_labels)}")
                        print(f"  - Columns: {len(execution.column_labels)}")
                        if execution.embedding_model_config:
                            print(f"  - Embedding model: {execution.embedding_model_config.name}")
                    
                except Exception as e:
                    if pipeline_config.verbose:
                        print(f"Failed to export matrix execution {matrix_count}: {str(e)}")
                    # Note: We don't modify the execution's error state since this is an export stage
