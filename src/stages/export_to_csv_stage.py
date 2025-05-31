from typing import List, Dict, Any, Optional, Type
import pandas as pd
from pathlib import Path
from src.stage import Stage
from src.execution import Execution
from src.common.types import *
import os

class ExportToCsvStage(Stage):
    """
    Stage that exports execution results to a CSV file.
    """
    
    DEFAULT_OUTPUT_FILE = "output.csv"
    
    def __init__(self, output_file: Optional[str] = None, columns: List[str] = None, skip_non_full_rows: bool = False, type_filter: Optional[List[Type[Execution]]] = None):
        """
        Initialize the export to CSV stage.
        
        Args:
            output_file: Path to the output CSV file. If not provided, defaults to "output.csv"
            columns: List of variable names to include as columns
            skip_non_full_rows: If True, rows with any empty fields will be skipped. Defaults to True.
            type_filter: List of Execution subtypes to include. If None, all types are included.
        """
        self.output_file = output_file or self.DEFAULT_OUTPUT_FILE
        self.columns = columns or []
        self.skip_non_full_rows = skip_non_full_rows
        self.type_filter = type_filter
    

    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions by extracting variables and saving them to a CSV file.
        If no columns are specified, all variables from all executions will be included.
        
        Args:
            executions: List of input executions
            
        Returns:
            The same list of executions (this stage doesn't modify executions)
        """
        # Extract variables from each execution
        rows = []
        all_columns = set(self.columns) if self.columns else set()
        
        for execution in executions:
            if execution.has_error():
                continue
                
            # Apply type filter if specified
            if self.type_filter is not None and not any(isinstance(execution, t) for t in self.type_filter):
                continue
                
            # Get all variables for this execution
            all_vars = execution.get_all_variables()
            
            # If no columns specified, add all new columns we find
            if not self.columns:
                all_columns.update(all_vars.keys())
            
            # Create row with either specified columns or all variables
            row = {}
            columns_to_use = self.columns or all_columns
            for column in columns_to_use:
                row[column] = all_vars.get(column)
            rows.append(row)
        
        # Convert to DataFrame
        if not rows:
            return executions
            
        new_data = pd.DataFrame(rows)
        
        # Filter out rows with empty fields if skip_non_full_rows is True
        if self.skip_non_full_rows:
            new_data = new_data.dropna(how='any')
        
        # Resolve output path relative to output folder if provided
        if pipeline_config.output_dir:
            output_path = Path(pipeline_config.output_dir) / self.output_file
        else:
            output_path = Path(pipeline_config.json_path).parent / self.output_file
            
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and handle accordingly
        if output_path.exists() and output_path.stat().st_size > 0 and pipeline_config.csv_append:
            try:
                # Try to read existing file
                existing_df = pd.read_csv(str(output_path))
                
                # Check if the file has headers
                has_headers = True
                if len(existing_df.columns) == 1 and existing_df.columns[0].startswith('Unnamed:'):
                    # File likely has no headers, read again without header
                    existing_df = pd.read_csv(str(output_path), header=None)
                    has_headers = False
                
                if not has_headers:
                    # Add headers to the existing file
                    existing_df.columns = self.columns[:len(existing_df.columns)]
                    # Add any missing columns
                    for col in self.columns:
                        if col not in existing_df.columns:
                            existing_df[col] = None
                
                else:
                    # Add any missing columns to existing dataframe
                    for col in self.columns:
                        if col not in existing_df.columns:
                            existing_df[col] = None
                
                # Append new data to existing data
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                
                # Save combined data
                combined_df.to_csv(str(output_path), index=False, na_rep='')
                
            except Exception as e:
                # In case of any error reading the existing file, overwrite with new data
                new_data.to_csv(str(output_path), index=False, na_rep='')
        else:
            # If file doesn't exist, is empty, or append is disabled, just write the new data
            new_data.to_csv(str(output_path), index=False, na_rep='')
        
        return executions 