from typing import List, Dict, Any, Optional, Type, Iterator
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
    
    def __init__(self, output_file_prefix: Optional[str] = None, columns: List[str] = None, skip_non_full_rows: bool = False, type_filter: Optional[List[Type[Execution]]] = None):
        """
        Initialize the export to CSV stage.
        
        Args:
            output_file: Path to the output CSV file. If not provided, defaults to "output.csv"
            columns: List of variable names to include as columns
            skip_non_full_rows: If True, rows with any empty fields will be skipped. Defaults to True.
            type_filter: List of Execution subtypes to include. If None, all types are included.
        """
        self.output_file_prefix = output_file_prefix or self.DEFAULT_OUTPUT_FILE
        self.columns = columns or []
        self.skip_non_full_rows = skip_non_full_rows
        self.type_filter = type_filter
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process input executions lazily by collecting them for CSV export.
        Since CSV export requires all data, we need to consume the iterator.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            Executions that were filtered out (either by type or non-full rows)
        """
        # Extract variables from each execution
        rows = []
        all_columns = set(self.columns) if self.columns else set()
        
        for execution in executions:                
            # Apply type filter if specified
            if self.type_filter is not None and not any(isinstance(execution, t) for t in self.type_filter):
                yield execution
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
            
            # If skip_non_full_rows is True and any value is None, yield the execution
            if self.skip_non_full_rows and any(v is None for v in row.values()):
                yield execution
                continue
                
            rows.append(row)
        
        # Convert to DataFrame and export
        if rows:
            new_data = pd.DataFrame(rows)
            
            # Resolve output path relative to output folder if provided
            if pipeline_config.output_dir:
                output_path = Path(pipeline_config.output_dir) / self.output_file_prefix
            else:
                output_path = Path(pipeline_config.json_path).parent / self.output_file_prefix
                
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