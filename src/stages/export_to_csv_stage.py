from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
import os

@StageRegistry.register("export_to_csv")
class ExportToCsvStage(Stage):
    """
    Stage that exports execution results to a CSV file.
    """
    
    DEFAULT_OUTPUT_FILE = "output.csv"
    
    def __init__(self, output_file: Optional[str] = None, columns: List[str] = None, input_folder: Optional[str] = None):
        """
        Initialize the export to CSV stage.
        
        Args:
            output_file: Path to the output CSV file. If not provided, defaults to "output.csv"
            columns: List of variable names to include as columns
            input_folder: Optional path to the input folder. If provided, output_file will be relative to this folder.
        """
        self.output_file = output_file or self.DEFAULT_OUTPUT_FILE
        self.columns = columns or []
        self.input_folder = input_folder
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ExportToCsvStage':
        """
        Create an ExportToCsvStage from configuration.
        
        Args:
            config: Dictionary containing:
                - 'output_file': Optional path to the output CSV file. If not provided, defaults to "output.csv"
                - 'columns': List of variable names to include as columns
                - 'input_folder': Optional path to the input folder
            
        Returns:
            An ExportToCsvStage instance
            
        Raises:
            ValueError: If the config is invalid
        """
        output_file = config.get('output_file')
        columns = config.get('columns')
        input_folder = config.get('input_folder')
        
        if not columns or not isinstance(columns, list):
            raise ValueError("ExportToCsvStage config must contain a 'columns' list")
        
        return cls(output_file=output_file, columns=columns, input_folder=input_folder)
    
    def process(self, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions by extracting variables and saving them to a CSV file.
        
        Args:
            executions: List of input executions
            
        Returns:
            The same list of executions (this stage doesn't modify executions)
        """
        # Extract variables from each execution
        rows = []
        for execution in executions:
            if execution.has_error():
                continue
                
            row = {}
            for column in self.columns:
                row[column] = execution.get_variable(column)
            rows.append(row)
        
        # Convert to DataFrame
        if not rows:
            return executions
            
        new_data = pd.DataFrame(rows)
        
        # Resolve output path relative to input folder if provided
        if self.input_folder:
            output_path = Path(self.input_folder) / self.output_file
        else:
            output_path = Path(self.output_file)
            
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and handle accordingly
        if output_path.exists() and output_path.stat().st_size > 0:
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
            # If file doesn't exist or is empty, just write the new data
            new_data.to_csv(str(output_path), index=False, na_rep='')
        
        return executions 