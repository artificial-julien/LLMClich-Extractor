#!/usr/bin/env python
"""
Refactored version using the new imperative-style common setup.
Demonstrates the new approach with maximum flexibility.
"""

# Setup project path first, before any src imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.script_runner import common_setup, add_argument, load_arguments
from src.stages import *
import src.common.script_runner as script_runner

def main():
    common_setup("Simple variables pipeline with CSV export")
    
    add_argument('--variable-count', type=int, default=3, 
                help='Number of variable sets to generate (default: 3)')
    add_argument('--custom-output-name', 
                help='Custom name for output file (default: output.csv)')
    
    load_arguments()
    
    variable_sets = []
    names = ["Alice", "Bob", "Charlie", "David", "Eve"]
    cities = ["New York", "San Francisco", "Chicago", "Boston", "Seattle"]
    
    for i in range(script_runner.global_config.custom_args.variable_count):
        variable_sets.append({
            "name": names[i % len(names)],
            "age": 25 + i * 5,
            "city": cities[i % len(cities)]
        })
    
    variables_stage = VariablesStage(variable_sets)
    
    output_file = script_runner.global_config.custom_args.custom_output_name or "output.csv"
    export_stage = ExportToCsvStage(
        output_file=output_file,
        columns=["name", "age", "city"]
    )
    
    pipeline = variables_stage | export_stage
    
    pipeline.invoke()

if __name__ == "__main__":
    main() 