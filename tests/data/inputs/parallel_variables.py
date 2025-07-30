#!/usr/bin/env python3
"""
Test file demonstrating parallel stage processing.
Shows how to process variables through multiple stages in parallel.
"""

# Setup project path first, before any src imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.script_runner import common_setup, load_arguments
from src.stages import *
import src.common.script_runner as script_runner

def main():
    common_setup("Parallel variables pipeline with CSV export")
    
    load_arguments()
    
    names = ["Alice", "Bob", "Jean"]
    cities = ["New York", "London", "Paris"]
    
    variables_stage = VariablesStage([
        {
            "name": names[i % len(names)],
            "age": 25 + i * 5,
            "city": cities[i % len(cities)]
        }
        for i in range(3)
    ])
    
    greeting_stage = VariablesStage([{"greeting": f"Hello, {name}!"} for name in names])
    
    city_stage = VariablesStage([{"city_info": f"Talking to {name}"} for name in names])
    
    export_stage = ExportToCsvStage(
        output_file="output.csv",
        columns=["name", "age", "city", "greeting", "city_info"]
    )
    
    pipeline = variables_stage | (greeting_stage & city_stage) | export_stage
    
    pipeline.invoke()

if __name__ == "__main__":
    main() 