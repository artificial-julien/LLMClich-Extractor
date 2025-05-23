import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.pipeline import Pipeline
from src.execution import Execution
from typing import Dict, Any, Optional
from src.commons import PipelineConfig


def process_json_file(pipeline_config: PipelineConfig) -> None:
    """
    Process a JSON configuration file and run the pipeline.
    
    Args:
        pipeline_config: PipelineConfig object containing all processing parameters
    """    
    # Load the JSON configuration
    json_path = Path(pipeline_config.json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        pipeline_definition = json.load(f)
    
    if pipeline_config.verbose:
        print(f"Loaded configuration from {json_path}")
        print(f"Pipeline has {len(pipeline_definition.get('foreach', []))} stages")
    
    # Create the pipeline with configuration
    pipeline = Pipeline.from_config(pipeline_definition)
    
    if pipeline_config.verbose:
        print(f"Created pipeline with {len(pipeline.stages)} stages")
    
    # Run the pipeline
    results = pipeline.run(pipeline_config=pipeline_config, initial_variables={})
    
    if pipeline_config.verbose:
        print(f"Pipeline execution complete, produced {len(results)} executions")

def main():
    """CLI entrypoint for processing JSON pipeline configurations."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Process JSON pipeline configurations')
    parser.add_argument('input_json', help='JSON input file with pipeline configuration')
    parser.add_argument('--output-dir', help='Base directory for output files (default: same as input file)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel requests (default: 1)')
    parser.add_argument('--batch-seed', type=int, help='Seed for batch generation reproducibility')
    parser.add_argument('--csv-append', action='store_true', default=False, help='Append to existing CSV files instead of overwriting')
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Create pipeline pipeline_config and process the JSON file
    pipeline_config = PipelineConfig(
        json_path=args.input_json,
        output_dir=args.output_dir,
        verbose=args.verbose,
        parallel=args.parallel,
        batch_seed=args.batch_seed,
        csv_append=args.csv_append
    )
    process_json_file(pipeline_config)

if __name__ == "__main__":
    main() 