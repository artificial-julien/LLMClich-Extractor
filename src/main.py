import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.pipeline import Pipeline
from src.execution import Execution
from typing import Dict, Any, Optional

def process_json_file(json_path: str, output_dir: Optional[str] = None, verbose: bool = False, parallel: int = 1) -> None:
    """
    Process a JSON configuration file and run the pipeline.
    
    Args:
        json_path: Path to the JSON configuration file
        output_dir: Optional base directory for output files. If not provided, uses the input file's directory.
        verbose: Whether to enable verbose logging
        parallel: Number of parallel requests
    """    
    # Load the JSON configuration
    json_path = Path(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if verbose:
        print(f"Loaded configuration from {json_path}")
        print(f"Pipeline has {len(config.get('foreach', []))} stages")
    
    # Determine input/output folder
    input_folder = str(json_path.parent)
    output_folder = output_dir if output_dir is not None else input_folder
    
    # Create the pipeline with input/output folder
    pipeline = Pipeline.from_config(config, input_folder=output_folder)
    
    if verbose:
        print(f"Created pipeline with {len(pipeline.stages)} stages")
    
    # Run the pipeline
    results = pipeline.run()
    
    if verbose:
        print(f"Pipeline execution complete, produced {len(results)} executions")
    
def main():
    """CLI entrypoint for processing JSON pipeline configurations."""
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process JSON pipeline configurations')
    parser.add_argument('input_json', help='JSON input file with pipeline configuration')
    parser.add_argument('--output-dir', help='Base directory for output files (default: same as input file)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--parallel', type=int, default=2, help='Number of parallel requests (default: 2)')
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Process the JSON file
    process_json_file(args.input_json, output_dir=args.output_dir, verbose=args.verbose, parallel=args.parallel)

if __name__ == "__main__":
    main() 