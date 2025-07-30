"""
Common script runner utilities for pipeline scripts.
Provides imperative-style setup functions and global configuration access.
"""

import os
import sys
from pathlib import Path

# Setup project path immediately on import, before other imports
def _setup_project_path():
    """Add project root to Python path for imports."""
    # Find project root (assuming this file is in src/common/)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Call immediately on import
_setup_project_path()

# Now we can safely import from src
import argparse
from dotenv import load_dotenv
from src.common.types import *

# Global configuration accessible to all scripts
global_config: PipelineConfig = None
parser: argparse.ArgumentParser = None

def setup_project_path():
    """Add project root to Python path for imports (already done on import)."""
    # This is now a no-op since we do it on import
    pass

def setup_environment():
    """Load environment variables and verify required settings."""
    load_dotenv()
    
    # Verify API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

def create_parser(description: str = "Pipeline script") -> argparse.ArgumentParser:
    """Create argument parser with common pipeline arguments."""
    global parser
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments
    parser.add_argument('--output-dir', 
                       help='Base directory for output files (default: "output")')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--parallel', type=int, default=1, 
                       help='Number of parallel requests (default: 1)')
    parser.add_argument('--batch-seed', type=int, 
                       help='Seed for batch generation reproducibility')
    parser.add_argument('--csv-append', action='store_true', default=False, 
                       help='Append to existing CSV files instead of overwriting')
    parser.add_argument('--llm-max-retries', type=int, default=1, 
                       help='Maximum number of retries for LLM calls (default: 1)')
    parser.add_argument('--llm-seed', type=int, 
                       help='Seed for LLM calls reproducibility')

def add_argument(*args, **kwargs):
    """Add a custom argument to the parser."""
    if parser is None:
        raise RuntimeError("Must call create_parser() first")
    return parser.add_argument(*args, **kwargs)

def load_arguments():
    """Parse command line arguments and create global configuration."""
    global global_config, args
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    global_config = PipelineConfig(
        output_dir=args.output_dir or "output",
        verbose=args.verbose,
        parallel=args.parallel,
        llm_max_tries=args.llm_max_retries,
        llm_seed=args.llm_seed,
        batch_seed=args.batch_seed,
        csv_append=args.csv_append,
        custom_args=args
    )

def common_setup(description: str = "Pipeline script"):
    """Perform all common setup steps in one call."""
    setup_environment()  # setup_project_path() is now done on import
    create_parser(description)