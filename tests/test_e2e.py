import os
import subprocess
import pytest
from pathlib import Path
import pandas as pd
import tempfile
import shutil
import multiprocessing

# Constants for directory paths
TESTS_DATA_DIR = os.path.join("tests", "data")
INPUTS_DIR = os.path.join(TESTS_DATA_DIR, "inputs")
ACTUAL_DIR = os.path.join(TESTS_DATA_DIR, "actual")
EXPECTED_DIR = os.path.join(TESTS_DATA_DIR, "expected")

def pytest_addoption(parser):
    parser.addoption(
        "--parallel",
        action="store",
        default="1",
        help="Number of parallel processes to use for testing"
    )

@pytest.fixture
def parallel_workers(request):
    """Fixture to get the number of parallel workers"""
    return int(request.config.getoption("--parallel"))

@pytest.fixture
def run_cli():
    """Fixture to run CLI commands and verify output"""
    def _run_cli(input_json: str, output_dir: str, parallel: int = 1) -> bool:
        cmd = ['python', '-m', 'src.main', input_json, '--batch-seed', '42', "--output-dir", output_dir, '--parallel', str(parallel)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True  # This will raise CalledProcessError if return code != 0
            )
            
            print("stdout:")
            print(result.stdout)
            
            assert result.returncode == 0, f"Error: Return code {result.returncode} != 0"
            
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Return code: {e.returncode}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise e
    
    return _run_cli

@pytest.mark.parametrize("case_name", [
    "complex_pipeline",
    "custom_variables",
    "elo.simple",
    "elo.ranking-numbers",
    "prompt_list_of_answers.many_variables_nodes", 
    "prompt_list_of_answers.simple", 
    "simple_variables",
])
@pytest.mark.parametrize("parallel", [1, 16])
def test_e2e(case_name, run_cli, generate_missing, parallel):
    """End-to-end test comparing all output files between actual and expected directories"""
    
    # Setup paths
    input_json_path = os.path.join(INPUTS_DIR, f"{case_name}.json")
    actual_case_dir = os.path.join(ACTUAL_DIR, f"{case_name}")
    expected_case_dir = os.path.join(EXPECTED_DIR, f"{case_name}")
    
    # Verify input file exists
    assert os.path.exists(input_json_path), f"Input file {input_json_path} not found"
    
    # Clear actual directory first
    if os.path.exists(actual_case_dir):
        shutil.rmtree(actual_case_dir)
    os.makedirs(actual_case_dir, exist_ok=True)
    
    # Run the CLI with actual directory as output
    run_cli(input_json_path, actual_case_dir, parallel)
    
    # If expected directory doesn't exist and generate_missing is True, copy the actual output
    if not os.path.exists(expected_case_dir) and generate_missing:
        shutil.copytree(actual_case_dir, expected_case_dir, dirs_exist_ok=True)
        pytest.skip(f"Generated expected output files in: {expected_case_dir}")
    
    # If expected directory doesn't exist and generate_missing is False, mark as failed
    if not os.path.exists(expected_case_dir):
        pytest.fail(f"Expected output directory {expected_case_dir} not found and --generate-missing not specified")
    
    # Compare all files in actual and expected directories
    actual_files = set(os.listdir(actual_case_dir))
    expected_files = set(os.listdir(expected_case_dir))
    
    # Check if all files match
    if actual_files != expected_files:
        pytest.fail(f"Files in actual and expected directories don't match.\n"
                   f"Actual files: {actual_files}\n"
                   f"Expected files: {expected_files}")
    
    # Compare each file
    for filename in actual_files:
        actual_path = os.path.join(actual_case_dir, filename)
        expected_path = os.path.join(expected_case_dir, filename)
        
        if filename.endswith('.csv'):
            # For CSV files, use pandas to compare with sorting
            actual_df = pd.read_csv(actual_path)
            expected_df = pd.read_csv(expected_path)
            
            # Sort both dataframes by all columns
            actual_df = actual_df.sort_values(by=list(actual_df.columns))
            expected_df = expected_df.sort_values(by=list(expected_df.columns))
            
            # Reset index after sorting
            actual_df = actual_df.reset_index(drop=True)
            expected_df = expected_df.reset_index(drop=True)
            
            # Compare the dataframes
            pd.testing.assert_frame_equal(actual_df, expected_df, check_names=True)
        else:
            # For non-CSV files, do direct file comparison
            with open(actual_path) as f1, open(expected_path) as f2:
                actual_content = f1.read()
                expected_content = f2.read()
                assert actual_content == expected_content, f"Content mismatch in file: {filename}" 