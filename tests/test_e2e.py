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
    parser.addoption(
        "--generate-missing",
        action="store_true",
        help="Generate missing expected output files"
    )

def pytest_generate_tests(metafunc):
    if "parallel" in metafunc.fixturenames:
        generate_missing = metafunc.config.getoption("--generate-missing", False)
        
        if generate_missing:
            # When generating missing, only use parallel=1
            metafunc.parametrize("parallel", [1])
        else:
            # Normal case, use both values
            metafunc.parametrize("parallel", [1, 16])

@pytest.fixture
def run_pipeline():
    """Fixture to run pipeline (either JSON via CLI or Python script directly)"""
    def _run_pipeline(input_file: str, output_dir: str, parallel: int = 1) -> bool:
        input_path = Path(input_file)
        
        cmd = ['python', input_file, '--parallel', str(parallel), "--batch-seed", "42", '--output-dir', output_dir]
        
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
    
    return _run_pipeline

@pytest.mark.parametrize("case_name", [
    ("complex_pipeline"),
    ("custom_variables"),
    ("elo.simple"),
    ("elo.draws"),
    ("embedding.matrix"),
    ("embedding.matrix.rect"),
    ("embedding.matrix.multiple"),
    ("prompt_list_of_answers.many_variables_nodes"), 
    ("prompt_list_of_answers.simple"), 
    ("simple_variables"),
])
def test_e2e(case_name, run_pipeline, generate_missing, parallel):
    """End-to-end test comparing all output files between actual and expected directories"""
    
    # Setup paths
    input_file_path = os.path.join(INPUTS_DIR, f"{case_name}.py")
    actual_case_dir = os.path.join(ACTUAL_DIR, case_name)
    expected_case_dir = os.path.join(EXPECTED_DIR, case_name)
    
    # Verify input file exists
    assert os.path.exists(input_file_path), f"Input file {input_file_path} not found"
    
    # Clear actual directory first
    if os.path.exists(actual_case_dir):
        shutil.rmtree(actual_case_dir)
    os.makedirs(actual_case_dir, exist_ok=True)
    
    # Run the pipeline with actual directory as output
    run_pipeline(input_file_path, actual_case_dir, parallel)
    
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