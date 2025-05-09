import os
import shutil
import subprocess
import tempfile
import csv
import pytest
import vcr

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
CASSETTES = os.path.join(os.path.dirname(__file__), 'cassettes')
MAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/main.py'))

os.environ['OPENAI_API_KEY'] = 'dummy-key'  # For all tests

def run_cli(input_json, output_csv=None, extra_args=None):
    args = ['python', MAIN, input_json]
    if output_csv:
        args += ['--output', output_csv]
    if extra_args:
        args += extra_args
    result = subprocess.run(args, capture_output=True, text=True)
    return result

@vcr.use_cassette(os.path.join(CASSETTES, 'basic_functionality.yaml'))
def test_basic_functionality():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(FIXTURES, 'input_basic.json')
        output_csv = os.path.join(tmpdir, 'output.csv')
        print(output_csv)
        result = run_cli(input_json, output_csv)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(output_csv)
        with open(output_csv, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 4  # 2 models x 2 iterations x 1 template
            for row in rows:
                assert 'a' in row and 'b' in row and 'model-name' in row
                assert row['chosen_answer'] in ('alpha', 'beta')

@vcr.use_cassette(os.path.join(CASSETTES, 'skip_rows.yaml'))
def test_skip_already_done():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(FIXTURES, 'input_skip.json')
        preexisting = os.path.join(FIXTURES, 'preexisting_output.csv')
        output_csv = os.path.join(tmpdir, 'output.csv')
        shutil.copy(preexisting, output_csv)
        result = run_cli(input_json, output_csv)
        assert result.returncode == 0, result.stderr
        with open(output_csv, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Should have 2 rows: 1 preexisting, 1 new
            assert len(rows) == 2
            seeds = set(row['seed'] for row in rows)
            assert '0' in seeds and '1' in seeds

@vcr.use_cassette(os.path.join(CASSETTES, 'error_handling.yaml'))
def test_error_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(FIXTURES, 'input_error.json')
        output_csv = os.path.join(tmpdir, 'output.csv')
        result = run_cli(input_json, output_csv)
        assert result.returncode != 0
        assert 'possible_answers' in result.stderr or 'possible_answers' in result.stdout
        assert not os.path.exists(output_csv)

@vcr.use_cassette(os.path.join(CASSETTES, 'model_config.yaml'))
def test_model_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(FIXTURES, 'input_model_config.json')
        output_csv = os.path.join(tmpdir, 'output.csv')
        result = run_cli(input_json, output_csv)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(output_csv)
        with open(output_csv, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2  # 2 models x 1 iteration
            # Check that model config is reflected in output
            names = set(row['model-name'] for row in rows)
            assert 'gpt-4.1-nano' in names and 'gpt-4o-mini' in names
            temps = set(row['temperature'] for row in rows)
            assert '0.5' in temps and '1.0' in temps 