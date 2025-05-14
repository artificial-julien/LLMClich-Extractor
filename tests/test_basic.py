import pytest
from pathlib import Path
import os

def test_test_data_dir(test_data_dir):
    """Test that test data directory is created and accessible"""
    assert isinstance(test_data_dir, Path)
    assert test_data_dir.exists()
    assert test_data_dir.is_dir()

def test_test_config(test_config):
    """Test that test configuration is properly set up"""
    assert isinstance(test_config, dict)
    assert "test_repo" in test_config
    assert "test_branch" in test_config
    assert "test_commit" in test_config

def test_environment_variables():
    """Test that environment variables are properly set"""
    assert os.environ.get("TEST_ENV") == "test"
    assert os.environ.get("PYTHONHASHSEED") == "0" 