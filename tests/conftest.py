import os
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration"""
    return {
        "test_repo": "test_repo",
        "test_branch": "test_branch",
        "test_commit": "test_commit"
    }

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    # Set test environment variables
    monkeypatch.setenv("TEST_ENV", "test")
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # For deterministic behavior 