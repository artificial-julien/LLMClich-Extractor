#!/bin/bash
set -e

source venv/bin/activate

mkdir -p .cache

# Pass all arguments to pytest
pytest tests/test_e2e.py "$@"
