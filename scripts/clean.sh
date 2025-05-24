#!/bin/bash

source venv/bin/activate

# Run autoflake on all Python files
autoflake --in-place --remove-all-unused-imports --recursive src/ tests/

# If there are changes, commit them
if git diff --quiet; then
    echo "No changes to commit"
else
    git add src/ tests/
    git commit -m "chore: remove unused imports"
fi