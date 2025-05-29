#!/usr/bin/env python3
"""
Pipeline demonstrating grammatical correctness evaluation with custom variables.
"""

# Setup project path first, before any src imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.script_runner import common_setup, add_argument, load_arguments
from src.stages import *
from src.common.types import *

def main():
    common_setup("Grammatical correctness evaluation pipeline")
    
    load_arguments()
    
    model = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        iterations=1
    )
    
    variable_sets = [
        {
            "adjective": "happy",
            "noun": "dog",
            "verb": "runs"
        },
        {
            "adjective": "lazy",
            "noun": "cat",
            "verb": "sleeps"
        },
        {
            "adjective": "clever",
            "noun": "fox",
            "verb": "jumps"
        }
    ]
    
    variables_stage = VariablesStage(variable_sets)
    
    prompt_stage = PromptListOfAnswersStage(
        models=[model],
        prompts=["Is this sentence grammatically correct: 'The [adjective] [noun] [verb] quickly'?"],
        possible_answers=["Yes", "No"],
        result_var_name="is_grammatical"
    )
    
    export_stage = ExportToCsvStage(
        output_file="output.csv",
        columns=["adjective", "noun", "verb", "is_grammatical"]
    )
    
    pipeline = variables_stage | prompt_stage | export_stage
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 