#!/usr/bin/env python3
"""
Pipeline demonstrating multiple variable nodes with string comparison.
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
    common_setup("String comparison pipeline with multiple variable nodes")
    
    load_arguments()
    
    models = [
        ModelConfig(
            name="gpt-4.1-nano",
            temperature=1.0,
            top_p=1.0,
            iterations=1
        ),
        ModelConfig(
            name="gpt-4o-mini",
            temperature=1.0,
            top_p=1.0,
            iterations=1
        )
    ]
    
    variable_sets_a = [
        {"a": "beta"},
        {"a": "alpha"}
    ]
    
    variable_sets_b = [
        {"b": "alpha"},
        {"b": "beta"}
    ]

    models_stage = ModelsStage(models)
    
    variables_stage_a = VariablesStage(variable_sets_a)
    variables_stage_b = VariablesStage(variable_sets_b)
    
    prompt_stage = PromptListOfAnswersStage(
        prompts=["does [a] equals [b]?"],
        possible_answers=["true", "false"],
        result_var_name="result"
    )
    
    export_stage = ExportToCsvStage(
        output_file="output.csv",
        columns=["a", "b", "result"]
    )
    
    pipeline = models_stage | variables_stage_a | variables_stage_b | prompt_stage | export_stage
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 