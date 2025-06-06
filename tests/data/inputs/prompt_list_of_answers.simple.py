#!/usr/bin/env python3
"""
Simple prompt list of answers pipeline demonstrating scientific accuracy evaluation.
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
    common_setup("Scientific accuracy evaluation pipeline")
    
    load_arguments()
    
    model = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        iterations=1
    )
    
    variable_sets = [
        {"statement": "The Earth is flat."},
        {"statement": "The Sun revolves around the Earth."}
    ]
    
    variables_stage = VariablesStage(variable_sets)
    
    models_stage = ModelsStage([model])
    
    prompt_stage = PromptListOfAnswersStage(
        prompts=["Is this statement scientifically accurate: '[statement]'"],
        possible_answers=["True", "False"],
        result_var_name="accuracy"
    )
    
    export_stage = ExportToCsvStage(
        output_file_prefix="output.csv",
        columns=["statement", "accuracy", "_model_name", "_model_temperature"]
    )
    
    pipeline = variables_stage | models_stage | prompt_stage | export_stage
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 