#!/usr/bin/env python3
"""
Complex pipeline demonstrating multiple stages with food classification and color analysis.
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
    common_setup("Food classification and color analysis pipeline")
    
    load_arguments()
    
    model_color = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        iterations=1
    )
    
    model_type = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.7,
        top_p=1.0,
        iterations=1
    )
    
    variable_sets = [
        {"food": "banana"},
        {"food": "orange"},
        {"food": "carrot"},
        {"food": "strawberry"}
    ]
    
    models_stage = ModelsStage([model_color])

    variables_stage = VariablesStage(variable_sets)
    
    color_stage = PromptListOfAnswersStage(
        prompts=["Of what color is [food]?"],
        possible_answers=["red", "yellow", "orange", "green", "brown", "purple", "pink"],
        result_var_name="food_color"
    )
    
    color_export = ExportToCsvStage(
        output_file_prefix="output.color.csv",
        columns=["food", "food_color"]
    )
    
    type_stage = PromptListOfAnswersStage(
        prompts=["Is [food] a fruit or a vegetable?"],
        possible_answers=["fruit", "vegetable"],
        result_var_name="food_type"
    )
    
    final_export = ExportToCsvStage(
        output_file_prefix="output.fruit_attributes.csv",
        columns=["food", "food_color", "food_type"]
    )
    
    pipeline = models_stage | variables_stage | color_stage | color_export | type_stage | final_export
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 