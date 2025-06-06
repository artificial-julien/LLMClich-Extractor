#!/usr/bin/env python3

# Setup project path first, before any src imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.script_runner import common_setup, add_argument, load_arguments
import src.common.script_runner as script_runner
from src.stages import *
from src.common.types import *

def main():
    common_setup("ELO rating pipeline for character funniness")
    
    load_arguments()
    
    model = EmbeddingModelConfig(
        name="text-embedding-3-small",
    )

    list = [
        "Banana",
        "Apple",
        "Orange",
        "Cat",
        "Dog",
        "Car",
        "Truck"
    ]

    list2 = [
        "Fruit",
        "Animal",
        "Vehicle",
        "Building"
    ]

    models_stage = EmbeddingModelsStage([model])
    embeddings_stage = EmbeddingStage(list)
    embeddings_stage2 = EmbeddingStage(list2, is_second_dimension=True)
    matrix_stage = MatrixOfDistancesStage('cosine')
    export_stage = ExportMatrixToCsvStage(output_file_prefix="embedding.matrix")

    pipeline = models_stage | embeddings_stage | embeddings_stage2 | matrix_stage | export_stage
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 