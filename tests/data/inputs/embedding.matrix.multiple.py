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
        "Airplane"
    ]

    models_stage = EmbeddingModelsStage([model])
    embeddings_stage = EmbeddingStage(list)
    matrix_stage = MatrixOfDistancesStage('cosine') | MatrixOfDistancesStage('euclidean')
    export_stage = ExportMatrixToCsvStage(output_file_prefix="embedding.matrix")

    pipeline = models_stage | embeddings_stage | matrix_stage | export_stage
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 