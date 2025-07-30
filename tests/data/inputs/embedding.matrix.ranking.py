#!/usr/bin/env python

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
    embedding_stage = EmbeddingStage(list)
    distances_stage = DistanceCalculationStage('cosine')
    pivot_table_stage = PivotStage(rows="_distance_item1", columns="_distance_item2", values="_distance_value")
    ranking_stage = RankingStage(direction="rows", sort_order="asc")
    export_stage = ExportMatrixToCsvStage(output_file_prefix="embedding.matrix")

    pipeline = models_stage | embedding_stage | distances_stage | pivot_table_stage | ranking_stage

    results = pipeline.invoke()

if __name__ == "__main__":
    main()