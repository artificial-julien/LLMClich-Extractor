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
    embeddings_stage = EmbeddingStage(list)
    distances_stages = DistanceCalculationStage('cosine')
    pivot_table_stage = PivotStage(rows="_distance_item1", columns="_distance_item2", values="_distance_value")
    export_stage = ExportMatrixToCsvStage(output_file_prefix="embedding.distance.matrix")

    group_by_stage = GroupByStage(
        key="_distance_metric",  # Group by the embedding model
        pipeline=pivot_table_stage | export_stage
    )
    
    pipeline = models_stage | embeddings_stage | distances_stages | group_by_stage

    results = pipeline.invoke()

if __name__ == "__main__":
    main()