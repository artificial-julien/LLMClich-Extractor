#!/usr/bin/env python

# Setup project path first, before any src imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.common.script_runner import common_setup, add_argument, load_arguments
from src.stages import *
from src.stages.embedding import EmbeddingExecution
from src.stages.embedding.embedding_models_stage import EmbeddingModelConfig
from src.common.types import *

def main():
    common_setup("Embedding Distance Matrix Pipeline Example")
    
    load_arguments()
    
    # Define items to compute embeddings for
    items = [
        "apple",
        "orange",
        "banana",
        "grape",
        "strawberry",
        "car",
        "truck",
        "bicycle",
        "motorcycle",
        "airplane",
        "dog",
        "cat",
        "elephant",
        "tiger",
        "bird"
    ]
    
    # Define embedding models to use
    embedding_models = [
        EmbeddingModelConfig(name="text-embedding-3-small")
    ]
    
    # Define distance metrics to compare
    distance_metrics = ["cosine", "euclidean", "manhattan"]
    
    # Process each distance metric
    for distance_metric in distance_metrics:
        print(f"\nProcessing with distance metric: {distance_metric}")
        
        # Create the pipeline stages
        embedding_models_stage = EmbeddingModelsStage(
            embedding_models=embedding_models
        )
        
        embedding_stage = EmbeddingStage(
            items=items
        )
        
        distances_stage = DistanceCalculationStage(
            distance_metric=distance_metric
        )

        pivot_table_stage = PivotStage(
            rows="_distance_item1",
            columns="_distance_item2",
            values="_distance_value")
        
        ranking_stage = RankingStage(
            direction="rows",
            sort_order="asc")
        
        export_stage = ExportMatrixToCsvStage(
            output_file_prefix=f"distance_matrix_{distance_metric}.csv"
        )
        
        # Also export the individual embeddings for reference
        embedding_export = ExportToCsvStage(
            output_file=f"embeddings_{distance_metric}",
            columns=[
                "_embedding_item",
                "_embedding_model_name", 
                "_embedding_model_dimensions"
            ],
            type_filter=[EmbeddingExecution]
        )
        
        # Create the complete pipeline
        pipeline = embedding_models_stage | embedding_stage | distances_stage | pivot_table_stage | ranking_stage | (export_stage & embedding_export)
        
        # Execute the pipeline
        results = pipeline.invoke()

if __name__ == "__main__":
    main() 