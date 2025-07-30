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
from src.stages.prompt_elo_rating.base import EloMatch, EloCompetitorRating

def main():
    common_setup("ELO rating pipeline for character funniness")
    
    load_arguments()
    
    model = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        iterations=1
    )

    competitors = [
        "The sun",
        "The moon",
        "The earth",
        "Sagittarius A*",
        "Jupiter",
        "Mars",
        "The Milky Way galaxy",
    ]
    
    prompts = [
        "What is the heaviest object between '[_elo_match_competitor_a]' and '[_elo_match_competitor_b]'?",
        "Between '[_elo_match_competitor_a]' and '[_elo_match_competitor_b]', which one would weigh more?"
    ]
    
    models_stage = ModelsStage([model])
    
    elo_stage = PromptEloRatingStage(
        competitors=competitors,
        prompts=prompts,
        batches_per_model=8,
        initial_rating=1000,
        symmetric_matches=True
    )
    
    ranking_export = ExportToCsvStage(
        output_file="elo.ranking.csv",
        type_filter=[EloCompetitorRating],
        columns=[
            "_model_name", "_model_temperature", "_model_top_p",
            "_elo_competitor", "_elo_rank"
        ]
    )
    
    pipeline = models_stage | elo_stage | ranking_export
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main()