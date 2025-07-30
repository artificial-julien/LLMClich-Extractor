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
from src.stages.prompt_elo_rating.base import EloMatch, EloCompetitorRating


def main():
    common_setup("ELO rating pipeline for famous personalities intelligence")
    
    load_arguments()

    competitors = [
        "Albert Einstein",
        "Abraham Lincoln",
        "Barack Obama",
        "Bill Gates",
        "Charles Darwin",
        "Isaac Newton",
        "Stephen Hawking",
        "Marie Curie",
        "Nikola Tesla",
        "Richard Feynman",
        "Donald Trump",
        "Kim Kardashian",
        "Paris Hilton",
        "Justin Bieber",
        "Kanye West",
        "Britney Spears",
        "Logan Paul",
        "Kylie Jenner",
        "Snoop Dogg",
        "Vladimir Putin",
        "Elon Musk",
        "George W. Bush",
        "Emmanuel Macron",
        "Xi Jinping",
        "Joe Biden"
    ]
    
    prompts = [
        "Who of [_elo_match_competitor_a] and [_elo_match_competitor_b] is the smartest?"
    ]

    models_stage = ModelsStage([
        ModelConfig(
            name="gpt-4o",
            temperature=1.0,
            top_p=1.0,
            iterations=2
        ),
        ModelConfig(
            name="gpt-4.1",
            temperature=1.0,
            top_p=1.0,
            iterations=2
        )
    ])
    
    elo_stage = PromptEloRatingStage(
        competitors=competitors,
        prompts=prompts,
        batches_per_model=8,
        initial_rating=1000,
        symmetric_matches=True
    )
    
    ranking_export = ExportToCsvStage(
        output_file="famous_personalities_intelligence/results.elo.ranking.csv",
        skip_non_full_rows=True,
        columns=[
            "_model_name",
            "_model_temperature",
            "_model_top_p",
            "_elo_competitor",
            "_elo_rating",
            "_elo_wins",
            "_elo_loss",
            "_elo_draws"
        ]
    )
    
    matches_export = ExportToCsvStage(
        output_file="famous_personalities_intelligence/results.elo.matches.csv",
        skip_non_full_rows=True,
        columns=[
            "_model_name",
            "_model_temperature",
            "_model_top_p",
            "_elo_match_competitor_a",
            "_elo_match_competitor_b",
            "_elo_match_winner",
            "_elo_match_draw",
            "_elo_match_wins_a",
            "_elo_match_wins_b"
        ]
    )
    
    pipeline = models_stage | elo_stage | (ranking_export & matches_export)
    
    results = pipeline.invoke()

if __name__ == "__main__":
    main() 