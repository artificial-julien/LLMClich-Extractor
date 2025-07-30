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
    common_setup("ELO rating pipeline for funniest fictional characters")
    
    load_arguments()

    competitors = [
        "Bugs Bunny",
        "Wile E. Coyote",
        "Donald Duck",
        "SpongeBob SquarePants", 
        "Patrick Star",
        "Squidward Tentacles", 
        "Plankton",
        "Homer Simpson",
        "Mr. Burns",
        "Eric Cartman", 
        "Kenny McCormick",
        "Randy Marsh",
        "Peter Griffin",
        "Stewie Griffin", 
        "Glenn Quagmire",
        "Bender from Futurama",
        "Courage the Cowardly Dog",
        "Stimpy"
    ]
    
    # All the following prompts will be run to determine to the winner by score
    prompts = [
        "Who is funnier between '[_elo_match_competitor_a]' and '[_elo_match_competitor_b]'?",
        "Between '[_elo_match_competitor_a]' and '[_elo_match_competitor_b]', which character makes people laugh more?",
        "Who has better comedic value - '[_elo_match_competitor_a]' or '[_elo_match_competitor_b]'?",
        "If you had to pick the more humorous character between '[_elo_match_competitor_a]' and '[_elo_match_competitor_b]', who would it be?",
        "Which character delivers more laughs: '[_elo_match_competitor_a]' or '[_elo_match_competitor_b]'?"
    ]

    # Will repeat the entire process once for 
    models_stage = ModelsStage([
        ModelConfig(
        name="gpt-4o-mini",
        temperature=0.0,
        top_p=1.0,
        iterations=1),
        ModelConfig(
        name="gpt-4.1-mini",
        temperature=0.0,
        top_p=1.0,
        iterations=1)
    ])
    
    elo_stage = PromptEloRatingStage(
        competitors=competitors,
        prompts=prompts,
        batches_per_model=16,
        initial_rating=1000,
        symmetric_matches=True
    )
    
    ranking_export = ExportToCsvStage(
        output_file="funniest_fictional_char/results.elo.ranking.csv",
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
        output_file="funniest_fictional_char/results.elo.matches.csv",
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