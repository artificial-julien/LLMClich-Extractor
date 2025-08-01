from typing import List, Dict, Any, Optional, Iterator
from src.stage import Stage
from src.execution import Execution, ModelConfig
from src.common.types import *
from .mixins.elo_calculator import EloCalculatorMixin
from .mixins.batch_generator import BatchGeneratorMixin
from .mixins.pair_selector import PairSelectorMixin
from .mixins.llm_processor import LLMProcessorMixin
from .mixins.batch_processor import BatchProcessorMixin
from .types import EloCompetitorRating, DEFAULT_INITIAL_RATING, EloRound, EloMatch, PairSelectionFactors
from tqdm import tqdm
import random

class PromptEloRatingStage(
    Stage,
    EloCalculatorMixin,
    BatchGeneratorMixin,
    PairSelectorMixin,
    LLMProcessorMixin,
    BatchProcessorMixin
):
    """
    Stage that implements Elo rating calculations for LLM-judged competitions.
    Uses mixins to separate different responsibilities.
    """
    
    def __init__(
        self,
        competitors: List[str],
        prompts: List[str],
        batches_per_model: int = 4,
        initial_rating: int = DEFAULT_INITIAL_RATING,
        symmetric_matches: bool = False,
        use_round_proportions: bool = False,
        pair_selection_factors: Optional[PairSelectionFactors] = None
    ):
        """
        Initialize the prompt Elo rating stage.
        
        Args:
            competitors: List of competitors to be ranked
            prompts: List of prompt templates
            batches_per_model: Number of batches to process for each model
            initial_rating: Initial Elo rating for all competitors
            symmetric_matches: Whether to run matches in both directions
            use_round_proportions: Whether to use the proportion of rounds won as the score instead of binary win/loss
            pair_selection_factors: Factors for scoring-based pair selection (if None, uses defaults)
        """
        # Initialize mixins
        LLMProcessorMixin.__init__(self)
        BatchProcessorMixin.__init__(self)
        PairSelectorMixin.__init__(self)
        
        # Store configuration
        self.competitors = competitors
        self.prompts = prompts
        self.batches_per_model = batches_per_model
        self.initial_rating = initial_rating
        self.symmetric_matches = symmetric_matches
        self.use_round_proportions = use_round_proportions
        
        self.pair_selection_factors = pair_selection_factors
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """ 
        Process input executions lazily, run Elo rating matches, and yield results.
        
        Args:
            pipeline_config: Pipeline configuration
            executions: Iterator of input executions to process
            
        Yields:
            Execution instances with results for each competitor and match
        """
        # Process each input execution separately
        for base_execution in executions:
            if base_execution.has_error():
                yield base_execution
                continue
                
            if not base_execution.model_config:
                error_execution = base_execution.copy()
                error_execution.set_error("No model configuration found in execution")
                yield error_execution
                continue
                
            model_config = base_execution.model_config
            
            # Initialize RNG for this execution
            rng = random.Random(pipeline_config.batch_seed)
            
            # Calculate total number of rounds for progress bar
            # For each llm_seed and batch:
            # - Number of matches = len(competitors) // 2
            # - Each match has len(prompts) rounds
            # - If symmetric_matches is True, double the number of rounds
            matches_per_batch = len(self.competitors) // 2
            rounds_per_match = len(self.prompts) * (2 if self.symmetric_matches else 1)
            rounds_per_batch = matches_per_batch * rounds_per_match
            total_rounds = model_config.iterations * self.batches_per_model * rounds_per_batch
            
            # Create a single progress bar for the entire process
            with tqdm(total=total_rounds, desc=f"Processing Elo rating rounds") as pbar:
                competitors_ratings: Dict[str, EloCompetitorRating] = {}
                for competitor in self.competitors:
                    rating = EloCompetitorRating(
                        model_config=model_config,
                        competitor=competitor,
                        rating=self.initial_rating
                    )
                    rating.import_variables_from(base_execution)
                    competitors_ratings[competitor] = rating
                
                # Process all seeds for this model
                for batch_counter in range(self.batches_per_model):
                    # Generate and process batch
                    matches: List[EloMatch] = []
                    
                    # Collect all matches for all seeds in this batch
                    for llm_seed in range(model_config.iterations):
                        if "grok" in model_config.name:
                            llm_seed = None
                        
                        batch_matches = self.generate_match_batch_with_scoring(
                            base_execution,
                            competitors_ratings,
                            model_config,
                            self.prompts,
                            llm_seed,
                            self.symmetric_matches,
                            self.pair_selection_factors or PairSelectionFactors(),
                            rng=rng
                        )
                        matches.extend(batch_matches)
                    
                    # Process all rounds within all matches in this batch
                    processed_matches: List[EloMatch] = self.process_batch(
                        pipeline_config=pipeline_config,
                        matches=matches,
                        pbar=pbar
                    )
                    
                    valid_matches = []
                    for match in processed_matches:
                        if not match.winner and not match.is_draw:
                            raise ValueError(f"Match between {match.competitor_a} and {match.competitor_b} has no winner and is not a draw")
                            
                        yield match
                        valid_matches.append(match)
                    
                    if valid_matches:
                        competitors_ratings = self.update_ratings_and_stats(
                            valid_matches,
                            competitors_ratings,
                            self.use_round_proportions
                        )
                
                sorted_ratings = sorted(
                    competitors_ratings.values(), 
                    key=lambda x: x.rating,
                    reverse=True
                )
                # Assign ranks (1-based, where 1 is highest rating)
                for i, rating in enumerate(sorted_ratings):
                    rating.rank = i + 1
                    yield rating
                    