from typing import List, Dict, Any, Optional
import random
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
from .mixins.elo_calculator import EloCalculatorMixin
from .mixins.batch_generator import BatchGeneratorMixin
from .mixins.llm_processor import LLMProcessorMixin
from .mixins.batch_processor import BatchProcessorMixin
from .mixins.execution_manager import ExecutionManagerMixin
from .mixins.config_handler import ConfigHandlerMixin
from .types import CompetitorStats, DEFAULT_INITIAL_RATING, Round, Match
from tqdm import tqdm

@StageRegistry.register("prompt_elo_rating")
class PromptEloRatingStage(
    Stage,
    EloCalculatorMixin,
    BatchGeneratorMixin,
    LLMProcessorMixin,
    BatchProcessorMixin,
    ExecutionManagerMixin,
    ConfigHandlerMixin
):
    """
    Stage that implements Elo rating calculations for LLM-judged competitions.
    Uses mixins to separate different responsibilities.
    """
    
    def __init__(
        self,
        models: List[Dict[str, Any]],
        competitors: List[str],
        prompts: List[str],
        batches_per_model: int = 4,
        initial_rating: int = DEFAULT_INITIAL_RATING,
        symmetric_matches: bool = False,
        parallel: int = 2,
        seed: Optional[int] = None
    ):
        """
        Initialize the prompt Elo rating stage.
        
        Args:
            models: List of model configurations
            competitors: List of competitors to be ranked
            prompts: List of prompt templates
            batches_per_model: Number of batches to process for each model
            initial_rating: Initial Elo rating for all competitors
            symmetric_matches: Whether to run matches in both directions
            parallel: Number of parallel requests
            seed: Optional seed for deterministic batch generation
        """
        # Initialize mixins
        LLMProcessorMixin.__init__(self)
        BatchProcessorMixin.__init__(self, parallel)
        
        # Store configuration
        self.models = models
        self.competitors = competitors
        self.prompts = prompts
        self.batches_per_model = batches_per_model
        self.initial_rating = initial_rating
        self.symmetric_matches = symmetric_matches
        self.seed = seed
        
        # Seed the random number generator if a seed is provided
        if seed is not None:
            random.seed(seed)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PromptEloRatingStage':
        """
        Create a PromptEloRatingStage from configuration.
        
        Args:
            config: Dictionary containing stage configuration
            
        Returns:
            A PromptEloRatingStage instance
        """
        values = cls.get_config_values(config)
        return cls(**values)
    
    def _initialize_stats(self) -> Dict[str, CompetitorStats]:
        """
        Initialize competitor statistics.
        
        Returns:
            Dictionary of competitor statistics
        """
        return {
            competitor: {
                'rating': self.initial_rating,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'matches_played': 0
            }
            for competitor in self.competitors
        }
    
    def process(self, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions, run Elo rating matches, and return results.
        
        Args:
            executions: List of input executions (typically just one)
            
        Returns:
            List of new executions with results for each competitor
        """
        base_execution = executions[0] if executions else Execution()
        all_result_executions = []

        # Calculate total number of rounds for progress bar
        # For each model, seed, and batch:
        # - Number of matches = len(competitors) // 2
        # - Each match has len(prompts) rounds
        # - If symmetric_matches is True, double the number of rounds
        total_rounds = 0
        for model_config in self.models:
            matches_per_batch = len(self.competitors) // 2
            rounds_per_match = len(self.prompts) * (2 if self.symmetric_matches else 1)
            rounds_per_batch = matches_per_batch * rounds_per_match
            total_rounds += model_config.iterations * self.batches_per_model * rounds_per_batch
        
        # Create a single progress bar for the entire process
        with tqdm(total=total_rounds, desc="Processing Elo rating rounds") as pbar:
            for model_config in self.models:
                # Initialize tracking for this model (single stats across all seeds)
                stats = self._initialize_stats()
                model_result_executions = []
                
                # Process all seeds for this model
                for seed in range(model_config.iterations):
                    # Process batches based on a counter instead of batches_per_model
                    num_batches = self.batches_per_model
                    
                    for batch_counter in range(num_batches):
                        # Generate and process batch
                        jobs = self.generate_match_batch(
                            self.competitors,
                            stats,
                            model_config,
                            self.prompts,
                            seed,
                            self.symmetric_matches
                        )
                        
                        # Process rounds (individual LLM calls)
                        round_results = self.process_batch(jobs, self.symmetric_matches, pbar)
                        
                        # Group rounds into matches
                        matches = self.group_rounds_into_matches(round_results)
                        
                        # Create executions for match results and update ratings
                        for match in matches:
                            if not match.winner and not match.is_draw:
                                continue
                                
                            match_execution = self.create_match_execution(base_execution, match)
                            model_result_executions.append(match_execution)
                            
                            # Update ratings and stats
                            a_score = 1.0 if match.winner == match.competitor_a else (0.5 if match.is_draw else 0.0)
                            b_score = 1.0 if match.winner == match.competitor_b else (0.5 if match.is_draw else 0.0)
                            self.update_ratings_and_stats(
                                match.competitor_a,
                                match.competitor_b,
                                a_score,
                                b_score,
                                stats
                            )
                
                # Add final ratings for this model (after all seeds)
                for competitor, competitor_stats in stats.items():
                    rating_execution = self.create_rating_execution(
                        base_execution,
                        competitor,
                        competitor_stats['rating'],
                        competitor_stats['wins'],
                        competitor_stats['losses'],
                        competitor_stats['draws'],
                        model_config,
                        -1  # Use -1 to indicate this is the final rating across all seeds
                    )
                    model_result_executions.append(rating_execution)
                
                all_result_executions.extend(model_result_executions)

        return all_result_executions 