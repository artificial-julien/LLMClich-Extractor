from typing import List, Dict, Any, Optional
from src.stage import Stage
from src.execution import Execution, ModelConfig
from src.common.types import *
from .mixins.elo_calculator import EloCalculatorMixin
from .mixins.batch_generator import BatchGeneratorMixin
from .mixins.llm_processor import LLMProcessorMixin
from .mixins.batch_processor import BatchProcessorMixin
from .types import EloCompetitorRating, DEFAULT_INITIAL_RATING, EloRound, EloMatch
from tqdm import tqdm

class PromptEloRatingStage(
    Stage,
    EloCalculatorMixin,
    BatchGeneratorMixin,
    LLMProcessorMixin,
    BatchProcessorMixin
):
    """
    Stage that implements Elo rating calculations for LLM-judged competitions.
    Uses mixins to separate different responsibilities.
    """
    
    def __init__(
        self,
        models: List[ModelConfig],
        competitors: List[str],
        prompts: List[str],
        batches_per_model: int = 4,
        initial_rating: int = DEFAULT_INITIAL_RATING,
        symmetric_matches: bool = False,
        use_round_proportions: bool = False
    ):
        """
        Initialize the prompt Elo rating stage.
        
        Args:
            models: List of ModelConfig instances
            competitors: List of competitors to be ranked
            prompts: List of prompt templates
            batches_per_model: Number of batches to process for each model
            initial_rating: Initial Elo rating for all competitors
            symmetric_matches: Whether to run matches in both directions
            use_round_proportions: Whether to use the proportion of rounds won as the score instead of binary win/loss
        """
        # Initialize mixins
        LLMProcessorMixin.__init__(self)
        BatchProcessorMixin.__init__(self)
        
        # Store configuration
        self.models = models
        self.competitors = competitors
        self.prompts = prompts
        self.batches_per_model = batches_per_model
        self.initial_rating = initial_rating
        self.symmetric_matches = symmetric_matches
        self.use_round_proportions = use_round_proportions
    

    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """ 
        Process input executions, run Elo rating matches, and return results.
        
        Args:
            pipeline_config: Pipeline configuration
            executions: List of input executions to process
            
        Returns:
            List of new executions with results for each competitor
        """
        all_result_executions: List[Execution] = []
        
        # Process each input execution separately
        for i, base_execution in enumerate(executions):
            # Calculate total number of rounds for progress bar
            # For each model, llm_seed, and batch:
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
                with tqdm(total=total_rounds, desc=f"Processing Elo rating rounds for execution {i+1}/{len(executions)}") as pbar:
                    for model_config in self.models:
                        competitors_ratings: Dict[str, EloCompetitorRating] = {}
                        for competitor in self.competitors:
                            rating = EloCompetitorRating(
                                model_config=model_config,
                                competitor=competitor,
                                rating=self.initial_rating
                            )
                            rating.import_variables_from(base_execution)
                            competitors_ratings[competitor] = rating
                        model_result_executions: List[Execution] = []
                        
                        # Process all seeds for this model
                        for batch_counter in range(self.batches_per_model):
                            # Generate and process batch
                            jobs: List[EloRound] = []
                            
                            # Collect all rounds for all seeds in this batch
                            for llm_seed in range(model_config.iterations):
                                batch_jobs = self.generate_match_batch(
                                    base_execution,
                                    competitors_ratings,
                                    model_config,
                                    self.prompts,
                                    llm_seed,
                                    self.symmetric_matches,
                                    batch_seed=pipeline_config.batch_seed
                                )
                                jobs.extend(batch_jobs)
                            
                            # Process all rounds for all seeds in this batch
                            round_results: List[EloRound] = self.process_batch(
                                pipeline_config=pipeline_config,
                                jobs=jobs,
                                pbar=pbar
                            )
                            
                            matches: List[EloMatch] = self.group_rounds_into_matches(base_execution, round_results)
                            
                            valid_matches = []
                            for match in matches:
                                if not match.winner and not match.is_draw:
                                    raise ValueError(f"Match between {match.competitor_a} and {match.competitor_b} has no winner and is not a draw")
                                    
                                model_result_executions.append(match)
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
                        model_result_executions.extend(sorted_ratings)
                        all_result_executions.extend(model_result_executions)

        return all_result_executions 