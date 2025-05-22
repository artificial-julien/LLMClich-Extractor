from typing import List, Dict, Any, Optional
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
from src.commons import PipelineConfig
from .mixins.elo_calculator import EloCalculatorMixin
from .mixins.batch_generator import BatchGeneratorMixin
from .mixins.llm_processor import LLMProcessorMixin
from .mixins.batch_processor import BatchProcessorMixin
from .mixins.config_handler import ConfigHandlerMixin
from .types import EloCompetitorRating, DEFAULT_INITIAL_RATING, EloRound, EloMatch
from tqdm import tqdm

@StageRegistry.register("prompt_elo_rating")
class PromptEloRatingStage(
    Stage,
    EloCalculatorMixin,
    BatchGeneratorMixin,
    LLMProcessorMixin,
    BatchProcessorMixin,
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
        symmetric_matches: bool = False
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
    
    @classmethod
    def from_config(cls, stage_definition: Dict[str, Any]) -> 'PromptEloRatingStage':
        """
        Create a PromptEloRatingStage from configuration.
        
        Args:
            stage_definition: Dictionary containing stage configuration
            
        Returns:
            A PromptEloRatingStage instance
        """
        values = cls.get_config_values(stage_definition)
        return cls(**values)
    
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
                        competitors_ratings: Dict[str, EloCompetitorRating] = {
                            competitor: EloCompetitorRating(
                                model_config=model_config,
                                competitor=competitor,
                                rating=self.initial_rating
                            ) for competitor in self.competitors
                        }
                        model_result_executions: List[Execution] = []
                        
                        # Process all seeds for this model
                        for llm_seed in range(model_config.iterations):
                            # Process batches based on a counter instead of batches_per_model
                            num_batches = self.batches_per_model
                            
                            for batch_counter in range(num_batches):
                                # Generate and process batch
                                jobs: List[EloRound] = self.generate_match_batch(
                                    base_execution,
                                    competitors_ratings,
                                    model_config,
                                    self.prompts,
                                    llm_seed,
                                    self.symmetric_matches,
                                    batch_seed=pipeline_config.batch_seed
                                )
                                
                                # Process rounds (individual LLM calls)
                                round_results: List[EloRound] = self.process_batch(
                                    pipeline_config=pipeline_config,
                                    jobs=jobs,
                                    symmetric_matches=self.symmetric_matches,
                                    pbar=pbar
                                )
                                
                                matches: List[EloMatch] = self.group_rounds_into_matches(base_execution, round_results)
                                
                                valid_matches = []
                                for match in matches:
                                    if not match.winner and not match.is_draw:
                                        continue
                                        
                                    model_result_executions.append(match)
                                    valid_matches.append(match)
                                
                                if valid_matches:
                                    competitors_ratings = self.update_ratings_and_stats(
                                        valid_matches,
                                        competitors_ratings
                                    )
                        
                        sorted_ratings = sorted(
                            competitors_ratings.values(), 
                            key=lambda x: x.rating,
                            reverse=True
                        )
                        model_result_executions.extend(sorted_ratings)
                        all_result_executions.extend(model_result_executions)

        return all_result_executions 