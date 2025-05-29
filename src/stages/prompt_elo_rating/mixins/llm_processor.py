from ..types import EloRound
from src.llm import LLMClient
from src.llm.prompt_utils import format_template
from src.common.types import *
from src.prompt_exception import LLMException

class LLMProcessorMixin:
    """Mixin providing LLM interaction functionality."""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def process_round(
        self,
        round: EloRound,
        pipeline_config: PipelineConfig
    ) -> EloRound:
        """
        Process a single round (LLM call) between two competitors.
        
        Args:
            round: EloRound object
            pipeline_config: Pipeline configuration
            
        Returns:
            EloRound result from the LLM call
        """
        formatted_prompt = format_template(round.prompt_template, round.get_all_variables())
        
        competitor_a = round.competitor_a
        competitor_b = round.competitor_b
        
        try:
            # Call the LLM with two possible answers (the two competitors)
            result = self.llm_client.generate_constrained_completion(
                model=round.model_config.name,
                prompt=formatted_prompt,
                possible_answers=[competitor_a, competitor_b],
                temperature=round.model_config.temperature,
                top_p=round.model_config.top_p,
                llm_seed=round.llm_seed,
                max_tries=pipeline_config.llm_max_tries
            )
        except LLMException as e:
            round.error = e.message
            return round
        
        round.winner = result.chosen_answer
        return round