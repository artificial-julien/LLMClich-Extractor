from typing import List, Dict, Any, Optional
from ..types import Round, ModelConfig
from src.llm import LLMClient
from src.llm.prompt_utils import format_template_variables, add_possible_answers
from src.execution import Execution

class LLMProcessorMixin:
    """Mixin providing LLM interaction functionality."""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def format_prompt(self, template: str, execution: Execution) -> str:
        """
        Format a prompt template with variables from execution.
        
        Args:
            template: Prompt template with [variable] placeholders
            execution: Execution object containing variables
            
        Returns:
            Formatted prompt
        """
        # Format the template with all variables from execution
        formatted_prompt = format_template_variables(template, execution.variables)
        
        # Add possible answers using competitor variables
        possible_answers = [execution.variables.get('_elo_match_competitor_a'), execution.variables.get('_elo_match_competitor_b')]
        return add_possible_answers(formatted_prompt, possible_answers)
    
    def process_round(
        self,
        execution: Execution,
        model_config: ModelConfig,
        prompt_template: str,
        llm_seed: int
    ) -> Round:
        """
        Process a single round (LLM call) between two competitors.
        
        Args:
            execution: Execution object containing variables
            model_config: Model configuration
            prompt_template: Prompt template
            llm_seed: llm_seed value for reproducibility
            
        Returns:
            Round result from the LLM call
        """
        # Format the prompt with variables from execution
        formatted_prompt = self.format_prompt(prompt_template, execution)
        
        # Get competitors from execution variables
        competitor_a = execution.variables.get('_elo_match_competitor_a')
        competitor_b = execution.variables.get('_elo_match_competitor_b')
        
        # Call the LLM with two possible answers (the two competitors)
        result = self.llm_client.generate_constrained_completion(
            model=model_config.name,
            prompt=formatted_prompt,
            possible_answers=[competitor_a, competitor_b],
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            llm_seed=llm_seed
        )
        
        # Create round result
        return Round(
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            winner=result['chosen_answer'] if not result['error'] else None,
            error=result['error'],
            model_name=model_config.name,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            llm_seed=llm_seed
        ) 