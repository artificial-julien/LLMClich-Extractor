from typing import List, Dict, Any, Optional
from ..types import Round, ModelConfig
from src.llm import LLMClient

class LLMProcessorMixin:
    """Mixin providing LLM interaction functionality."""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def format_prompt(self, template: str, competitor_a: str, competitor_b: str) -> str:
        """
        Format a prompt template with competitor variables.
        
        Args:
            template: Prompt template with [_elo_match_competitor_a] and [_elo_match_competitor_b] placeholders
            competitor_a: First competitor
            competitor_b: Second competitor
            
        Returns:
            Formatted prompt
        """
        formatted_prompt = template.replace("[_elo_match_competitor_a]", competitor_a).replace("[_elo_match_competitor_b]", competitor_b)
        
        # Add possible answers to the prompt
        formatted_prompt += f"\n\nPossible answers:\n1. {competitor_a}\n2. {competitor_b}"
        
        return formatted_prompt
    
    def process_round(
        self,
        competitor_a: str,
        competitor_b: str,
        model_config: ModelConfig,
        prompt_template: str,
        seed: int
    ) -> Round:
        """
        Process a single round (LLM call) between two competitors.
        
        Args:
            competitor_a: First competitor
            competitor_b: Second competitor
            model_config: Model configuration
            prompt_template: Prompt template
            seed: Seed value for reproducibility
            
        Returns:
            Round result from the LLM call
        """
        # Format the prompt with competitor names
        formatted_prompt = self.format_prompt(prompt_template, competitor_a, competitor_b)
        
        # Call the LLM with two possible answers (the two competitors)
        result = self.llm_client.generate_constrained_completion(
            model=model_config.name,
            prompt=formatted_prompt,
            possible_answers=[competitor_a, competitor_b],
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            seed=seed
        )
        
        # Create round result
        return Round(
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            winner=result['chosen_answer'] if not result['error'] else None,
            is_draw=False,  # Will be updated by batch processor if needed
            error=result['error'],
            model_name=model_config.name,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            seed=seed
        ) 