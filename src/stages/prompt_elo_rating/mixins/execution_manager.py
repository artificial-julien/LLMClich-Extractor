from typing import List, Dict, Any
from src.execution import Execution
from ..types import MatchResult, ModelConfig

class ExecutionManagerMixin:
    """Mixin providing execution management functionality."""
    
    def create_base_execution_vars(
        self,
        execution: Execution,
        model_name: str,
        temperature: float,
        top_p: float,
        seed: int
    ) -> None:
        """
        Add common model-related variables to an execution.
        
        Args:
            execution: Execution to add variables to
            model_name: Name of the model
            temperature: Model temperature
            top_p: Model top_p
            seed: Seed value
        """
        execution.add_variable('_seed', seed)
        execution.add_variable('_model_name', model_name)
        execution.add_variable('_model_temperature', temperature)
        execution.add_variable('_model_top_p', top_p)
    
    def create_match_execution(
        self,
        base_execution: Execution,
        match_result: MatchResult
    ) -> Execution:
        """
        Create an execution for a match result.
        
        Args:
            base_execution: Base execution to copy from
            match_result: Match result to create execution for
            
        Returns:
            New execution with match result variables
        """
        match_execution = base_execution.copy()
        match_execution.add_variable('_elo_match_competitor_a', match_result.competitor_a)
        match_execution.add_variable('_elo_match_competitor_b', match_result.competitor_b)
        match_execution.add_variable('_elo_match_winner', None if match_result.is_draw else match_result.winner)
        match_execution.add_variable('_elo_match_draw', match_result.is_draw)
        
        self.create_base_execution_vars(
            match_execution,
            match_result.model_name,
            match_result.temperature,
            match_result.top_p,
            match_result.seed
        )
        
        return match_execution
    
    def create_rating_execution(
        self,
        base_execution: Execution,
        competitor: str,
        rating: float,
        wins: int,
        losses: int,
        draws: int,
        model_config: ModelConfig,
        seed: int
    ) -> Execution:
        """
        Create an execution for a competitor's final rating.
        
        Args:
            base_execution: Base execution to copy from
            competitor: Competitor name
            rating: Final Elo rating
            wins: Number of wins
            losses: Number of losses
            draws: Number of draws
            model_config: Model configuration
            seed: Seed value
            
        Returns:
            New execution with rating variables
        """
        rating_execution = base_execution.copy()
        rating_execution.add_variable('_elo_competitor', competitor)
        rating_execution.add_variable('_elo_rating', int(rating))
        rating_execution.add_variable('_elo_wins', wins)
        rating_execution.add_variable('_elo_loss', losses)
        rating_execution.add_variable('_elo_draws', draws)
        
        self.create_base_execution_vars(
            rating_execution,
            model_config.name,
            model_config.temperature,
            model_config.top_p,
            seed
        )
        
        return rating_execution 