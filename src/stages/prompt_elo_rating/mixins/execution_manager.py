from typing import Dict, Any, Optional, List
from src.execution import Execution
from ..types import Round, Match

class ExecutionManagerMixin:
    """Mixin providing execution management functionality."""
    
    def create_base_execution_vars(
        self,
        execution: Execution,
        model_name: str,
        temperature: float,
        top_p: float,
        llm_seed: int
    ) -> None:
        """
        Create base execution variables.
        
        Args:
            execution: Execution to update
            model_name: Model name
            temperature: Temperature value
            top_p: Top-p value
            llm_seed: llm_seed value
        """
        execution.add_variable('_model_name', model_name)
        execution.add_variable('_model_temperature', temperature)
        execution.add_variable('_model_top_p', top_p)
        execution.add_variable('_seed', llm_seed)
    
    def create_round_execution(
        self,
        base_execution: Execution,
        round_result: Round
    ) -> Execution:
        """
        Create an execution for a round result.
        
        Args:
            base_execution: Base execution to copy from
            round_result: Round result to create execution for
            
        Returns:
            New execution with round result variables
        """
        round_execution = base_execution.copy()
        round_execution.add_variable('_elo_match_competitor_a', round_result.competitor_a)
        round_execution.add_variable('_elo_match_competitor_b', round_result.competitor_b)
        round_execution.add_variable('_elo_match_winner', round_result.winner)
        
        self.create_base_execution_vars(
            round_execution,
            round_result.model_name,
            round_result.temperature,
            round_result.top_p,
            round_result.llm_seed
        )
        
        return round_execution
    
    def create_match_execution(
        self,
        base_execution: Execution,
        match: Match
    ) -> Execution:
        """
        Create an execution for a match result (aggregation of rounds).
        
        Args:
            base_execution: Base execution to copy from
            match: Match result to create execution for
            
        Returns:
            New execution with match result variables
        """
        match_execution = base_execution.copy()
        match_execution.add_variable('_elo_match_competitor_a', match.competitor_a)
        match_execution.add_variable('_elo_match_competitor_b', match.competitor_b)
        match_execution.add_variable('_elo_match_winner', None if match.is_draw else match.winner)
        match_execution.add_variable('_elo_match_draw', match.is_draw)
        match_execution.add_variable('_elo_match_wins_a', int(match.round_wins_a))
        match_execution.add_variable('_elo_match_wins_b', int(match.round_wins_b))
        
        # Use the first round for model information
        if match.rounds:
            first_round = match.rounds[0]
            self.create_base_execution_vars(
                match_execution,
                first_round.model_name,
                first_round.temperature,
                first_round.top_p,
                first_round.llm_seed
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
        model_config: Any,
        llm_seed: int
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
            llm_seed: llm seed value
            
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
            llm_seed
        )
        
        return rating_execution 