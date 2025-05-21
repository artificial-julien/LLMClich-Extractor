from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.execution import Execution
@dataclass
class ModelConfig:
    name: str
    temperature: float
    top_p: float
    iterations: int

@dataclass
class Job(ABC):
    """
    Abstract base class for all job types in the system.
    """
    execution: Execution
    model_config: ModelConfig

    def to_execution_variables(self) -> Dict[str, Any]:
        """Convert job properties to execution variables."""
        return {
            '_model_name': self.model_config.name,
            '_model_temperature': self.model_config.temperature,
            '_model_top_p': self.model_config.top_p,
            '_model_iterations': self.model_config.iterations,
            '_error': self.execution.error
        }

@dataclass
class EloRound(Job):
    """
    Represents a single LLM call between two competitors.
    """
    competitor_a: str
    competitor_b: str
    winner: Optional[str]
    prompt_template: str
    llm_seed: int

    def to_execution_variables(self) -> Dict[str, Any]:
        vars = super().to_execution_variables()
        vars.update({
            '_elo_round_competitor_a': self.competitor_a,
            '_elo_round_competitor_b': self.competitor_b,
            '_elo_round_winner': self.winner,
            '_elo_round_prompt_template': self.prompt_template,
            '_elo_round_llm_seed': self.llm_seed
        })
        return vars

@dataclass
class EloMatch(Job):
    """
    Represents an aggregation of multiple rounds between 2 competitors within a batch.
    """
    competitor_a: str
    competitor_b: str
    rounds: List[EloRound]
    winner: Optional[str]
    is_draw: bool
    
    @property
    def round_wins_a(self) -> int:
        return sum(1 for round in self.rounds if round.winner == self.competitor_a)
    
    @property
    def round_wins_b(self) -> int:
        return sum(1 for round in self.rounds if round.winner == self.competitor_b)

    def to_execution_variables(self) -> Dict[str, Any]:
        vars = super().to_execution_variables()
        vars.update({
            '_elo_match_competitor_a': self.competitor_a,
            '_elo_match_competitor_b': self.competitor_b,
            '_elo_match_winner': self.winner,
            '_elo_match_is_draw': self.is_draw,
            '_elo_match_round_wins_a': self.round_wins_a,
            '_elo_match_round_wins_b': self.round_wins_b
        })
        return vars

@dataclass
class EloCompetitorRating(Job):
    competitor: str
    rating: float
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def matches_played(self) -> int:
        return self.wins + self.losses + self.draws

    def to_execution_variables(self) -> Dict[str, Any]:
        vars = super().to_execution_variables()
        vars.update({
            '_elo_competitor': self.competitor,
            '_elo_rating': self.rating,
            '_elo_wins': self.wins,
            '_elo_losses': self.losses,
            '_elo_draws': self.draws
        })
        return vars

# Constants
DEFAULT_INITIAL_RATING = 1000
DEFAULT_K_FACTOR = 32 