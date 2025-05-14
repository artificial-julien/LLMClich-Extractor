from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    temperature: float
    top_p: float
    iterations: int

@dataclass
class Round:
    """
    Represents a single LLM call between two competitors.
    """
    competitor_a: str
    competitor_b: str
    winner: Optional[str]
    is_draw: bool
    error: Optional[str]
    model_name: str
    temperature: float
    top_p: float
    seed: int

@dataclass
class Match:
    """
    Represents an aggregation of multiple rounds between 2 competitors within a batch.
    """
    competitor_a: str
    competitor_b: str
    rounds: List[Round]
    winner: Optional[str]
    is_draw: bool
    
    @property
    def round_wins_a(self) -> int:
        return sum(1 for round in self.rounds if round.winner == self.competitor_a)
    
    @property
    def round_wins_b(self) -> int:
        return sum(1 for round in self.rounds if round.winner == self.competitor_b)
    
    @property
    def round_draws(self) -> int:
        return sum(1 for round in self.rounds if round.is_draw)

class CompetitorStats(TypedDict):
    rating: float
    wins: int
    losses: int
    draws: int
    matches_played: int

class RoundJob(TypedDict):
    competitor_a: str
    competitor_b: str
    model_config: ModelConfig
    prompt_template: str
    seed: int

# Maintain backward compatibility
# TODO: Can be removed after refactoring all code to use the new types
MatchJob = RoundJob

# Constants
DEFAULT_INITIAL_RATING = 1000
DEFAULT_K_FACTOR = 32
DEFAULT_PARALLEL_WORKERS = 2 