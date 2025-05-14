from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    temperature: float
    top_p: float
    iterations: int
    
    @classmethod
    def create(cls, name: str, temperature: float = 0.0, top_p: float = 1.0, iterations: int = 1) -> 'ModelConfig':
        """Create a ModelConfig instance with default values."""
        return cls(
            name=name,
            temperature=temperature,
            top_p=top_p,
            iterations=iterations
        )

@dataclass
class Round:
    """
    Represents a single LLM call between two competitors.
    """
    competitor_a: str
    competitor_b: str
    winner: Optional[str]
    error: Optional[str]
    model_name: str
    temperature: float
    top_p: float
    seed: int
    
    @classmethod
    def create(cls, competitor_a: str, competitor_b: str, model_name: str, seed: int, 
              temperature: float = 0.0, top_p: float = 1.0, winner: Optional[str] = None, 
              error: Optional[str] = None) -> 'Round':
        """Create a Round instance with default values."""
        return cls(
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            winner=winner,
            error=error,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed
        )

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
    
    @classmethod
    def create(cls, competitor_a: str, competitor_b: str, rounds: List[Round], 
              winner: Optional[str] = None, is_draw: bool = False) -> 'Match':
        """Create a Match instance with default values."""
        return cls(
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            rounds=rounds,
            winner=winner,
            is_draw=is_draw
        )

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