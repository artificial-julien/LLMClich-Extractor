from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    temperature: float
    top_p: float
    iterations: int

@dataclass
class MatchResult:
    competitor_a: str
    competitor_b: str
    winner: Optional[str]
    is_draw: bool
    error: Optional[str]
    model_name: str
    temperature: float
    top_p: float
    seed: int

class CompetitorStats(TypedDict):
    rating: float
    wins: int
    losses: int
    draws: int
    matches_played: int

class MatchJob(TypedDict):
    competitor_a: str
    competitor_b: str
    model_config: ModelConfig
    prompt_template: str
    seed: int

# Constants
DEFAULT_INITIAL_RATING = 1000
DEFAULT_K_FACTOR = 32
DEFAULT_PARALLEL_WORKERS = 2 