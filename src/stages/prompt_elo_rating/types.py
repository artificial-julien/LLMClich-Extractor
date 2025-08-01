from typing import TypedDict, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from src.execution import Execution, ModelConfig

@dataclass
class PairSelectionFactors:
    uncertainty_factor: float = 1.0
    pair_repetition_count_factor: float = 1.0
    rating_difference_penalty_factor: float = 1.0

@dataclass
class PairScore:
    score: float
    competitor_a: str
    competitor_b: str

@dataclass(kw_only=True)
class EloRound(Execution):
    """
    Represents a single LLM call between two competitors.
    """
    competitor_a: str
    competitor_b: str
    winner: Optional[str]
    prompt_template: str
    llm_seed: int
    is_mirror: bool

    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            # Note : prefix is _elo_match here because it's used in the prompt template
            '_elo_match_competitor_a': self.competitor_a,
            '_elo_match_competitor_b': self.competitor_b,
            '_elo_round_winner': self.winner,
            '_elo_round_prompt_template': self.prompt_template,
            '_elo_round_llm_seed': self.llm_seed
        }

@dataclass(kw_only=True)
class EloMatch(Execution):
    """
    Represents an aggregation of multiple rounds between 2 competitors within a batch.
    """
    competitor_a: str
    competitor_b: str
    rounds: List[EloRound]
    winner: Optional[str]
    is_draw: bool
    pair_score: float
    
    @property
    def round_wins_a(self) -> int:
        return sum(1 for round in self.rounds if round.winner == self.competitor_a)
    
    @property
    def round_wins_b(self) -> int:
        return sum(1 for round in self.rounds if round.winner == self.competitor_b)
    
    @property
    def errored_rounds(self) -> int:
        return sum(1 for round in self.rounds if round.error is not None)

    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_elo_match_competitor_a': self.competitor_a,
            '_elo_match_competitor_b': self.competitor_b,
            '_elo_match_winner': self.winner,
            '_elo_match_draw': self.is_draw,
            '_elo_match_wins_a': self.round_wins_a,
            '_elo_match_wins_b': self.round_wins_b,
            '_elo_match_errored_rounds': self.errored_rounds,
            '_elo_match_pair_score': self.pair_score
        }

@dataclass(kw_only=True)
class EloCompetitorRating(Execution):
    competitor: str
    rating: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rank: int = None

    @property
    def matches_played(self) -> int:
        return self.wins + self.losses + self.draws

    def get_specific_variables(self) -> Dict[str, Any]:
        return {
            '_elo_competitor': self.competitor,
            '_elo_rating': int(self.rating),
            '_elo_wins': int(self.wins),
            '_elo_loss': int(self.losses),
            '_elo_draws': int(self.draws),
            '_elo_rank': self.rank if self.rank is not None else 0
        }

# Constants
DEFAULT_INITIAL_RATING = 1000
DEFAULT_K_FACTOR = 32 