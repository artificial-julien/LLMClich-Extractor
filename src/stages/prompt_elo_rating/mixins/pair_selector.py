import random
import math
from typing import List, Set, Tuple, Dict, Optional
from ..types import *
from src.execution import Execution

class PairSelectorMixin:
    """Mixin providing advanced pair selection functionality with scoring system."""
    
    def __init__(self):
        # Track how often each pair has been matched
        self._pair_counts: Dict[Tuple[str, str], int] = {}
    
    def _get_normalized_pair_key(self, competitor_a: str, competitor_b: str) -> Tuple[str, str]:
        """Get normalized pair key (alphabetically sorted)."""
        return tuple(sorted([competitor_a, competitor_b]))
    
    def _get_pair_repetition_count(self, competitor_a: str, competitor_b: str) -> int:
        """Get how many times this pair has played together."""
        pair_key = self._get_normalized_pair_key(competitor_a, competitor_b)
        return self._pair_counts.get(pair_key, 0)
    
    def _update_pair_count(self, competitor_a: str, competitor_b: str) -> None:
        """Update the count for a pair."""
        pair_key = self._get_normalized_pair_key(competitor_a, competitor_b)
        self._pair_counts[pair_key] = self._pair_counts.get(pair_key, 0) + 1
    
    def _calculate_uncertainty(self, competitor_rating: EloCompetitorRating) -> float:
        """
        Calculate uncertainty factor based on number of matches played.
        Returns higher values for competitors with fewer matches (higher uncertainty).
        """
        if competitor_rating.matches_played == 0:
            return 1.0
        # Exponential decay: more matches = lower uncertainty
        return math.exp(-competitor_rating.matches_played / 10.0)
    
    def _calculate_pair_score(
        self,
        competitor_a: str,
        competitor_b: str,
        stats: Dict[str, EloCompetitorRating],
        factors: PairSelectionFactors
    ) -> float:
        """
        Calculate score for a pair. Higher scores indicate better pairs to select.
        
        Args:
            competitor_a: First competitor
            competitor_b: Second competitor
            stats: Current competitor statistics
            factors: Weighting factors for different components
            
        Returns:
            Score for this pair (higher = better)
        """
        rating_a = stats[competitor_a]
        rating_b = stats[competitor_b]
        
        # Uncertainty component - reward pairs with uncertain competitors
        uncertainty_a = self._calculate_uncertainty(rating_a)
        uncertainty_b = self._calculate_uncertainty(rating_b)
        uncertainty_score = (uncertainty_a + uncertainty_b) / 2.0
        
        # Repetition penalty - penalize pairs that have played many times
        repetition_count = self._get_pair_repetition_count(competitor_a, competitor_b)
        repetition_penalty = math.exp(-repetition_count * 0.5)  # Exponential penalty
        
        # Rating difference penalty - prefer close ratings but don't completely exclude different ones
        rating_diff = abs(rating_a.rating - rating_b.rating)
        # Normalize by typical rating range (assuming 400-1600 range)
        normalized_diff = rating_diff / 400.0
        rating_diff_penalty = math.exp(-normalized_diff)
        
        # Combine components with factors
        total_score = (
            factors.uncertainty_factor * uncertainty_score +
            factors.pair_repetition_count_factor * repetition_penalty +
            factors.rating_difference_penalty_factor * rating_diff_penalty
        )
        
        return total_score
    
    def select_best_pairs(
        self,
        competitors: List[str],
        stats: Dict[str, EloCompetitorRating],
        n_pairs: int,
        factors: PairSelectionFactors,
        rng: random.Random,
        scheduled_pairs: Optional[Set[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        """
        Select the n best pairs based on scoring system.
        
        Args:
            competitors: List of all competitors
            stats: Current competitor statistics
            n_pairs: Number of pairs to select
            factors: Weighting factors for scoring
            rng: Random number generator for tie-breaking
            scheduled_pairs: Set of already scheduled pairs to avoid
            
        Returns:
            List of selected pairs
        """
        if scheduled_pairs is None:
            scheduled_pairs = set()
        
        # Generate all possible pairs with their scores
        pair_scores: List[PairScore] = []
        
        for i in range(len(competitors)):
            for j in range(i + 1, len(competitors)):
                competitor_a, competitor_b = competitors[i], competitors[j]
                
                # Skip already scheduled pairs
                if ((competitor_a, competitor_b) in scheduled_pairs or 
                    (competitor_b, competitor_a) in scheduled_pairs):
                    continue
                
                score = self._calculate_pair_score(competitor_a, competitor_b, stats, factors)
                pair_scores.append(PairScore(score, competitor_a, competitor_b))
        
        # Sort by score descending, with random tie-breaking
        rng.shuffle(pair_scores)  # Randomize first for consistent tie-breaking
        pair_scores.sort(key=lambda x: x.score, reverse=True)
        
        # Select top n pairs, ensuring no competitor appears twice
        selected_pairs: List[Tuple[str, str]] = []
        used_competitors: Set[str] = set()
        
        for pair_score in pair_scores:
            if len(selected_pairs) >= n_pairs:
                break
                
            if pair_score.competitor_a not in used_competitors and pair_score.competitor_b not in used_competitors:
                selected_pairs.append((pair_score.competitor_a, pair_score.competitor_b))
                used_competitors.add(pair_score.competitor_a)
                used_competitors.add(pair_score.competitor_b)
                
                # Update pair count tracking
                self._update_pair_count(pair_score.competitor_a, pair_score.competitor_b)
        
        return selected_pairs
    
    def generate_match_batch_with_scoring(
        self,
        base_execution: Execution,
        stats: Dict[str, EloCompetitorRating],
        model_config: ModelConfig,
        prompt_templates: List[str],
        llm_seed: int,
        symmetric_matches: bool,
        factors: PairSelectionFactors,
        rng: random.Random
    ) -> List[EloMatch]:
        """
        Generate a batch of match jobs using the scoring-based pair selection.
        
        Args:
            base_execution: Base execution to inherit variables from
            stats: Current competitor statistics
            model_config: Model configuration
            prompt_templates: List of prompt templates
            llm_seed: Current llm_seed value
            symmetric_matches: Whether to generate symmetric matches
            factors: Weighting factors for pair selection
            rng: Random number generator instance
            
        Returns:
            List of match jobs
        """
        competitors = list(stats.keys())
        n_pairs = len(competitors) // 2
        
        # Select best pairs using scoring system
        selected_pairs = self.select_best_pairs(
            competitors=competitors,
            stats=stats,
            n_pairs=n_pairs,
            factors=factors,
            rng=rng
        )
        
        # Create EloMatch objects for each selected pair
        matches: List[EloMatch] = []
        for competitor_a, competitor_b in selected_pairs:
            rounds: List[EloRound] = []
            
            # Calculate the pair score for this match
            pair_score = self._calculate_pair_score(competitor_a, competitor_b, stats, factors)
            
            for prompt_template in prompt_templates:
                def create_elo_round(comp_a: str, comp_b: str, is_mirror: bool) -> EloRound:
                    round = EloRound(
                        competitor_a=comp_a,
                        competitor_b=comp_b,
                        model_config=model_config,
                        prompt_template=prompt_template,
                        llm_seed=llm_seed,
                        winner=None,
                        is_mirror=is_mirror
                    )
                    round.import_variables_from(base_execution)
                    return round

                rounds.append(create_elo_round(competitor_a, competitor_b, False))
                if symmetric_matches:
                    rounds.append(create_elo_round(competitor_b, competitor_a, True))
            
            # Create EloMatch with all rounds for this competitor pair
            match = EloMatch(
                competitor_a=competitor_a,
                competitor_b=competitor_b,
                rounds=rounds,
                winner=None,  # Will be determined after processing
                is_draw=False,  # Will be determined after processing
                model_config=model_config,
                pair_score=pair_score
            )
            match.import_variables_from(base_execution)
            matches.append(match)
                    
        return matches 