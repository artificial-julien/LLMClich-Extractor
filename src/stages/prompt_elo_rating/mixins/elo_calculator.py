from typing import Dict, Tuple
from ..types import EloCompetitorRating, DEFAULT_K_FACTOR

class EloCalculatorMixin:
    """Mixin providing Elo rating calculation functionality."""
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate the expected score for player A when playing against player B.
        
        Args:
            rating_a: Elo rating of player A
            rating_b: Elo rating of player B
            
        Returns:
            Expected score (between 0 and 1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_elo_rating(
        self, 
        old_rating: float, 
        expected_score: float, 
        actual_score: float, 
        k_factor: float = DEFAULT_K_FACTOR
    ) -> float:
        """
        Update Elo rating based on the match result.
        
        Args:
            old_rating: Current Elo rating
            expected_score: Expected score from calculate_expected_score
            actual_score: Actual match outcome (1 for win, 0.5 for draw, 0 for loss)
            k_factor: K-factor determining the maximum rating change
            
        Returns:
            Updated Elo rating
        """
        return old_rating + k_factor * (actual_score - expected_score)
    
    def update_ratings_and_stats(
        self,
        competitor_a: str,
        competitor_b: str,
        a_score: float,
        b_score: float,
        stats: Dict[str, EloCompetitorRating]
    ) -> None:
        """
        Update Elo ratings and match statistics for a pair of competitors.
        
        Args:
            competitor_a: First competitor
            competitor_b: Second competitor
            a_score: Score for competitor A
            b_score: Score for competitor B
            stats: Dictionary of competitor statistics
        """
        a_expected = self.calculate_expected_score(stats[competitor_a].rating, stats[competitor_b].rating)
        b_expected = 1 - a_expected
        
        stats[competitor_a].rating = self.update_elo_rating(stats[competitor_a].rating, a_expected, a_score)
        stats[competitor_b].rating = self.update_elo_rating(stats[competitor_b].rating, b_expected, b_score)
        
        if a_score == 1:
            stats[competitor_a].wins += 1
            stats[competitor_b].losses += 1
        elif b_score == 1:
            stats[competitor_b].wins += 1
            stats[competitor_a].losses += 1
        elif a_score == 0.5:
            stats[competitor_a].draws += 1
            stats[competitor_b].draws += 1
