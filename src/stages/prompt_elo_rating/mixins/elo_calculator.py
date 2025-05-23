from typing import Dict, Tuple, List
from ..types import EloCompetitorRating, DEFAULT_K_FACTOR, EloMatch
import copy

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
        matches: List[EloMatch],
        stats: Dict[str, EloCompetitorRating]
    ) -> Dict[str, EloCompetitorRating]:
        """
        Update Elo ratings and match statistics for a batch of matches using original ratings.
        
        Args:
            matches: List of EloMatch objects containing match results
            stats: Dictionary of competitor statistics with original ratings
            
        Returns:
            Updated dictionary of competitor statistics
        """
        updated_stats = copy.deepcopy(stats)
        
        # Calculate all expected scores and rating changes using original ratings
        rating_changes = {}
        for match in matches:
            if not match.winner and not match.is_draw:
                continue
                
            a_score = 1.0 if match.winner == match.competitor_a else (0.5 if match.is_draw else 0.0)
            b_score = 1.0 if match.winner == match.competitor_b else (0.5 if match.is_draw else 0.0)
            
            a_expected = self.calculate_expected_score(stats[match.competitor_a].rating, stats[match.competitor_b].rating)
            b_expected = 1 - a_expected
            
            # Calculate rating changes but don't apply them yet
            a_change = self.update_elo_rating(stats[match.competitor_a].rating, a_expected, a_score) - stats[match.competitor_a].rating
            b_change = self.update_elo_rating(stats[match.competitor_b].rating, b_expected, b_score) - stats[match.competitor_b].rating
            
            # Accumulate rating changes
            rating_changes[match.competitor_a] = rating_changes.get(match.competitor_a, 0) + a_change
            rating_changes[match.competitor_b] = rating_changes.get(match.competitor_b, 0) + b_change
            
            # Update match statistics
            if a_score == 1:
                updated_stats[match.competitor_a].wins += 1
                updated_stats[match.competitor_b].losses += 1
            elif b_score == 1:
                updated_stats[match.competitor_b].wins += 1
                updated_stats[match.competitor_a].losses += 1
            elif a_score == 0.5:
                updated_stats[match.competitor_a].draws += 1
                updated_stats[match.competitor_b].draws += 1
        
        # Apply accumulated rating changes to original ratings
        for competitor, change in rating_changes.items():
            updated_stats[competitor].rating = stats[competitor].rating + change
            
        return updated_stats
