import random
from typing import List, Set, Tuple, Dict, Optional
from ..types import EloCompetitorRating, EloRound, ModelConfig
from src.execution import Execution
class BatchGeneratorMixin:
    """Mixin providing Swiss system match generation functionality."""
    
    def generate_match_batch(
        self,
        base_execution: Execution,
        stats: Dict[str, EloCompetitorRating],
        model_config: ModelConfig,
        prompt_templates: List[str],
        llm_seed: int,
        symmetric_matches: bool,
        batch_seed: Optional[int] = None
    ) -> List[EloRound]:
        """
        Generate a batch of round jobs using Swiss system approach.
        
        Args:
            stats: Current competitor statistics
            model_config: Model configuration
            prompt_templates: List of prompt templates
            llm_seed: Current llm_seed value
            symmetric_matches: Whether to generate symmetric matches
            batch_seed: Optional seed for reproducible batch generation
            
        Returns:
            List of round jobs
        """
        rng = random.Random(batch_seed)
            
        matches = []
        scheduled_pairs: Set[Tuple[str, str]] = set()
        competitors = list(stats.keys())
        
        def available_pairs() -> List[Tuple[str, str]]:
            pairs = []
            for i in range(len(competitors)):
                for j in range(i+1, len(competitors)):
                    a, b = competitors[i], competitors[j]
                    if (a, b) not in scheduled_pairs and (b, a) not in scheduled_pairs:
                        pairs.append((a, b))
            return pairs

        while True:
            pairs = available_pairs()
            if not pairs:
                break
                
            # Find pairs with minimum matches played
            min_matches = min(stats[a].matches_played for pair in pairs for a in pair)
            min_pairs = [
                pair for pair in pairs 
                if stats[pair[0]].matches_played == min_matches 
                and stats[pair[1]].matches_played == min_matches
            ]
            if not min_pairs:
                min_pairs = pairs
                
            # Find pairs with minimum Elo difference
            min_elo_diff = min(abs(stats[a].rating - stats[b].rating) for a, b in min_pairs)
            best_pairs = [
                pair for pair in min_pairs 
                if abs(stats[pair[0]].rating - stats[pair[1]].rating) == min_elo_diff
            ]
            
            # Choose random pair from best pairs using the dedicated RNG instance
            chosen_pair = rng.choice(best_pairs)
            a, b = chosen_pair
            matches.append((a, b))
            scheduled_pairs.add((a, b))
            
            # Stop after generating a certain number of matches
            if len(matches) >= len(competitors) // 2:
                break

        # Create jobs for each match and prompt
        jobs: List[EloRound] = []
        for a, b in matches:
            for prompt_template in prompt_templates:
                def create_elo_round(competitor_a: str, competitor_b: str, is_mirror: bool) -> EloRound:
                    round = EloRound(
                        competitor_a=competitor_a,
                        competitor_b=competitor_b,
                        model_config=model_config,
                        prompt_template=prompt_template,
                        llm_seed=llm_seed,
                        winner=None,
                        is_mirror=is_mirror
                    )
                    round.import_variables_from(base_execution)

                    return round

                jobs.append(create_elo_round(a, b, False))
                if symmetric_matches:
                    jobs.append(create_elo_round(b, a, True))
                    
        return jobs 