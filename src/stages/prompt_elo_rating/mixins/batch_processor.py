from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, DefaultDict, Optional
from collections import defaultdict
from tqdm import tqdm
from ....commons import PipelineConfig
from ..types import RoundJob, Round, Match, CompetitorStats

class BatchProcessorMixin:
    """Mixin providing batch processing functionality."""
    
    def __init__(self):
        self._current_rounds: List[Round] = []
    
    def process_batch(
        self,
        pipeline_config: PipelineConfig,
        jobs: List[RoundJob],
        symmetric_matches: bool,
        pbar: Optional[tqdm] = None
    ) -> List[Round]:
        """
        Process a batch of rounds in parallel.
        
        Args:
            jobs: List of round jobs to process
            symmetric_matches: Whether to run matches in both directions
            pbar: Optional progress bar to update
            
        Returns:
            List of round results
        """
        rounds = []
        
        # Run rounds in parallel
        with ThreadPoolExecutor(max_workers=pipeline_config.parallel) as executor:
            futures = []
            for job in jobs:
                future = executor.submit(
                    self.process_round,
                    job['competitor_a'],
                    job['competitor_b'],
                    job['model_config'],
                    job['prompt_template'],
                    job['llm_seed']
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    rounds.append(result)
                    self._current_rounds.append(result)
                    if pbar is not None:
                        pbar.update(1)
                except Exception as e:
                    job = jobs[len(rounds)]
                    rounds.append(Round(
                        competitor_a=job['competitor_a'],
                        competitor_b=job['competitor_b'],
                        winner=None,
                        error=str(e),
                        model_name=job['model_config'].name,
                        temperature=job['model_config'].temperature,
                        top_p=job['model_config'].top_p,
                        llm_seed=job['llm_seed']
                    ))
                    if pbar is not None:
                        pbar.update(1)
        
        # Update symmetric matches if needed
        if symmetric_matches:
            self._update_symmetric_matches(rounds)
        
        return rounds
    
    def group_rounds_into_matches(self, rounds: List[Round]) -> List[Match]:
        """
        Group rounds into matches based on competitors.
        
        Args:
            rounds: List of rounds to group
            
        Returns:
            List of matches containing grouped rounds
        """
        # Group rounds by competitor pairs
        match_groups = defaultdict(list)
        for round_result in rounds:
            # Standardize key to ensure (A,B) and (B,A) are grouped together
            competitors = sorted([round_result.competitor_a, round_result.competitor_b])
            key = (competitors[0], competitors[1])
            match_groups[key].append(round_result)
        
        # Create Match objects from grouped rounds
        matches = []
        for (comp_a, comp_b), grouped_rounds in match_groups.items():
            # Count wins for each competitor
            wins_a = sum(1 for r in grouped_rounds if r.winner == comp_a)
            wins_b = sum(1 for r in grouped_rounds if r.winner == comp_b)
            
            # Determine match winner
            is_draw = wins_a == wins_b
            winner = None
            if not is_draw:
                winner = comp_a if wins_a > wins_b else comp_b
            
            match = Match(
                competitor_a=comp_a,
                competitor_b=comp_b,
                rounds=grouped_rounds,
                winner=winner,
                is_draw=is_draw
            )
            matches.append(match)
        
        return matches
    
    def _update_symmetric_matches(self, rounds: List[Round]) -> None:
        """
        Update symmetric matches to ensure consistency.
        
        Args:
            rounds: List of rounds to update
        """
        # Group rounds by competitor pairs
        round_map = {}
        for round_result in rounds:
            key = (round_result.competitor_a, round_result.competitor_b)
            if key in round_map:
                continue
            round_map[key] = round_result
        
        # Update symmetric matches
        for (a, b), round_a_b in round_map.items():
            reverse_key = (b, a)
            if reverse_key in round_map:
                round_b_a = round_map[reverse_key]
                # If both agree on the same winner, keep it
                if (round_a_b.winner == a and round_b_a.winner == a) or (round_a_b.winner == b and round_b_a.winner == b):
                    continue
                # Otherwise, set both to None (will be handled at match level)
                round_a_b.winner = None
                round_b_a.winner = None
