from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, DefaultDict, Optional
from collections import defaultdict
from tqdm import tqdm
from ....commons import PipelineConfig
from ..types import EloRound, EloRound, EloMatch
from src.execution import Execution


class BatchProcessorMixin:
    """Mixin providing batch processing functionality."""
    
    def __init__(self):
        self._current_rounds: List[EloRound] = []
    
    def process_batch(
        self,
        pipeline_config: PipelineConfig,
        jobs: List[EloRound],
        pbar: Optional[tqdm] = None
    ) -> List[EloRound]:
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
                    job,
                    pipeline_config
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result : EloRound = future.result()
                    rounds.append(result)
                    self._current_rounds.append(result)
                except Exception as e:
                    job = jobs[len(rounds)]
                    job.error = str(e)
                    rounds.append(job)
                finally:
                    if pbar is not None:
                        pbar.update(1)
    
        
        return rounds
    
    def group_rounds_into_matches(self, base_execution: Execution, rounds: List[EloRound]) -> List[EloMatch]:
        """
        Group rounds into matches based on competitors.
        
        Args:
            rounds: List of rounds to group
            
        Returns:
            List of matches containing grouped rounds
        """
        # Group rounds by competitor pairs
        match_groups: Dict[Tuple[str, str], List[EloRound]] = defaultdict(list)
        for round_result in rounds:
            # Standardize key to ensure (A,B) and (B,A) are grouped together
            competitors = sorted([round_result.competitor_a, round_result.competitor_b])
            key = (competitors[0], competitors[1])
            match_groups[key].append(round_result)
            
        
        # Create EloMatch objects from grouped rounds
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
            
            match = EloMatch(
                competitor_a=comp_a,
                competitor_b=comp_b,
                rounds=grouped_rounds,
                winner=winner,
                is_draw=is_draw,
                model_config=grouped_rounds[0].model_config,
            )
            match.import_variables_from(base_execution)
            matches.append(match)
        
        return matches
