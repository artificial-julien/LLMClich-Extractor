from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, DefaultDict, Optional
from collections import defaultdict
from tqdm import tqdm
from src.common.types import *
from ..types import EloRound, EloMatch
from src.execution import Execution


class BatchProcessorMixin:
    """Mixin providing batch processing functionality."""
    
    def __init__(self):
        self._current_rounds: List[EloRound] = []
    
    def process_batch(
        self,
        pipeline_config: PipelineConfig,
        matches: List[EloMatch],
        pbar: Optional[tqdm] = None
    ) -> List[EloMatch]:
        all_rounds = self._collect_all_rounds(matches)
        processed_rounds = self._process_rounds_parallel(all_rounds, pipeline_config, pbar)
        self._update_matches_with_processed_rounds(matches, processed_rounds)
        return matches
    
    def _collect_all_rounds(self, matches: List[EloMatch]) -> List[EloRound]:
        all_rounds: List[EloRound] = []
        for match in matches:
            all_rounds.extend(match.rounds)
        return all_rounds
    
    def _process_rounds_parallel(
        self, 
        all_rounds: List[EloRound], 
        pipeline_config: PipelineConfig, 
        pbar: Optional[tqdm]
    ) -> List[EloRound]:
        round_results = [None] * len(all_rounds)
        
        with ThreadPoolExecutor(max_workers=pipeline_config.parallel) as executor:
            future_to_round = self._submit_round_jobs(executor, all_rounds, pipeline_config)
            self._collect_round_results(future_to_round, round_results, pbar)
        
        return round_results
    
    def _submit_round_jobs(
        self, 
        executor: ThreadPoolExecutor, 
        all_rounds: List[EloRound], 
        pipeline_config: PipelineConfig
    ) -> Dict:
        future_to_round = {}
        for i, round_job in enumerate(all_rounds):
            future = executor.submit(
                self.process_round,
                round_job,
                pipeline_config
            )
            future_to_round[future] = (i, round_job)
        return future_to_round
    
    def _collect_round_results(
        self, 
        future_to_round: Dict, 
        round_results: List[Optional[EloRound]], 
        pbar: Optional[tqdm]
    ) -> None:
        for future in as_completed(future_to_round.keys()):
            try:
                result: EloRound = future.result()
                round_index, original_round = future_to_round[future]
                round_results[round_index] = result
                self._current_rounds.append(result)
            except Exception as e:
                round_index, original_round = future_to_round[future]
                original_round.error = str(e)
                round_results[round_index] = original_round
            finally:
                if pbar is not None:
                    pbar.update(1)
    
    def _update_matches_with_processed_rounds(
        self, 
        matches: List[EloMatch], 
        processed_rounds: List[EloRound]
    ) -> None:
        round_index = 0
        for match in matches:
            num_rounds = len(match.rounds)
            match.rounds = processed_rounds[round_index:round_index + num_rounds]
            self._determine_match_winner(match)
            round_index += num_rounds
    
    def _determine_match_winner(self, match: EloMatch) -> None:
        wins_a = sum(1 for r in match.rounds if r.winner == match.competitor_a)
        wins_b = sum(1 for r in match.rounds if r.winner == match.competitor_b)
        
        match.is_draw = wins_a == wins_b
        if not match.is_draw:
            match.winner = match.competitor_a if wins_a > wins_b else match.competitor_b
        else:
            match.winner = None
