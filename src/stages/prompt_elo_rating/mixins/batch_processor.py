from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from tqdm import tqdm
from ..types import MatchJob, MatchResult, CompetitorStats, DEFAULT_PARALLEL_WORKERS

class BatchProcessorMixin:
    """Mixin providing batch processing functionality."""
    
    def __init__(self, parallel_workers: int = DEFAULT_PARALLEL_WORKERS):
        self.parallel_workers = parallel_workers
        self._current_match_results: List[MatchResult] = []
    
    def process_batch(
        self,
        jobs: List[MatchJob],
        symmetric_matches: bool
    ) -> Tuple[List[MatchResult], Dict[str, CompetitorStats]]:
        """
        Process a batch of matches in parallel.
        
        Args:
            jobs: List of match jobs to process
            symmetric_matches: Whether matches are symmetric
            
        Returns:
            Tuple of (match results, updated competitor stats)
        """
        match_results = []
        self._current_match_results = []
        
        # Run matches in parallel
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = []
            for job in jobs:
                future = executor.submit(
                    self.process_match,
                    job['competitor_a'],
                    job['competitor_b'],
                    job['model_config'],
                    job['prompt_template'],
                    job['seed']
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Elo matches"):
                try:
                    result = future.result()
                    match_results.append(result)
                    self._current_match_results.append(result)
                except Exception as e:
                    job = jobs[len(match_results)]
                    match_results.append(MatchResult(
                        competitor_a=job['competitor_a'],
                        competitor_b=job['competitor_b'],
                        winner=None,
                        is_draw=False,
                        error=str(e),
                        model_name=job['model_config'].name,
                        temperature=job['model_config'].temperature,
                        top_p=job['model_config'].top_p,
                        seed=job['seed']
                    ))
        
        # Update draw status for symmetric matches
        if symmetric_matches:
            self._update_draw_status(match_results)
        
        return match_results
    
    def _update_draw_status(self, match_results: List[MatchResult]) -> None:
        """
        Update draw status for symmetric matches.
        
        Args:
            match_results: List of match results to update
        """
        for result in match_results:
            if result.error or not result.winner:
                continue
                
            # Find the corresponding reverse match
            reverse_match = next(
                (m for m in self._current_match_results if 
                 m.competitor_a == result.competitor_b and 
                 m.competitor_b == result.competitor_a and
                 m.model_name == result.model_name and
                 m.seed == result.seed),
                None
            )
            
            # If reverse match exists and has same winner, it's a draw
            if reverse_match and reverse_match.winner == result.winner:
                result.is_draw = True
                reverse_match.is_draw = True 