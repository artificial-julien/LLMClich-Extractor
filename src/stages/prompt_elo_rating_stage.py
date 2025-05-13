from typing import List, Dict, Any, Optional, Tuple
import random
import itertools
import math
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
from src.llm import LLMClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

@StageRegistry.register("prompt_elo_rating")
class PromptEloRatingStage(Stage):
    """
    Stage that implements Elo rating calculations for LLM-judged competitions.
    """
    
    def __init__(
        self, 
        models: List[Dict[str, Any]],
        competitors: List[str],
        prompts: List[str],
        matches_per_entity: int,
        initial_rating: int,
        symmetric_matches: bool = False,
        parallel: int = 2
    ):
        """
        Initialize the prompt Elo rating stage.
        
        Args:
            models: List of model configurations (name, temperature, iterations, etc.)
            competitors: List of competitors to be ranked
            prompts: List of prompt templates
            matches_per_entity: Number of matches per competitor
            initial_rating: Initial Elo rating for all competitors
            symmetric_matches: Whether to run matches in both directions
            parallel: Number of parallel requests
        """
        self.models = models
        self.competitors = competitors
        self.prompts = prompts
        self.matches_per_entity = matches_per_entity
        self.initial_rating = initial_rating
        self.symmetric_matches = symmetric_matches
        self.parallel = parallel
        self.llm_client = LLMClient()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PromptEloRatingStage':
        """
        Create a PromptEloRatingStage from configuration.
        
        Args:
            config: Dictionary containing stage configuration
            
        Returns:
            A PromptEloRatingStage instance
            
        Raises:
            ValueError: If the config is invalid
        """
        models = config.get('models')
        competitors = config.get('competitors')
        prompts = config.get('prompts')
        matches_per_entity = config.get('matches_per_entity', 4)
        initial_rating = config.get('initial_rating', 1000)
        symmetric_matches = config.get('symmetric_matches', False)
        
        if not models or not isinstance(models, list):
            raise ValueError("PromptEloRatingStage config must contain a 'models' list")
        
        if not competitors or not isinstance(competitors, list):
            raise ValueError("PromptEloRatingStage config must contain a 'competitors' list")
        
        if not prompts or not isinstance(prompts, list):
            raise ValueError("PromptEloRatingStage config must contain a 'prompts' list")
        
        return cls(
            models=models,
            competitors=competitors,
            prompts=prompts,
            matches_per_entity=matches_per_entity,
            initial_rating=initial_rating,
            symmetric_matches=symmetric_matches
        )
    
    def format_prompt(self, template: str, _elo_match_competitor_a: str, _elo_match_competitor_b: str) -> str:
        """
        Format a prompt template with competitor variables.
        
        Args:
            template: Prompt template with [_elo_match_competitor_a] and [_elo_match_competitor_b] placeholders
            _elo_match_competitor_a: First competitor
            _elo_match_competitor_b: Second competitor
            
        Returns:
            Formatted prompt
        """
        formatted_prompt = template.replace("[_elo_match_competitor_a]", _elo_match_competitor_a).replace("[_elo_match_competitor_b]", _elo_match_competitor_b)
        
        # Add possible answers to the prompt
        formatted_prompt += f"\n\nPossible answers:\n1. {_elo_match_competitor_a}\n2. {_elo_match_competitor_b}"
        
        return formatted_prompt
    
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
    
    def update_elo_rating(self, old_rating: float, expected_score: float, actual_score: float, k_factor: float = 32) -> float:
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
    
    def process_match(
        self, 
        _elo_match_competitor_a: str, 
        _elo_match_competitor_b: str, 
        model_config: Dict[str, Any],
        prompt_template: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Process a single match between two competitors.
        
        Args:
            _elo_match_competitor_a: First competitor
            _elo_match_competitor_b: Second competitor
            model_config: Model configuration
            prompt_template: Prompt template
            iteration: Iteration number (for seed)
            
        Returns:
            Dictionary with match results
        """
        model_name = model_config.get('name')
        temperature = float(model_config.get('temperature', 0.0))
        top_p = float(model_config.get('top_p', 1.0))
        
        # Format the prompt with competitor names
        formatted_prompt = self.format_prompt(prompt_template, _elo_match_competitor_a, _elo_match_competitor_b)
        
        # Call the LLM with two possible answers (the two competitors)
        result = self.llm_client.generate_constrained_completion(
            model=model_name,
            prompt=formatted_prompt,
            possible_answers=[_elo_match_competitor_a, _elo_match_competitor_b],
            temperature=temperature,
            top_p=top_p,
            seed=iteration
        )
        
        # Determine match outcome
        if result['error']:
            return {
                '_elo_match_competitor_a': _elo_match_competitor_a,
                '_elo_match_competitor_b': _elo_match_competitor_b,
                'winner': None,
                'is_draw': False,
                'error': result['error'],
                'model_name': model_name,
                'temperature': temperature,
                'top_p': top_p,
                'seed': iteration
            }
        
        winner = result['chosen_answer']
        
        # For symmetric matches, we need to check if this is a draw
        is_draw = False
        if self.symmetric_matches:
            # Get the other match result for this pair
            other_match = next(
                (m for m in self._current_match_results if 
                 m['_elo_match_competitor_a'] == _elo_match_competitor_b and 
                 m['_elo_match_competitor_b'] == _elo_match_competitor_a and
                 m['model_name'] == model_name and
                 m['seed'] == iteration),
                None
            )
            if other_match and other_match['winner'] == winner:
                is_draw = True
        
        return {
            '_elo_match_competitor_a': _elo_match_competitor_a,
            '_elo_match_competitor_b': _elo_match_competitor_b,
            'winner': winner,
            'is_draw': is_draw,
            'error': None,
            'model_name': model_name,
            'temperature': temperature,
            'top_p': top_p,
            'seed': iteration
        }
    
    def generate_matches(self) -> List[Tuple[str, str]]:
        """
        Generate matches between competitors using a Swiss system approach.

        Returns:
            List of tuples (_elo_match_competitor_a, _elo_match_competitor_b) representing matches
        """
        matches = []
        matches_played = {competitor: 0 for competitor in self.competitors}
        # For Swiss, we use initial Elo for all rounds (since ratings are updated after all matches)
        ratings = {competitor: self.initial_rating for competitor in self.competitors}
        # Track already scheduled pairs to avoid duplicates
        scheduled_pairs = set()
        competitors = list(self.competitors)
        total_matches = math.ceil(len(competitors) * self.matches_per_entity / 2)
        if self.symmetric_matches:
            total_matches = math.ceil(total_matches / 2)  # Halve the total since each match will be played twice

        def available_pairs():
            # Return all possible pairs that haven't reached match limits and aren't already scheduled
            pairs = []
            for i in range(len(competitors)):
                for j in range(i+1, len(competitors)):
                    a, b = competitors[i], competitors[j]
                    if (
                        matches_played[a] < self.matches_per_entity and
                        matches_played[b] < self.matches_per_entity and
                        (a, b) not in scheduled_pairs and (b, a) not in scheduled_pairs
                    ):
                        pairs.append((a, b))
            return pairs

        while True:
            pairs = available_pairs()
            if not pairs:
                break
            # Find the minimum number of matches played among all competitors
            min_matches = min(matches_played[a] for pair in pairs for a in pair)
            # Filter pairs where both have the minimum matches
            min_pairs = [pair for pair in pairs if matches_played[pair[0]] == min_matches and matches_played[pair[1]] == min_matches]
            if not min_pairs:
                min_pairs = pairs
            # Among those, find pairs with the smallest Elo difference
            min_elo_diff = min(abs(ratings[a] - ratings[b]) for a, b in min_pairs)
            best_pairs = [pair for pair in min_pairs if abs(ratings[pair[0]] - ratings[pair[1]]) == min_elo_diff]
            # Pick randomly among best pairs
            chosen_pair = random.choice(best_pairs)
            a, b = chosen_pair
            matches.append((a, b))
            scheduled_pairs.add((a, b))
            matches_played[a] += 1
            matches_played[b] += 1
            # Stop if all competitors have reached matches_per_entity
            if all(m == self.matches_per_entity for m in matches_played.values()):
                break
            # Stop if enough matches
            if len(matches) >= total_matches:
                break
        # If symmetric matches are enabled, add the reverse of each match
        if self.symmetric_matches:
            matches = matches + [(b, a) for (a, b) in matches]
        return matches
    
    def _process_model_iteration(
        self,
        model_config: Dict[str, Any],
        seed: int,
        base_execution: Execution
    ) -> List[Execution]:
        """
        Process matches for a single model configuration and seed.
        
        Args:
            model_config: Model configuration dictionary
            seed: Current seed value
            base_execution: Base execution to copy from
            
        Returns:
            List of executions containing match results and ratings
        """
        model_name = model_config.get('name')
        temperature = float(model_config.get('temperature', 0.0))
        top_p = float(model_config.get('top_p', 1.0))
        
        # Generate matches for this (model, seed)
        matches = self.generate_matches()
        
        # Create match jobs
        jobs = []
        for _elo_match_competitor_a, _elo_match_competitor_b in matches:
            for prompt_template in self.prompts:
                jobs.append((_elo_match_competitor_a, _elo_match_competitor_b, model_config, prompt_template, seed))
        
        # Process matches in parallel
        match_results = self._run_parallel_matches(jobs, model_name, temperature, top_p, seed)
        
        # Calculate ratings and create result executions
        return self._create_result_executions(match_results, model_name, temperature, top_p, seed, base_execution)

    def _run_parallel_matches(
        self,
        jobs: List[Tuple],
        model_name: str,
        temperature: float,
        top_p: float,
        seed: int
    ) -> List[Dict[str, Any]]:
        """
        Run matches in parallel using ThreadPoolExecutor.
        
        Args:
            jobs: List of match jobs to process
            model_name: Name of the model being used
            temperature: Model temperature setting
            top_p: Model top_p setting
            seed: Current seed value
            
        Returns:
            List of match results
        """
        match_results = []
        self._current_match_results = []  # Track current match results for draw detection
        
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = []
            for _elo_match_competitor_a, _elo_match_competitor_b, model_config, prompt_template, seed_val in jobs:
                future = executor.submit(
                    self.process_match,
                    _elo_match_competitor_a,
                    _elo_match_competitor_b,
                    model_config,
                    prompt_template,
                    seed_val
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Elo matches for {model_name} (seed {seed})"):
                try:
                    result = future.result()
                    match_results.append(result)
                    self._current_match_results.append(result)
                except Exception as e:
                    job = jobs[len(match_results)]
                    _elo_match_competitor_a, _elo_match_competitor_b = job[0], job[1]
                    match_results.append({
                        '_elo_match_competitor_a': _elo_match_competitor_a,
                        '_elo_match_competitor_b': _elo_match_competitor_b,
                        'winner': None,
                        'is_draw': False,
                        'error': str(e),
                        'model_name': model_name,
                        'temperature': temperature,
                        'top_p': top_p,
                        'seed': seed
                    })
        
        return match_results

    def _calculate_ratings(self, match_results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Calculate Elo ratings and match statistics from match results.
        
        Args:
            match_results: List of match results
            
        Returns:
            Tuple of (ratings, wins, losses, draws) dictionaries
        """
        ratings = {competitor: self.initial_rating for competitor in self.competitors}
        wins = {competitor: 0 for competitor in self.competitors}
        losses = {competitor: 0 for competitor in self.competitors}
        draws = {competitor: 0 for competitor in self.competitors}

        if self.symmetric_matches:
            self._process_symmetric_matches(match_results, ratings, wins, losses, draws)
        else:
            self._process_regular_matches(match_results, ratings, wins, losses, draws)

        return ratings, wins, losses, draws

    def _process_symmetric_matches(
        self,
        match_results: List[Dict[str, Any]],
        ratings: Dict[str, float],
        wins: Dict[str, int],
        losses: Dict[str, int],
        draws: Dict[str, int]
    ) -> None:
        """Process matches in pairs for symmetric matches to detect draws."""
        match_pairs = {}
        for match in match_results:
            if match['error'] or not match['winner']:
                continue
            a, b = match['_elo_match_competitor_a'], match['_elo_match_competitor_b']
            pair_key = tuple(sorted([a, b]))
            if pair_key not in match_pairs:
                match_pairs[pair_key] = []
            match_pairs[pair_key].append(match)
        
        for pair_key, pair_matches in match_pairs.items():
            if len(pair_matches) != 2:
                continue
            
            a, b = pair_key
            a_wins = sum(1 for m in pair_matches if m['winner'] == a)
            b_wins = sum(1 for m in pair_matches if m['winner'] == b)
            
            if a_wins == b_wins == 1:  # Draw
                draws[a] += 1
                draws[b] += 1
                a_score = 0.5
                b_score = 0.5
            else:
                if a_wins > b_wins:
                    wins[a] += 1
                    losses[b] += 1
                    a_score = 1
                    b_score = 0
                else:
                    wins[b] += 1
                    losses[a] += 1
                    a_score = 0
                    b_score = 1
            
            a_expected = self.calculate_expected_score(ratings[a], ratings[b])
            b_expected = 1 - a_expected
            ratings[a] = self.update_elo_rating(ratings[a], a_expected, a_score)
            ratings[b] = self.update_elo_rating(ratings[b], b_expected, b_score)

    def _process_regular_matches(
        self,
        match_results: List[Dict[str, Any]],
        ratings: Dict[str, float],
        wins: Dict[str, int],
        losses: Dict[str, int],
        draws: Dict[str, int]
    ) -> None:
        """Process matches individually for non-symmetric matches."""
        for match in match_results:
            if match['error'] or not match['winner']:
                continue
            a = match['_elo_match_competitor_a']
            b = match['_elo_match_competitor_b']
            winner = match['winner']
            
            if winner == a:
                wins[a] += 1
                losses[b] += 1
                a_score = 1
                b_score = 0
            else:
                wins[b] += 1
                losses[a] += 1
                a_score = 0
                b_score = 1
            
            a_expected = self.calculate_expected_score(ratings[a], ratings[b])
            b_expected = 1 - a_expected
            ratings[a] = self.update_elo_rating(ratings[a], a_expected, a_score)
            ratings[b] = self.update_elo_rating(ratings[b], b_expected, b_score)

    def _create_result_executions(
        self,
        match_results: List[Dict[str, Any]],
        model_name: str,
        temperature: float,
        top_p: float,
        seed: int,
        base_execution: Execution
    ) -> List[Execution]:
        """
        Create execution objects for match results and ratings.
        
        Args:
            match_results: List of match results
            model_name: Name of the model used
            temperature: Model temperature setting
            top_p: Model top_p setting
            seed: Current seed value
            base_execution: Base execution to copy from
            
        Returns:
            List of executions containing match results and ratings
        """
        result_executions = []
        
        # Calculate ratings
        ratings, wins, losses, draws = self._calculate_ratings(match_results)
        
        # Add match results
        for match in match_results:
            if match['error'] or not match['winner']:
                continue
            match_execution = base_execution.copy()
            match_execution.add_variable('_elo_match_competitor_a', match['_elo_match_competitor_a'])
            match_execution.add_variable('_elo_match_competitor_b', match['_elo_match_competitor_b'])
            match_execution.add_variable('_elo_match_winner', None if match['is_draw'] else match['winner'])
            match_execution.add_variable('_elo_match_draw', match['is_draw'])
            match_execution.add_variable('_seed', seed)
            match_execution.add_variable('_model_name', model_name)
            match_execution.add_variable('_model_temperature', temperature)
            match_execution.add_variable('_model_top_p', top_p)
            result_executions.append(match_execution)
        
        # Add final Elo ratings
        for competitor, rating in ratings.items():
            rating_execution = base_execution.copy()
            rating_execution.add_variable('_elo_competitor', competitor)
            rating_execution.add_variable('_elo_rating', int(rating))
            rating_execution.add_variable('_elo_wins', wins[competitor])
            rating_execution.add_variable('_elo_loss', losses[competitor])
            rating_execution.add_variable('_elo_draws', draws[competitor])
            rating_execution.add_variable('_seed', seed)
            rating_execution.add_variable('_model_name', model_name)
            rating_execution.add_variable('_model_temperature', temperature)
            rating_execution.add_variable('_model_top_p', top_p)
            result_executions.append(rating_execution)
        
        return result_executions

    def process(self, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions, run Elo rating matches, and return results.
        
        Args:
            executions: List of input executions (typically just one)
            
        Returns:
            List of new executions with results for each competitor
        """
        # We only need one base execution to work with
        base_execution = executions[0] if executions else Execution()
        result_executions = []

        for model_config in self.models:
            iterations = int(model_config.get('iterations', 1))
            for seed in range(iterations):
                result_executions.extend(
                    self._process_model_iteration(model_config, seed, base_execution)
                )

        return result_executions 