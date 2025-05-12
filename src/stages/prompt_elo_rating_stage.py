from typing import List, Dict, Any, Optional, Tuple
import random
import itertools
import math
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
from src.llm_client import LLMClient
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
    
    def format_prompt(self, template: str, player_a: str, player_b: str) -> str:
        """
        Format a prompt template with player variables.
        
        Args:
            template: Prompt template with [player_a] and [player_b] placeholders
            player_a: First player/competitor
            player_b: Second player/competitor
            
        Returns:
            Formatted prompt
        """
        formatted_prompt = template.replace("[player_a]", player_a).replace("[player_b]", player_b)
        
        # Add possible answers to the prompt
        formatted_prompt += f"\n\nPossible answers:\n1. {player_a}\n2. {player_b}"
        
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
        player_a: str, 
        player_b: str, 
        model_config: Dict[str, Any],
        prompt_template: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Process a single match between two players.
        
        Args:
            player_a: First player
            player_b: Second player
            model_config: Model configuration
            prompt_template: Prompt template
            iteration: Iteration number (for seed)
            
        Returns:
            Dictionary with match results
        """
        model_name = model_config.get('name')
        temperature = float(model_config.get('temperature', 0.0))
        top_p = float(model_config.get('top_p', 1.0))
        
        # Format the prompt with player names
        formatted_prompt = self.format_prompt(prompt_template, player_a, player_b)
        
        # Call the LLM with only two possible answers (the two players)
        result = self.llm_client.generate_constrained_completion(
            model=model_name,
            prompt=formatted_prompt,
            possible_answers=[player_a, player_b],
            temperature=temperature,
            top_p=top_p,
            seed=iteration
        )
        
        # Determine match outcome
        if result['error']:
            return {
                'player_a': player_a,
                'player_b': player_b,
                'winner': None,
                'error': result['error'],
                'model_name': model_name,
                'temperature': temperature,
                'top_p': top_p,
                'seed': iteration
            }
        
        winner = result['chosen_answer']
        
        return {
            'player_a': player_a,
            'player_b': player_b,
            'winner': winner,
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
            List of tuples (player_a, player_b) representing matches
        """
        matches = []
        matches_played = {competitor: 0 for competitor in self.competitors}
        # For Swiss, we use initial Elo for all rounds (since ratings are updated after all matches)
        ratings = {competitor: self.initial_rating for competitor in self.competitors}
        # Track already scheduled pairs to avoid duplicates
        scheduled_pairs = set()
        competitors = list(self.competitors)
        total_matches = math.ceil(len(competitors) * self.matches_per_entity / 2)

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
            model_name = model_config.get('name')
            temperature = float(model_config.get('temperature', 0.0))
            top_p = float(model_config.get('top_p', 1.0))
            iterations = int(model_config.get('iterations', 1))

            for seed in range(iterations):
                # Generate matches for this (model, seed)
                matches = self.generate_matches()

                # Create match jobs for this (model, seed)
                jobs = []
                for player_a, player_b in matches:
                    for prompt_template in self.prompts:
                        jobs.append((player_a, player_b, model_config, prompt_template, seed))

                # Process matches in parallel for this (model, seed)
                match_results = []
                with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                    futures = []
                    for player_a, player_b, model_config, prompt_template, seed_val in jobs:
                        future = executor.submit(
                            self.process_match,
                            player_a,
                            player_b,
                            model_config,
                            prompt_template,
                            seed_val
                        )
                        futures.append(future)
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Elo matches for {model_name} (seed {seed})"):
                        try:
                            result = future.result()
                            match_results.append(result)
                        except Exception as e:
                            job = jobs[len(match_results)]
                            player_a, player_b = job[0], job[1]
                            match_results.append({
                                'player_a': player_a,
                                'player_b': player_b,
                                'winner': None,
                                'error': str(e),
                                'model_name': model_name,
                                'temperature': temperature,
                                'top_p': top_p,
                                'seed': seed
                            })

                # Calculate Elo ratings for this (model, seed)
                ratings = {competitor: self.initial_rating for competitor in self.competitors}
                wins = {competitor: 0 for competitor in self.competitors}
                losses = {competitor: 0 for competitor in self.competitors}

                for match in match_results:
                    if match['error'] or not match['winner']:
                        continue
                    player_a = match['player_a']
                    player_b = match['player_b']
                    winner = match['winner']
                    if winner == player_a:
                        wins[player_a] += 1
                        losses[player_b] += 1
                    else:
                        wins[player_b] += 1
                        losses[player_a] += 1
                    a_score = 1 if winner == player_a else 0
                    b_score = 1 - a_score
                    a_expected = self.calculate_expected_score(ratings[player_a], ratings[player_b])
                    b_expected = 1 - a_expected
                    ratings[player_a] = self.update_elo_rating(ratings[player_a], a_expected, a_score)
                    ratings[player_b] = self.update_elo_rating(ratings[player_b], b_expected, b_score)

                # Add match results for this (model, seed)
                for match in match_results:
                    if match['error'] or not match['winner']:
                        continue
                    match_execution = base_execution.copy()
                    match_execution.add_variable('_elo_match_competitor_a', match['player_a'])
                    match_execution.add_variable('_elo_match_competitor_b', match['player_b'])
                    match_execution.add_variable('_elo_match_winner', match['winner'])
                    match_execution.add_variable('_elo_match_draw', False)  # No draws in current implementation
                    match_execution.add_variable('_seed', seed)
                    match_execution.add_variable('_model_name', model_name)
                    match_execution.add_variable('_model_temperature', temperature)
                    match_execution.add_variable('_model_top_p', top_p)
                    result_executions.append(match_execution)

                # Add final Elo ratings for this (model, seed)
                for competitor, rating in ratings.items():
                    rating_execution = base_execution.copy()
                    rating_execution.add_variable('_elo_competitor', competitor)
                    rating_execution.add_variable('_elo_rating', int(rating))
                    rating_execution.add_variable('_elo_wins', wins[competitor])
                    rating_execution.add_variable('_elo_loss', losses[competitor])
                    rating_execution.add_variable('_elo_draws', 0)  # No draws in current implementation
                    rating_execution.add_variable('_seed', seed)
                    rating_execution.add_variable('_model_name', model_name)
                    rating_execution.add_variable('_model_temperature', temperature)
                    rating_execution.add_variable('_model_top_p', top_p)
                    result_executions.append(rating_execution)

        return result_executions 