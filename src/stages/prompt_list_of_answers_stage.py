from typing import List, Dict, Any, Optional
import os
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
from src.llm import LLMClient
from src.commons import PipelineConfig
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.prompt_exception import LLMException
from src.execution import ModelConfig

@StageRegistry.register("prompt_list_of_answers")
class PromptListOfAnswersStage(Stage):
    """
    Stage that sends prompts to LLMs and constrains responses to a list of possible answers.
    """
    
    def __init__(
        self,
        models: List[Dict[str, Any]],
        prompts: List[str],
        possible_answers: List[str],
        result_var_name: str
    ):
        """
        Initialize the prompt list of answers stage.
        
        Args:
            models: List of model configurations (name, temperature, iterations, etc.)
            prompts: List of prompt templates
            possible_answers: List of allowed answers
            result_var_name: Variable name to store the result
        """
        self.models = [ModelConfig.from_dict(model) for model in models]
        self.prompts = prompts
        self.possible_answers = possible_answers
        self.result_var_name = result_var_name
        self.llm_client = LLMClient()
    
    @classmethod
    def from_dict(cls, stage_definition: Dict[str, Any]) -> 'PromptListOfAnswersStage':
        """
        Create a PromptListOfAnswersStage from configuration.
        
        Args:
            stage_definition: Dictionary containing:
                - 'models': List of model configurations
                - 'prompts': List of prompt templates
                - 'possible_answers': List of allowed answers
                - 'result_var_name': Variable name to store the result
            
        Returns:
            A PromptListOfAnswersStage instance
            
        Raises:
            ValueError: If the stage_definition is invalid
        """
        models = stage_definition.get('models')
        prompts = stage_definition.get('prompts')
        possible_answers = stage_definition.get('possible_answers')
        result_var_name = stage_definition.get('result_var_name')
        
        if not models or not isinstance(models, list):
            raise ValueError("PromptListOfAnswersStage stage_definition must contain a 'models' list")
        
        if not prompts or not isinstance(prompts, list):
            raise ValueError("PromptListOfAnswersStage stage_definition must contain a 'prompts' list")
        
        if not possible_answers or not isinstance(possible_answers, list):
            raise ValueError("PromptListOfAnswersStage stage_definition must contain a 'possible_answers' list")
        
        if not result_var_name:
            raise ValueError("PromptListOfAnswersStage stage_definition must contain a 'result_var_name'")
        
        return cls(
            models=models,
            prompts=prompts,
            possible_answers=possible_answers,
            result_var_name=result_var_name
        )
    
    def _process_execution(
        self, 
        execution: Execution, 
        model_config: ModelConfig, 
        prompt_template: str,
        llm_seed: int,
        pipeline_config: PipelineConfig
    ) -> Execution:
        """
        Process a single execution with a specific model, prompt, and iteration.
        
        Args:
            execution: Input execution with variables
            model_config: Model configuration
            prompt_template: Prompt template
            iteration: Iteration number (for llm_seed)
            
        Returns:
            New execution with result variables
        """
        
        new_execution = execution.copy()
        new_execution.model_config = model_config
        new_execution.add_variable('llm_seed', llm_seed)

        # Format template with variables
        formatted = prompt_template
        for key, value in execution.get_all_variables().items():
            formatted = formatted.replace(f"[{key}]", str(value))
        
        # Add numbered list of possible answers
        answers_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(self.possible_answers)])
        formatted_prompt = f"{formatted}\n\nPossible answers:\n{answers_str}"
        try:
            result = self.llm_client.generate_constrained_completion(
                model=model_config.name,
                prompt=formatted_prompt,
                possible_answers=self.possible_answers,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                llm_seed=llm_seed,
                max_tries=pipeline_config.llm_max_tries
            )
        
            new_execution.add_variable(self.result_var_name, result.chosen_answer)

            for idx, answer in enumerate(result.probability_result.probabilities, start=0):
                prob_value = result.probability_result.probabilities.get(answer) if result.probability_result.probabilities else None
                new_execution.add_variable(f'prob_{idx+1}_{answer[:20]}', prob_value)
        except LLMException as e:
            new_execution.error = e.message
        
        return new_execution
    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions, send prompts to LLMs, and return new executions with results.
        
        Args:
            executions: List of input executions
            
        Returns:
            List of new executions with results
        """
        result_executions = []
        
        # Create all combinations of executions, models, prompts, and iterations
        jobs = []
        for execution in executions:
            for model_config in self.models:
                for prompt_template in self.prompts:
                    iterations = model_config.iterations
                    for iteration in range(iterations):
                        jobs.append((execution, model_config, prompt_template, iteration))
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=pipeline_config.parallel) as executor:
            # Create and submit futures
            futures = []
            for execution, model_config, prompt_template, iteration in jobs:
                future = executor.submit(
                    self._process_execution,
                    execution,
                    model_config,
                    prompt_template,
                    iteration,
                    pipeline_config
                )
                futures.append(future)
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing prompts"):
                try:
                    result_execution = future.result()
                    result_executions.append(result_execution)
                except Exception as e:
                    # Create an execution with error
                    base_execution = jobs[len(result_executions)][0]
                    error_execution = base_execution.copy()
                    error_execution.set_error(f"Error processing prompt: {str(e)}")
                    result_executions.append(error_execution)
        
        return result_executions 