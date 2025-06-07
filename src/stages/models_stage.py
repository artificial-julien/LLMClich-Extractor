from typing import List, Dict, Any, Iterator
from src.stage import Stage
from src.execution import Execution
from src.common.types import *

class ModelsStage(Stage):
    """
    Stage that processes executions with multiple model configurations.
    
    For each input execution, creates new executions for each model configuration.
    This allows for parallel processing of the same input with different models.
    """
    
    def __init__(self, models: List[ModelConfig]):
        """
        Initialize the models stage with a list of model configurations.
        
        Args:
            models: List of ModelConfig instances to process with
        """
        self.models = models
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process input executions lazily and create new executions for each model configuration.
        
        For each input execution, creates multiple new executions with different
        model configurations while preserving the original variables.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            New executions, one for each model configuration
        """
        for execution in executions:
            for model_config in self.models:
                new_execution = execution.copy()
                new_execution.model_config = model_config
                yield new_execution