from typing import List, Dict, Any
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig, EmbeddingModelConfig

class EmbeddingModelsStage(Stage):
    """
    Stage that processes executions with multiple embedding model configurations.
    
    For each input execution, creates new executions for each embedding model configuration.
    This allows for parallel processing of the same input with different embedding models.
    """
    
    def __init__(self, embedding_models: List[EmbeddingModelConfig]):
        """
        Initialize the embedding models stage with a list of embedding model configurations.
        
        Args:
            embedding_models: List of EmbeddingModelConfig instances to process with
        """
        self.embedding_models = embedding_models
    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions and create new executions for each embedding model configuration.
        
        For each input execution, creates multiple new executions with different
        embedding model configurations while preserving the original variables.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: List of input executions
            
        Returns:
            List of new executions, one for each embedding model configuration
        """
        result_executions : List[Execution] = []
        
        for execution in executions:
            for embedding_model in self.embedding_models:
                new_execution = execution.copy()
                new_execution.embedding_model_config = embedding_model
                result_executions.append(new_execution)
        
        return result_executions 