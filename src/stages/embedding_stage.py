from typing import List, Dict, Any
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from src.stages.embedding.types import EmbeddingExecution
from src.llm.client import LLMClient
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class EmbeddingStage(Stage):
    """
    Stage that generates embeddings for a list of items.
    
    For each input execution, creates new Embedding executions for each item.
    This allows for processing multiple items with embeddings in parallel.
    Only processes executions that have embedding model configuration.
    """
    
    def __init__(self, items: List[str], is_second_dimension: bool = False, max_workers: int = 4):
        """
        Initialize the embedding stage with a list of items.
        
        Args:
            items: List of items (strings) to generate embeddings for
            is_second_dimension: Whether this is a second dimension embedding
            max_workers: Maximum number of parallel workers for embedding generation
        """
        self.items = items
        self.llm_client = LLMClient()
        self.is_second_dimension = is_second_dimension
        self.max_workers = max_workers
    
    def _generate_embedding(self, item: str, embedding_model: str, dimensions: int) -> List[float]:
        """
        Generate embedding for a single item using OpenAI's embedding API.
        
        Args:
            item: The item to generate embedding for
            embedding_model: The embedding model to use
            dimensions: Number of dimensions for the embedding
            
        Returns:
            List of floats representing the embedding vector
        """
        result = self.llm_client.generate_embedding(
            model=embedding_model,
            input_text=item,
            dimensions=dimensions
        )
        return result.embedding
    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions and create new Embedding executions for each item.
        
        Only processes executions that have embedding model configuration.
        Executions without embedding model config are passed through unchanged.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: List of input executions
            
        Returns:
            List containing new Embedding executions and unchanged executions
        """
        result_executions = []
        
        for base_execution in executions:
            result_executions.append(base_execution)
            if isinstance(base_execution, EmbeddingExecution):
                continue

            try:
                embedding_model = base_execution.embedding_model_config.name
                dimensions = base_execution.embedding_model_config.dimensions
                
                # Create progress bar for this execution's items
                with tqdm(total=len(self.items), desc=f"Generating embeddings for {embedding_model}") as pbar:
                    # Process items in parallel
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Submit all embedding generation tasks
                        future_to_item = {
                            executor.submit(
                                self._generate_embedding,
                                item,
                                embedding_model,
                                dimensions
                            ): item for item in self.items
                        }
                        
                        # Process completed tasks as they finish
                        for future in as_completed(future_to_item):
                            item = future_to_item[future]
                            try:
                                embedding_vector = future.result()
                                
                                embedding_execution = EmbeddingExecution(
                                    embedding_model_config=base_execution.embedding_model_config,
                                    item=item,
                                    embedding=embedding_vector,
                                    is_second_dimension=self.is_second_dimension
                                )
                                
                                embedding_execution.import_variables_from(base_execution)
                                result_executions.append(embedding_execution)
                                
                            except Exception as e:
                                error_execution = base_execution.copy()
                                error_execution.set_error(f"Failed to generate embedding for item '{item}': {str(e)}")
                                result_executions.append(error_execution)
                            
                            pbar.update(1)
                    
            except Exception as e:
                error_execution = base_execution.copy()
                error_execution.set_error(f"Failed to generate embeddings: {str(e)}")
                result_executions.append(error_execution)
        
        return result_executions 