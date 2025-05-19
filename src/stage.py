from typing import List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from src.execution import Execution
from src.commons import PipelineConfig
class Stage(ABC):
    """
    Interface for all pipeline stages (node types).
    
    Each stage processes a list of input executions and returns a list of output executions.
    """
    
    @abstractmethod
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process the input executions and return a list of new executions.
        
        Args:
            executions: List of input execution contexts
            
        Returns:
            List of resulting execution contexts
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, stage_definition: Dict[str, Any]) -> 'Stage':
        """
        Create a stage instance from configuration dictionary.
        
        Args:
            stage_definition: Configuration dictionary for this stage
            
        Returns:
            An instance of the stage
        """
        pass 