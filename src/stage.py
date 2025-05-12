from typing import List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from src.execution import Execution

class Stage(ABC):
    """
    Interface for all pipeline stages (node types).
    
    Each stage processes a list of input executions and returns a list of output executions.
    """
    
    @abstractmethod
    def process(self, executions: List[Execution]) -> List[Execution]:
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
    def from_config(cls, config: Dict[str, Any]) -> 'Stage':
        """
        Create a stage instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary for this stage
            
        Returns:
            An instance of the stage
        """
        pass 