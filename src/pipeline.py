from typing import List, Dict, Any
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry

class Pipeline:
    """
    A pipeline that processes data through a sequence of stages.
    
    The pipeline takes a configuration dictionary and initializes all the stages
    in the correct order, then provides methods to execute the pipeline.
    """
    
    def __init__(self, stages: List[Stage]):
        """
        Initialize a pipeline with a list of stages.
        
        Args:
            stages: List of Stage instances to be executed in order
        """
        self.stages = stages
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Pipeline':
        """
        Create a pipeline from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with a 'foreach' list of stage configs
            
        Returns:
            A Pipeline instance
            
        Raises:
            ValueError: If the config is invalid
        """
        foreach = config.get('foreach')
        if not foreach or not isinstance(foreach, list):
            raise ValueError("Configuration must contain a 'foreach' list")
        
        stages = []
        for stage_config in foreach:
            stage = StageRegistry.create_stage(stage_config)
            stages.append(stage)
        
        return cls(stages)
    
    def run(self, initial_variables: Dict[str, Any] = None) -> List[Execution]:
        """
        Run the pipeline with an optional set of initial variables.
        
        Args:
            initial_variables: Optional dictionary of initial variables
            
        Returns:
            List of resulting Execution instances after all stages are processed
        """
        # Initialize with a single execution containing the initial variables
        executions = [Execution(variables=initial_variables or {})]
        
        # Process through each stage
        for stage in self.stages:            
            processed_executions = stage.process([exec for exec in executions if not exec.has_error()])
            
            errored_executions = [exec for exec in executions if exec.has_error()]
            executions = processed_executions + errored_executions
            
            if not executions:
                break
        
        return executions 