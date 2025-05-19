from typing import List, Dict, Any, Optional
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry
from src.commons import PipelineConfig

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
            pipeline_config: PipelineConfig instance containing pipeline configuration
        """
        self.stages = stages
    
    @classmethod
    def from_config(cls, pipeline_definition: Dict[str, Any]) -> 'Pipeline':
        """
        Create a pipeline from a configuration dictionary.
        
        Args:
            pipeline_definition: Configuration dictionary with a 'foreach' list of stage configs
            
        Returns:
            A Pipeline instance
            
        Raises:
            ValueError: If the pipeline_config is invalid
        """
        foreach = pipeline_definition.get('foreach')
        if not foreach or not isinstance(foreach, list):
            raise ValueError("Configuration must contain a 'foreach' list")
        
        stages = []
        for stage_definition in foreach:
            stage = StageRegistry.create_stage(stage_definition)
            stages.append(stage)
        
        return cls(stages)
    
    def run(self, pipeline_config: PipelineConfig, initial_variables: Dict[str, Any]) -> List[Execution]:
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
            processed_executions = stage.process(executions=[exec for exec in executions if not exec.has_error()], pipeline_config=pipeline_config)
            
            errored_executions = [exec for exec in executions if exec.has_error()]
            executions = processed_executions + errored_executions
            
            if not executions:
                break
        
        return executions