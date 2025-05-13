from typing import List, Dict, Any, Optional
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry

class Pipeline:
    """
    A pipeline that processes data through a sequence of stages.
    
    The pipeline takes a configuration dictionary and initializes all the stages
    in the correct order, then provides methods to execute the pipeline.
    """
    
    def __init__(self, stages: List[Stage], input_folder: Optional[str] = None):
        """
        Initialize a pipeline with a list of stages.
        
        Args:
            stages: List of Stage instances to be executed in order
            input_folder: Optional path to the input folder
        """
        self.stages = stages
        self.input_folder = input_folder
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], input_folder: Optional[str] = None) -> 'Pipeline':
        """
        Create a pipeline from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with a 'foreach' list of stage configs
            input_folder: Optional path to the input folder
            
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
            # Add input folder to stage config if it's an export_to_csv stage
            if stage_config.get('node_type') == 'export_to_csv':
                stage_config['input_folder'] = input_folder
            stage = StageRegistry.create_stage(stage_config)
            stages.append(stage)
        
        return cls(stages, input_folder)
    
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
            
            executions = stage.process(executions)
            
            if not executions:
                break
        
        return executions 