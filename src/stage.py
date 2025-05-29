from typing import List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from src.execution import Execution
from src.common.types import *

import src.common.script_runner as script_runner

class Stage(ABC):
    """
    Interface for all pipeline stages (node types).
    
    Each stage processes a list of input executions and returns a list of output executions.
    Stages can be composed using the pipe operator (|) to create composite stages.
    """
    
    @abstractmethod
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process the input executions and return a list of new executions.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: List of input execution contexts
            
        Returns:
            List of resulting execution contexts
        """
        pass
    
    def __or__(self, other: 'Stage') -> 'Stage':
        """
        Compose this stage with another stage using the pipe operator.
        
        Args:
            other: The stage to compose with this one
            
        Returns:
            A CompositeStage that executes this stage followed by the other stage
        """
        return CompositeStage([self, other])
    
    def invoke(self, initial_variables: Dict[str, Any] = None) -> List[Execution]:
        """
        Execute this stage (or composite stage) with initial variables.
        
        Args:
            initial_variables: Optional dictionary of initial variables
            
        Returns:
            List of resulting Execution instances after processing
        """
        # Check if global_config is available
        if script_runner.global_config is None:
            raise RuntimeError("global_config is not initialized. Call script_runner.load_arguments() first.")
            
        # Initialize with a single execution containing the initial variables
        executions = [Execution(variables=initial_variables or {})]
        
        # Process through this stage
        processed_executions = self.process(pipeline_config=script_runner.global_config, executions=[exec for exec in executions if not exec.has_error()])
        
        errored_executions = [exec for exec in executions if exec.has_error()]
        executions = processed_executions + errored_executions

        # Report any errors that occurred during stage processing
        if script_runner.global_config.verbose:
            errored = [exec for exec in executions if exec.has_error()]
            if errored:
                print(f"Stage {self.__class__.__name__} had {len(errored)} errors:")
                for exec in errored:
                    print(f"  Error: {exec.error}")
        
        return executions


class CompositeStage(Stage):
    """
    A composite stage that executes multiple stages in sequence.
    
    This allows for stage composition using the pipe operator.
    """
    
    def __init__(self, stages: List[Stage]):
        """
        Initialize a composite stage with a list of stages.
        
        Args:
            stages: List of Stage instances to be executed in order
        """
        self.stages = stages
    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process executions through all stages in sequence.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: List of input execution contexts
            
        Returns:
            List of resulting execution contexts after all stages are processed
        """
        current_executions = executions
        
        # Process through each stage
        for stage in self.stages:            
            processed_executions = stage.process(pipeline_config=pipeline_config, executions=[exec for exec in current_executions if not exec.has_error()])
            
            errored_executions = [exec for exec in current_executions if exec.has_error()]
            current_executions = processed_executions + errored_executions

            # Report any errors that occurred during stage processing
            if pipeline_config.verbose:
                errored = [exec for exec in current_executions if exec.has_error()]
                if errored:
                    print(f"Stage {stage.__class__.__name__} had {len(errored)} errors:")
                    for exec in errored:
                        print(f"  Error: {exec.error}")
            
            if not current_executions:
                break
        
        return current_executions
    
    def __or__(self, other: 'Stage') -> 'Stage':
        """
        Compose this composite stage with another stage.
        
        Args:
            other: The stage to compose with this composite stage
            
        Returns:
            A new CompositeStage with the additional stage appended
        """
        if isinstance(other, CompositeStage):
            return CompositeStage(self.stages + other.stages)
        else:
            return CompositeStage(self.stages + [other]) 