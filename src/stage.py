from typing import List, Dict, Any, Protocol, Iterator
from abc import ABC, abstractmethod
from src.execution import Execution
from src.common.types import *
from itertools import tee

import src.common.script_runner as script_runner

class Stage(ABC):
    """
    Interface for all pipeline stages (node types).
    
    Each stage processes a list of input executions and returns a list of output executions.
    Stages can be composed using the pipe operator (|) to create composite stages.
    Stages can be composed in parallel using the & operator to create parallel stages.
    """
    
    @abstractmethod
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process input executions lazily and yield new executions.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            Execution instances as they are processed
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
    
    def __and__(self, other: 'Stage') -> 'Stage':
        """
        Compose this stage with another stage in parallel using the & operator.
        
        Args:
            other: The stage to compose with this one in parallel
            
        Returns:
            A ParallelStage that executes this stage and the other stage in parallel
        """
        return ParallelStage([self, other])
    
    def invoke(self, initial_variables: Dict[str, Any] = {}) -> List[Execution]:
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
        initial_executions = [Execution(variables=initial_variables)]
        
        # Process through this stage using lazy evaluation
        result_executions = []
        
        for execution in self.process(pipeline_config=script_runner.global_config, executions=iter(initial_executions)):
            result_executions.append(execution)
            
            # Report errors as they occur
            if script_runner.global_config.verbose and execution.has_error():
                print(f"Stage {self.__class__.__name__} had error: {execution.error}")
        
        return result_executions


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
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process executions through all stages in sequence using lazy evaluation.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input execution contexts
            
        Yields:
            Execution instances as they are processed through all stages
        """
        current_executions = executions
        
        # Process through each stage
        for stage in self.stages:
            # Use lazy evaluation for each stage
            current_executions = stage.process(pipeline_config=script_runner.global_config, executions=current_executions)
        
        # Yield all final executions
        for execution in current_executions:
            yield execution
    
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


class ParallelStage(Stage):
    """
    A parallel stage that executes multiple stages concurrently.
    
    This allows for parallel stage composition using the & operator.
    Each input execution is processed by all stages in parallel, and their outputs are combined.
    """
    
    def __init__(self, stages: List[Stage]):
        """
        Initialize a parallel stage with a list of stages.
        
        Args:
            stages: List of Stage instances to be executed in parallel
        """
        self.stages = stages
    
    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process executions through all stages in parallel using lazy evaluation.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input execution contexts
            
        Yields:
            Execution instances as they are processed through all stages in parallel
        """
        # Create independent iterators for each stage
        stage_iterators = tee(executions, len(self.stages))
        
        # Process each stage with its own iterator
        for stage, stage_iterator in zip(self.stages, stage_iterators):
            # Process each execution through this stage
            for result in stage.process(pipeline_config=script_runner.global_config, executions=stage_iterator):
                yield result
    
    def __and__(self, other: 'Stage') -> 'Stage':
        """
        Compose this parallel stage with another stage in parallel.
        
        Args:
            other: The stage to compose with this parallel stage
            
        Returns:
            A new ParallelStage with the additional stage added
        """
        if isinstance(other, ParallelStage):
            return ParallelStage(self.stages + other.stages)
        else:
            return ParallelStage(self.stages + [other]) 