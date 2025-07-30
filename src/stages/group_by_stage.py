from typing import List, Dict, Any, Iterator, Callable
from src.stage import Stage
from src.execution import Execution
from src.common.types import PipelineConfig
from itertools import groupby
from operator import attrgetter

class GroupByStage(Stage):
    """
    Stage that groups executions by a key, processes each group through a pipeline,
    and aggregates the results.
    """
    
    def __init__(self, 
                 key: str,
                 pipeline: Stage,
                 group_key_getter: Callable[[Execution], Any] = None):
        """
        Initialize the group by stage.
        
        Args:
            key: The key to group executions by (can be a variable name or attribute)
            pipeline: The pipeline to process each group with
            group_key_getter: Optional function to extract the group key from an execution
        """
        self.key = key
        self.pipeline = pipeline
        self.group_key_getter = group_key_getter or (lambda x: x.get_variable(key))

    def process(self, pipeline_config: PipelineConfig, executions: Iterator[Execution]) -> Iterator[Execution]:
        """
        Process executions by grouping them, running each group through the pipeline,
        and yielding the results.
        
        Args:
            pipeline_config: Configuration for the pipeline execution
            executions: Iterator of input executions
            
        Yields:
            Processed executions from each group
        """
        # Collect all executions
        executions_list = list(executions)
        
        # Group executions by the key
        groups = groupby(executions_list, key=self.group_key_getter)
        
        # Process each group through the pipeline
        for group_key, group_executions in groups:
            # Convert group iterator to list since we need to process it multiple times
            group_list = list(group_executions)
            
            # Process the group through the pipeline
            for result in self.pipeline.process(pipeline_config, iter(group_list)):
                yield result