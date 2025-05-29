from typing import List, Dict, Any
from src.stage import Stage
from src.execution import Execution
from src.common.types import *

class VariablesStage(Stage):
    """
    Stage that processes variable sets and creates multiple executions.
    
    For example, a variables node with:
    - 1 execution in entry 
    - 2 variables set A and B
    will return 
    - 2 executions, containing the variables of the original execution and enriched with A and B
    """
    
    def __init__(self, variable_sets: List[Dict[str, Any]]):
        """
        Initialize the variables stage with a list of variable sets.
        
        Args:
            variable_sets: List of dictionaries, each containing variable key-value pairs
        """
        self.variable_sets = variable_sets
    

    
    def process(self, pipeline_config: PipelineConfig, executions: List[Execution]) -> List[Execution]:
        """
        Process input executions and create new executions with the variable sets.
        
        For each input execution, creates multiple new executions with variables
        from the original execution plus the variables from each variable set.
        
        Args:
            executions: List of input executions
            
        Returns:
            List of new executions with enriched variables
        """
        result_executions = []
        
        for execution in executions:
            for variable_set in self.variable_sets:
                new_execution = execution.copy()
                
                # Add the variables from this set
                new_execution.add_variables(variable_set)
                
                result_executions.append(new_execution)
        
        return result_executions 