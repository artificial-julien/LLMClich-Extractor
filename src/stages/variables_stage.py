from typing import List, Dict, Any
from src.stage import Stage
from src.execution import Execution
from src.registry import StageRegistry

@StageRegistry.register("variables")
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
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VariablesStage':
        """
        Create a VariablesStage from configuration.
        
        Args:
            config: Dictionary containing 'list' key with variable sets
            
        Returns:
            A VariablesStage instance
            
        Raises:
            ValueError: If the config is invalid
        """
        variable_list = config.get('list')
        if not variable_list or not isinstance(variable_list, list):
            raise ValueError("VariablesStage config must contain a 'list' of variable sets")
        
        return cls(variable_sets=variable_list)
    
    def process(self, executions: List[Execution]) -> List[Execution]:
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
        executions = [exec for exec in executions if not exec.has_error()]
        
        for execution in executions:
            for variable_set in self.variable_sets:
                # Create a copy of the input execution
                new_execution = execution.copy()
                
                # Add the variables from this set
                new_execution.add_variables(variable_set)
                
                # Add to the result list
                result_executions.append(new_execution)
        
        return result_executions 