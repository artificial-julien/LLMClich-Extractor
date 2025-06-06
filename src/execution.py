from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.common.types import *

@dataclass(kw_only=True)
class Execution(ABC):
    """
    Represents an execution context that contains variables and error state.
    Abstract base class for all execution types in the system.
    
    Each execution contains:
    - A dictionary of variables
    - An optional error state
    - A model configuration
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    model_config: Optional[ModelConfig] = None
    embedding_model_config: Optional[EmbeddingModelConfig] = None
    error: Optional[str] = None
    
    def get_specific_variables(self) -> Dict[str, Any]:
        """Get variables specific to the implementing class. By default returns an empty dictionary."""
        return {}

    def get_all_variables(self) -> Dict[str, Any]:
        """Convert execution properties to variables in specific order."""
        variables = {}
        variables['_error'] = self.error
        if self.model_config:
            variables.update({
                '_model_name': self.model_config.name,
                '_model_temperature': self.model_config.temperature,
                '_model_top_p': self.model_config.top_p,
                '_model_iterations': self.model_config.iterations,
            })
        variables.update(self.get_specific_variables())
        variables.update(self.variables)
        return variables
    
    def copy(self) -> 'Execution':
        """Create a copy of this execution with the same variables and error state."""
        return Execution(model_config=self.model_config, variables=self.variables.copy(), error=self.error)
    
    def add_variable(self, key: str, value: Any) -> None:
        """Add or update a variable in the execution context."""
        self.variables[key] = value
    
    def add_variables(self, variables: Dict[str, Any]) -> None:
        """Add or update multiple variables in the execution context."""
        self.variables.update(variables)
    
    def import_variables_from(self, other: 'Execution', variable_keys: Optional[List[str]] = None) -> None:
        """
        Import specific variables from another Execution instance.
        
        Args:
            other: The Execution instance to import variables from
            variable_keys: Optional list of specific variable keys to import. If None, imports all variables.
        """
        if variable_keys is None:
            variable_keys = list(other.variables.keys())
        
        for key in variable_keys:
            if key in other.variables:
                self.variables[key] = other.variables[key]
    
    def set_error(self, error: str) -> None:
        """Set the error state of this execution."""
        self.error = error
    
    def has_error(self) -> bool:
        """Check if this execution has an error."""
        return self.error is not None
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the execution context with an optional default value."""
        return self.get_all_variables().get(key, default)
    
    def __str__(self) -> str:
        """String representation of the execution."""
        if self.has_error():
            return f"Execution(error={self.error}, variables={self.variables})"
        return f"Execution(variables={self.variables})" 