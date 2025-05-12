from typing import Dict, Any, List, Optional

class Execution:
    """
    Represents an execution context that contains variables and error state.
    
    Each execution contains:
    - A dictionary of variables
    - An optional error state
    """
    
    def __init__(self, variables: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """
        Initialize an execution context.
        
        Args:
            variables: Initial variables dictionary
            error: Optional error message
        """
        self.variables = variables or {}
        self.error = error
    
    def copy(self) -> 'Execution':
        """Create a copy of this execution with the same variables and error state."""
        return Execution(variables=self.variables.copy(), error=self.error)
    
    def add_variable(self, key: str, value: Any) -> None:
        """Add or update a variable in the execution context."""
        self.variables[key] = value
    
    def add_variables(self, variables: Dict[str, Any]) -> None:
        """Add or update multiple variables in the execution context."""
        self.variables.update(variables)
    
    def set_error(self, error: str) -> None:
        """Set the error state of this execution."""
        self.error = error
    
    def has_error(self) -> bool:
        """Check if this execution has an error."""
        return self.error is not None
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the execution context with an optional default value."""
        return self.variables.get(key, default)
    
    def __str__(self) -> str:
        """String representation of the execution."""
        if self.has_error():
            return f"Execution(error={self.error}, variables={self.variables})"
        return f"Execution(variables={self.variables})" 