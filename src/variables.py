from typing import Dict, Any, List, Optional

class Variables:
    """
    Class for storing and managing variables in the pipeline.
    Supports initialization from a dict or list of dicts, updating, and retrieval.
    """
    def __init__(self, initial: Optional[List[Dict[str, Any]]] = None):
        self._variables: Dict[str, Any] = {}
        if initial:
            for item in initial:
                self.update(item)

    def update(self, new_vars: Dict[str, Any]):
        """Update the variables with a new dictionary."""
        self._variables.update(new_vars)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable by key, with optional default."""
        return self._variables.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Return a copy of the variables as a dictionary."""
        return dict(self._variables)

    def __getitem__(self, key: str) -> Any:
        return self._variables[key]

    def __setitem__(self, key: str, value: Any):
        self._variables[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._variables

    def __repr__(self):
        return f"Variables({self._variables})" 