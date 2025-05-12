from typing import Dict, Type, Any, Optional
from src.stage import Stage

class StageRegistry:
    """
    Registry for mapping node_type strings to Stage classes.
    
    This allows for dynamic loading of stage implementations based on the node_type
    specified in the configuration.
    """
    
    _registry: Dict[str, Type[Stage]] = {}
    
    @classmethod
    def register(cls, node_type: str) -> callable:
        """
        Decorator for registering a Stage class with a node_type.
        
        Args:
            node_type: The node_type identifier string
            
        Returns:
            Decorator function
        """
        def decorator(stage_class: Type[Stage]) -> Type[Stage]:
            cls._registry[node_type] = stage_class
            return stage_class
        return decorator
    
    @classmethod
    def get_stage_class(cls, node_type: str) -> Optional[Type[Stage]]:
        """
        Get the Stage class for a given node_type.
        
        Args:
            node_type: The node_type identifier string
            
        Returns:
            The Stage class or None if not registered
        """
        return cls._registry.get(node_type)
    
    @classmethod
    def create_stage(cls, config: Dict[str, Any]) -> Stage:
        """
        Create a Stage instance from a configuration dictionary.
        
        Args:
            config: The stage configuration dictionary, must contain 'node_type'
            
        Returns:
            An instance of the appropriate Stage class
            
        Raises:
            ValueError: If node_type is missing or not registered
        """
        node_type = config.get('node_type')
        if not node_type:
            raise ValueError("Configuration missing 'node_type' field")
        
        stage_class = cls.get_stage_class(node_type)
        if not stage_class:
            raise ValueError(f"Unknown node_type: {node_type}")
        
        return stage_class.from_config(config) 