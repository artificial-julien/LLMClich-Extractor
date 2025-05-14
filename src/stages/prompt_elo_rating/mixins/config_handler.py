from typing import List, Dict, Any
from ..types import ModelConfig, DEFAULT_INITIAL_RATING

class ConfigHandlerMixin:
    """Mixin providing configuration handling functionality."""
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """
        Validate the stage configuration.
        
        Args:
            config: Dictionary containing stage configuration
            
        Raises:
            ValueError: If the config is invalid
        """
        models = config.get('models')
        competitors = config.get('competitors')
        prompts = config.get('prompts')
        
        if not models or not isinstance(models, list):
            raise ValueError("PromptEloRatingStage config must contain a 'models' list")
        
        if not competitors or not isinstance(competitors, list):
            raise ValueError("PromptEloRatingStage config must contain a 'competitors' list")
        
        if not prompts or not isinstance(prompts, list):
            raise ValueError("PromptEloRatingStage config must contain a 'prompts' list")
    
    @classmethod
    def parse_model_config(cls, model_config: Dict[str, Any]) -> ModelConfig:
        """
        Parse a model configuration dictionary into a ModelConfig object.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            ModelConfig object
        """
        return ModelConfig(
            name=model_config['name'],
            temperature=float(model_config.get('temperature', 0.0)),
            top_p=float(model_config.get('top_p', 1.0)),
            iterations=int(model_config.get('iterations', 1))
        )
    
    @classmethod
    def get_config_values(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate configuration values.
        
        Args:
            config: Stage configuration dictionary
            
        Returns:
            Dictionary of configuration values
        """
        cls.validate_config(config)
        
        return {
            'models': [cls.parse_model_config(m) for m in config['models']],
            'competitors': config['competitors'],
            'prompts': config['prompts'],
            'batches_per_model': config.get('batches_per_model', 4),
            'initial_rating': config.get('initial_rating', DEFAULT_INITIAL_RATING),
            'symmetric_matches': config.get('symmetric_matches', False),
            'parallel': config.get('parallel', 2)
        } 