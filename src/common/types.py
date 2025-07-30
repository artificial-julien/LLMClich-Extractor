from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    """Configuration class for pipeline processing."""
    output_dir: str
    verbose: bool = False
    parallel: int = 1 
    llm_max_tries: int = 1
    llm_seed: Optional[int] = None
    batch_seed: Optional[int] = None
    csv_append: bool = False  # Controls whether to append to existing CSV files overwise it will overwrite the file
    custom_args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    iterations: int = 1

@dataclass
class EmbeddingModelConfig:
    name: str
    dimensions: int = field(init=False)
    
    def __post_init__(self):
        """Set default dimensions based on model name."""
        if "ada" in self.name:
            self.dimensions = 1536
        elif "3-small" in self.name:
            self.dimensions = 1536
        elif "3-large" in self.name:
            self.dimensions = 3072
        else:
            # Default to 1536 for unknown models
            self.dimensions = 1536

    def __hash__(self) -> int:
        """Make the class hashable based on name only."""
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        """Compare instances based on name only."""
        if not isinstance(other, EmbeddingModelConfig):
            return NotImplemented
        return self.name == other.name
