from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    """Configuration class for pipeline processing."""
    output_dir: Optional[str] = None
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
