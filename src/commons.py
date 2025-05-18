from typing import Optional
from dataclasses import dataclass
@dataclass
class PipelineConfig:
    """Configuration class for pipeline processing."""
    json_path: str
    output_dir: Optional[str] = None
    verbose: bool = False
    parallel: int = 1 
    llm_seed: Optional[int] = None
    batch_seed: Optional[int] = None