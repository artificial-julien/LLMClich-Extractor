from typing import Optional
from dataclasses import dataclass
@dataclass
class PipelineConfig:
    """Configuration class for pipeline processing."""
    json_path: str
    output_dir: Optional[str] = None
    verbose: bool = False
    parallel: int = 1 
    seed: Optional[int] = None