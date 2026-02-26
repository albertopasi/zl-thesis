# THU-EP Dataset Module
"""THU-EP dataset processing module.

Submodules:
- exploration: Data exploration utilities
- preprocessing: EEG preprocessing pipeline
"""

from .config import THUEPConfig, get_config
from .preprocessing import THUEPPreprocessingPipeline

__all__ = [
    "THUEPConfig",
    "get_config",
    "THUEPPreprocessingPipeline",
]
