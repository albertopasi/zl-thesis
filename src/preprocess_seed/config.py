"""Configuration for SEED data exploration."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class SEEDConfig:
    """Configuration for SEED data exploration."""
    
    # Paths
    seed_raw_dir: str = "data/SEED/SEED_RAW_EEG"
    montage_file: str = "data/SEED/channel_62_pos.locs"
    
    def __post_init__(self):
        """Expand paths to absolute if relative."""
        root = Path(__file__).parent.parent.parent
        
        if not Path(self.seed_raw_dir).is_absolute():
            self.seed_raw_dir = str(root / self.seed_raw_dir)
        
        if not Path(self.montage_file).is_absolute():
            self.montage_file = str(root / self.montage_file)
