"""Configuration for SEED EEG preprocessing pipeline and data exploration."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SEEDPreprocessingConfig:
    """Configuration for SEED EEG preprocessing pipeline"""
    
    # Paths
    seed_raw_dir: str = "data/SEED/SEED_RAW_EEG"
    montage_file: str = "data/SEED/channel_62_pos.locs"
    time_markers_file: str = "data/SEED/SEED_RAW_EEG/time.txt"
    preprocessed_output_dir: str = "data/SEED/preprocessed_seed"
    
    # Pipeline Steps (Enable/Disable)
    steps_enabled: Dict[str, bool] = field(default_factory=lambda: {
        'load_raw': True,
        'drop_non_eeg': True,
        'bandpass_filter': True,
        'downsample': True,
        'extract_movie_windows': True,
        'extract_last_30s': True,
        'z_normalize': True,
        'artifact_clipping': True,
        'export_npy': True,
    })
    
    # Drop Non-EEG Channels
    non_eeg_channels: List[str] = field(default_factory=lambda: [
        'M1', 'M2', 'VEO', 'HEO'
    ])
    
    # Bandpass Filter
    filter_lowcut_hz: float = 0.5
    filter_highcut_hz: float = 99.5
    
    # Downsampling
    downsample_freq_hz: int = 200  # Hz
    
    # Movie Window Extraction
    # Uses time.txt to extract windows
    
    # Last 30s Extraction
    window_duration_sec: float = 30.0  # Extract last 30s from each movie window
    
    # Z-Normalization
    # Applied per 30s window
    
    # Artifact Clipping
    artifact_threshold_std: float = 15.0  # Clip values exceeding this many std
    
    # Export Settings
    export_format: str = 'npy'  # 'npy' or 'npz'
    save_metadata: bool = True  # Save subject/session/trial info
    
    # Processing
    verbose: bool = True
    
    def __post_init__(self):
        """Expand paths to absolute if relative."""
        root = Path(__file__).parent.parent.parent
        
        paths_to_expand = [
            'seed_raw_dir',
            'montage_file',
            'time_markers_file',
            'preprocessed_output_dir'
        ]
        
        for path_attr in paths_to_expand:
            path_val = getattr(self, path_attr)
            if not Path(path_val).is_absolute():
                setattr(self, path_attr, str(root / path_val))
    
    def is_step_enabled(self, step_name: str) -> bool:
        """Check if a preprocessing step is enabled."""
        return self.steps_enabled.get(step_name, True)
    
    def enable_step(self, step_name: str):
        """Enable a preprocessing step."""
        self.steps_enabled[step_name] = True
    
    def disable_step(self, step_name: str):
        """Disable a preprocessing step."""
        self.steps_enabled[step_name] = False
    
    def enable_all_steps(self):
        """Enable all preprocessing steps."""
        for step in self.steps_enabled:
            self.steps_enabled[step] = True
    
    def disable_all_steps(self):
        """Disable all preprocessing steps."""
        for step in self.steps_enabled:
            self.steps_enabled[step] = False
