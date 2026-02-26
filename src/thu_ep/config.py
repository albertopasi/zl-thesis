"""
THU-EP configuration loader.

Provides access to the thu_ep.yml configuration values for use across
preprocessing and exploration modules.
"""

from pathlib import Path
from typing import List, Dict, Any

import yaml


def get_config_path() -> Path:
    """Get the path to the thu_ep.yml config file."""
    return Path(__file__).parent.parent.parent / "configs" / "thu_ep.yml"


def load_config() -> Dict[str, Any]:
    """
    Load the THU-EP configuration from YAML.
    
    Returns:
        Dictionary containing all configuration values.
    """
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class THUEPConfig:
    """
    Provides access to THU-EP configuration values.
    
    This class loads the configuration once and provides properties
    for accessing commonly used values.
    
    Usage:
        config = THUEPConfig()
        print(config.raw_data_dir)
        print(config.all_channels)
    """
    
    def __init__(self, config_path: Path = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Optional path to config file. Uses default if None.
        """
        if config_path is None:
            config_path = get_config_path()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        self._project_root = config_path.parent.parent
    
    # =========================================================================
    # Path properties (returns Path objects relative to project root)
    
    @property
    def raw_data_dir(self) -> Path:
        """Path to raw EEG data directory."""
        return self._project_root / self._config['paths']['raw_data_dir']
    
    @property
    def ratings_dir(self) -> Path:
        """Path to ratings directory."""
        return self._project_root / self._config['paths']['ratings_dir']
    
    @property
    def others_dir(self) -> Path:
        """Path to others directory (contains label.mat)."""
        return self._project_root / self._config['paths']['others_dir']
    
    @property
    def preprocessed_dir(self) -> Path:
        """Path to preprocessed output directory."""
        return self._project_root / self._config['paths']['preprocessed_output_dir']
    
    # =========================================================================
    # Channel properties
    
    @property
    def all_channels(self) -> List[str]:
        """List of all 32 channel names in order."""
        return self._config['channels']['all_channels'].copy()
    
    @property
    def channels_to_remove(self) -> List[str]:
        """List of reference channels to remove during preprocessing."""
        return self._config['channels']['channels_to_remove'].copy()
    
    @property
    def final_channels(self) -> List[str]:
        """List of channels after preprocessing (excluding removed channels)."""
        return [ch for ch in self.all_channels if ch not in self.channels_to_remove]
    
    # =========================================================================
    # Band properties
    
    @property
    def band_names(self) -> List[str]:
        """List of frequency band names (indices 0-5)."""
        return self._config['bands']['names'].copy()
    
    @property
    def broad_band_index(self) -> int:
        """Index of broad-band frequency (0.5-47 Hz)."""
        return self._config['bands']['extract_band_index']
    
    # =========================================================================
    # Sampling properties
    
    @property
    def original_sfreq(self) -> float:
        """Original sampling frequency in Hz."""
        return self._config['sampling']['original_sfreq_hz']
    
    @property
    def target_sfreq(self) -> float:
        """Target sampling frequency in Hz after preprocessing."""
        return self._config['sampling']['target_sfreq_hz']
    
    @property
    def original_n_samples(self) -> int:
        """Number of samples per stimulus at original sampling rate."""
        return self._config['sampling']['original_n_samples']
    
    @property
    def target_n_samples(self) -> int:
        """Number of samples per stimulus after downsampling."""
        return self._config['sampling']['target_n_samples']
    
    # =========================================================================
    # Dataset properties
    
    @property
    def n_subjects(self) -> int:
        """Number of subjects in the dataset."""
        return self._config['dataset']['n_subjects']
    
    @property
    def n_stimuli(self) -> int:
        """Number of stimuli per subject."""
        return self._config['dataset']['n_stimuli']
    
    @property
    def n_channels(self) -> int:
        """Number of original channels."""
        return self._config['dataset']['n_channels']
    
    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        return self._config['dataset']['n_bands']
    
    @property
    def expected_raw_shape(self) -> tuple:
        """Expected shape of raw data: (samples, channels, stimuli, bands)."""
        return tuple(self._config['dataset']['expected_raw_shape'])
    
    @property
    def expected_preprocessed_shape(self) -> tuple:
        """Expected shape of preprocessed data: (stimuli, channels, samples)."""
        return tuple(self._config['dataset']['expected_preprocessed_shape'])
    
    # =========================================================================
    # Preprocessing properties
    
    @property
    def artifact_threshold_std(self) -> float:
        """Artifact clipping threshold in standard deviations."""
        return self._config['preprocessing']['artifact_threshold_std']
    
    @property
    def steps_enabled(self) -> Dict[str, bool]:
        """Dictionary of preprocessing steps and whether they are enabled."""
        return self._config['preprocessing']['steps_enabled'].copy()
    
    @property
    def verbose(self) -> bool:
        """Whether to print verbose output."""
        return self._config['options']['verbose']
    
    # =========================================================================
    # Computed properties (derived from config values)
    
    @property
    def channels_to_remove_indices(self) -> List[int]:
        """Indices of channels to remove during preprocessing."""
        all_ch = self.all_channels
        return [all_ch.index(ch) for ch in self.channels_to_remove if ch in all_ch]
    
    @property
    def n_channels_final(self) -> int:
        """Number of channels after preprocessing."""
        return len(self.final_channels)
    
    @property
    def downsample_factor(self) -> float:
        """Downsampling factor (original_sfreq / target_sfreq)."""
        return self.original_sfreq / self.target_sfreq
    
    # =========================================================================
    # Methods
    
    def is_step_enabled(self, step_name: str) -> bool:
        """Check if a preprocessing step is enabled."""
        return self._config['preprocessing']['steps_enabled'].get(step_name, False)


# Module-level singleton for convenience
_config_instance: THUEPConfig = None


def get_config() -> THUEPConfig:
    """
    Get the singleton THUEPConfig instance.
    
    Returns:
        THUEPConfig instance with loaded configuration.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = THUEPConfig()
    return _config_instance
