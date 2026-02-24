"""
Configuration dataclass for THU-EP preprocessing pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class THUEPPreprocessingConfig:
    """Configuration for THU-EP EEG preprocessing pipeline.
    
    THU-EP dataset structure:
    - 80 subjects
    - 28 stimuli per subject (film clips)
    - 32 EEG channels
    - 6 frequency bands per channel
    - 250 Hz sampling rate
    - 7500 samples per stimulus (30 seconds)
    
    Preprocessing steps (Phase 1 from plan.md):
    1. Remove A1, A2 reference channels (32 -> 30 channels)
    2. Extract broad-band (band index 5 = 0.5-47 Hz)
    3. Downsample 250 Hz -> 200 Hz (7500 -> 6000 samples)
    4. Global Z-score normalization per channel (across all stimuli)
    5. Artifact clipping at ±15 standard deviations
    6. Export as .npy files
    
    Output per subject: (28, 30, 6000) = (stimuli, channels, samples)
    """
    
    # Paths
    raw_data_dir: str = "data/thu ep/EEG data"
    ratings_dir: str = "data/thu ep/Ratings"
    preprocessed_output_dir: str = "data/thu ep/preprocessed"
    
    # Channel configuration
    all_channel_names: List[str] = field(default_factory=lambda: [
        "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
        "Cz", "C3", "C4", "T7", "T8", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
        "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2"
    ])
    channels_to_remove: List[str] = field(default_factory=lambda: ["A1", "A2"])
    
    # Band configuration
    # Bands: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma, 5=broad-band
    band_names: List[str] = field(default_factory=lambda: [
        "delta", "theta", "alpha", "beta", "gamma", "broad-band"
    ])
    extract_band_index: int = 5  # broad-band (0.5-47 Hz)
    
    # Sampling configuration
    original_sfreq_hz: float = 250.0
    target_sfreq_hz: float = 200.0
    original_n_samples: int = 7500  # 30 seconds at 250 Hz
    target_n_samples: int = 6000    # 30 seconds at 200 Hz
    
    # Normalization
    artifact_threshold_std: float = 15.0
    
    # Processing control
    steps_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "remove_reference_channels": True,
        "extract_band": True,
        "downsample": True,
        "z_normalize": True,
        "artifact_clipping": True,
        "export_npy": True,
    })
    
    # Output options
    verbose: bool = True
    n_stimuli: int = 28
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths are Path objects internally
        self._raw_data_path = Path(self.raw_data_dir)
        self._ratings_path = Path(self.ratings_dir)
        self._output_path = Path(self.preprocessed_output_dir)
        
        # Compute channel indices to remove
        self.channels_to_remove_indices = [
            self.all_channel_names.index(ch) 
            for ch in self.channels_to_remove
            if ch in self.all_channel_names
        ]
        
        # Compute final channel names after removal
        self.final_channel_names = [
            ch for ch in self.all_channel_names 
            if ch not in self.channels_to_remove
        ]
    
    def is_step_enabled(self, step_name: str) -> bool:
        """Check if a preprocessing step is enabled."""
        return self.steps_enabled.get(step_name, False)
    
    @property
    def raw_data_path(self) -> Path:
        return self._raw_data_path
    
    @property
    def ratings_path(self) -> Path:
        return self._ratings_path
    
    @property
    def output_path(self) -> Path:
        return self._output_path
    
    @property
    def n_channels_original(self) -> int:
        return len(self.all_channel_names)
    
    @property
    def n_channels_final(self) -> int:
        return len(self.final_channel_names)
    
    @property
    def downsample_factor(self) -> float:
        return self.original_sfreq_hz / self.target_sfreq_hz
