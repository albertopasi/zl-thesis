"""
Base preprocessing interface for EEG data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class EEGPreprocessor(ABC):
    """Abstract base class for EEG preprocessing."""
    
    def __init__(self, eeg_data: np.ndarray, 
                 eeg_timestamps: np.ndarray,
                 markers: List[str],
                 marker_timestamps: np.ndarray,
                 channel_labels: List[str] = None,
                 sampling_rate: float = 500.0):
        """
        Initialize preprocessor.
        
        Args:
            eeg_data: np.ndarray of shape (samples, channels)
            eeg_timestamps: np.ndarray of EEG timestamps
            markers: list of marker strings
            marker_timestamps: np.ndarray of marker timestamps
            channel_labels: list of channel labels
            sampling_rate: sampling rate in Hz
        """
        self.eeg_data = eeg_data
        self.eeg_timestamps = eeg_timestamps
        self.markers = markers
        self.marker_timestamps = marker_timestamps
        self.channel_labels = channel_labels or [f"CH{i}" for i in range(eeg_data.shape[1])]
        self.sampling_rate = sampling_rate
        
        self.raw = None
        self.epochs = None
        self.epoch_labels = []
        self.epoch_metadata = []
    
    @abstractmethod
    def preprocess(self, **kwargs) -> np.ndarray:
        """
        Apply preprocessing to EEG data.
        
        Returns:
            np.ndarray: Preprocessed EEG data
        """
        pass
    
    @abstractmethod
    def extract_epochs(self, tmin: float = -1.5, tmax: float = 1.5) -> Tuple[np.ndarray, List, List]:
        """
        Extract epochs around marker events.
        
        Args:
            tmin: Start time relative to marker (seconds)
            tmax: End time relative to marker (seconds)
            
        Returns:
            tuple: (epochs_data, labels, metadata)
        """
        pass
    
    @abstractmethod
    def normalize_epochs(self, method: str = 'zscore') -> np.ndarray:
        """
        Normalize extracted epochs.
        
        Args:
            method: Normalization method ('zscore', 'minmax', etc.)
            
        Returns:
            np.ndarray: Normalized epochs
        """
        pass
    
    def get_processed_epochs(self, **kwargs) -> Tuple[np.ndarray, List, List]:
        """
        Get fully processed epochs (preprocess -> extract -> normalize).
        
        Returns:
            tuple: (epochs, labels, metadata)
        """
        self.preprocess(**kwargs)
        epochs, labels, metadata = self.extract_epochs(**kwargs)
        epochs = self.normalize_epochs(**kwargs)
        return epochs, labels, metadata
