"""
Preprocessing module with modular EEG preprocessing pipelines.

Usage:
    from data_loader import get_data_loader
    from preprocess import MNEPreprocessor
    
    # Load raw data
    loader = get_data_loader('zl')
    data = loader.load_subject_data('sub-PD089', 'ses-S001')
    
    # Preprocess
    preprocessor = MNEPreprocessor(
        data['eeg'], 
        data['eeg_timestamps'],
        data['markers'],
        data['marker_timestamps'],
        channel_labels=data['channel_labels'],
        sampling_rate=data['sampling_rate']
    )
    
    # Get fully processed epochs
    epochs, labels, metadata = preprocessor.get_processed_epochs()
"""

from .base import EEGPreprocessor
from .mne_preprocessor import MNEPreprocessor

__all__ = [
    'EEGPreprocessor',
    'MNEPreprocessor',
]
