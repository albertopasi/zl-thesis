"""
ZL_Dataset preprocessing module.

Usage:
    from preprocess_ZL import ZLDataset, ZLPreprocessingPipeline
    
    # Load raw data
    loader = ZLDataset('/path/to/data')
    data = loader.load_subject_data('sub-PD089', 'ses-S001')
    
    # Preprocess
    pipeline = ZLPreprocessingPipeline(
        data['eeg'], 
        data['eeg_timestamps'],
        data['markers'],
        data['marker_timestamps'],
        channel_labels=data['channel_labels'],
        sampling_rate=data['sampling_rate']
    )
    
    # Get fully processed epochs
    epochs, labels, metadata = pipeline.get_processed_epochs()
"""

from .zl_dataset import (
    ZLDataset, 
    get_zl_dataset,
    ZL_EEG_STREAM_PATTERN,
    ZL_MARKER_STREAM_PATTERN,
    ZL_EEG_FILE_PATTERN,
    ZL_SAMPLING_RATE,
    ZL_NUM_CHANNELS,
    ZL_TOTAL_CHANNELS,
)
from .zl_preprocessing_pipeline import ZLPreprocessingPipeline, ZLMarkerHandler

__all__ = [
    # Dataset loader
    'ZLDataset',
    'get_zl_dataset',
    # Config constants
    'ZL_EEG_STREAM_PATTERN',
    'ZL_MARKER_STREAM_PATTERN',
    'ZL_EEG_FILE_PATTERN',
    'ZL_SAMPLING_RATE',
    'ZL_NUM_CHANNELS',
    'ZL_TOTAL_CHANNELS',
    # Preprocessing
    'ZLPreprocessingPipeline',
    'ZLMarkerHandler',
]
