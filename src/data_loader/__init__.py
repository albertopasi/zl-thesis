"""
Modular data loader for multiple EEG datasets.

Usage:
    # Load ZL_Dataset with auto-detection
    loader = get_data_loader()
    
    # Or specify explicitly
    loader = get_data_loader('zl', dataset_root='/path/to/data')
    
    # Get all subjects
    subjects = loader.get_all_subjects()
    
    # Load data for a subject
    data = loader.load_subject_data('sub-PD089', 'ses-S001')
    
    # Register custom loader
    DatasetRegistry.register('my_dataset', MyCustomLoader)
    custom = get_data_loader('my_dataset', '/path/to/data')
"""

from .base import DataLoader
from .zl_dataset import ZLDataset
from .registry import DatasetRegistry, get_data_loader

__all__ = [
    'DataLoader',
    'ZLDataset',
    'DatasetRegistry',
    'get_data_loader',
]
