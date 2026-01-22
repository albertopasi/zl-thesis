"""
Template for creating a custom dataset loader.

This file shows how to create a loader for a different dataset format.
Modify for the specific dataset structure.
"""

from pathlib import Path
from typing import Dict, Any

from .base import DataLoader
import numpy as np


class CustomDatasetLoader(DataLoader):
    """
    Template loader for a custom dataset format.
    
    Modify this to match your dataset structure.
    
    Example structure (customize as needed):
        dataset_root/
        ├── subject_001/
        │   ├── session_1/
        │   │   ├── eeg.pkl
        │   │   ├── markers.txt
        │   │   └── ...
        │   └── session_2/
        └── subject_002/
    """
    
    def _discover_subjects(self):
        """
        Discover subjects in your dataset format.
        Must populate self.subjects and self.sessions.
        """
        print(f"Discovering subjects in {self.dataset_root}...")
        
        # TODO: Modify this to match your directory structure
        # Example: iterate over directories matching your subject pattern
        
        # subject_dirs = sorted([d for d in self.dataset_root.iterdir() 
        #                        if d.is_dir() and d.name.startswith('subject_')])
        # 
        # for subject_dir in subject_dirs:
        #     subject_id = subject_dir.name
        #     session_dirs = [s.name for s in subject_dir.iterdir() if s.is_dir()]
        #     
        #     if session_dirs:
        #         self.subjects.append(subject_id)
        #         self.sessions[subject_id] = session_dirs
        
        raise NotImplementedError("Implement _discover_subjects for your dataset")
    
    def get_eeg_file_path(self, subject_id: str, session_id: str) -> Path:
        """
        Return path to EEG data for subject/session.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            Path to EEG file
        """
        # TODO: Modify to return path to your EEG file
        # Example:
        # return self.dataset_root / subject_id / session_id / 'eeg.pkl'
        
        raise NotImplementedError("Implement get_eeg_file_path for your dataset")
    
    def load_subject_data(self, subject_id: str, session_id: str) -> Dict[str, Any]:
        """
        Load raw data for subject/session.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            dict with keys:
                - 'eeg': np.ndarray of shape (samples, channels)
                - 'markers': list of marker strings
                - 'eeg_timestamps': np.ndarray of EEG timestamps
                - 'marker_timestamps': np.ndarray of marker timestamps
                - 'sampling_rate': float sampling rate in Hz
                - 'channel_labels': list of channel names
                - 'subject_id': str
                - 'session_id': str
        """
        # TODO: Modify to load your data format
        # Example (if using pickle files):
        # import pickle
        # 
        # eeg_path = self.get_eeg_file_path(subject_id, session_id)
        # with open(eeg_path, 'rb') as f:
        #     eeg_data = pickle.load(f)
        # 
        # # Load markers from text file
        # markers_path = eeg_path.parent / 'markers.txt'
        # with open(markers_path, 'r') as f:
        #     markers = [line.strip() for line in f]
        
        raise NotImplementedError("Implement load_subject_data for your dataset")
        
        # return {
        #     'eeg': eeg_data,  # np.ndarray
        #     'markers': markers,  # list of strings
        #     'eeg_timestamps': np.arange(eeg_data.shape[0]) / sampling_rate,
        #     'marker_timestamps': np.array([...]),  # marker times in seconds
        #     'sampling_rate': 500.0,  # Hz
        #     'channel_labels': [f'CH{i}' for i in range(eeg_data.shape[1])],
        #     'subject_id': subject_id,
        #     'session_id': session_id,
        # }


# After implementing, register your loader:
# 
# from data_loader import DatasetRegistry
# DatasetRegistry.register('custom', CustomDatasetLoader)
# 
# Then use it:
# loader = get_data_loader('custom', '/path/to/custom/data')
#
# Note: ZL_Dataset is already registered as 'zl'
