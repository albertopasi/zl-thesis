"""
Data loader for the ZL_Dataset.
Handles dynamic subject discovery.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import re

from .base import DataLoader
from .zl_config import ZL_EEG_STREAM_PATTERN, ZL_MARKER_STREAM_PATTERN, ZL_EEG_FILE_PATTERN


class ZLDataset(DataLoader):
    """
    Loader for the ZL_Dataset EEG dataset (Zander Labs).
    
    Uses precise stream name patterns specific to ZL_Dataset:
    - EEG stream: 'actiCHamp'
    - Marker stream: 'ZLT-markers'
    
    Expected structure (flexible - subjects can be at any depth):
        dataset_root/
        ├── [optional_subfolder/]
        │   ├── sub-*/
        │   │   ├── ses-*/
        │   │   │   ├── eeg/
        │   │   │   │   └── *.xdf (or other EEG format)
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        └── ...
    
    Discovers subjects dynamically by searching recursively for sub-* directories.
    """
    
    def __init__(self, dataset_root: str, eeg_pattern: str = ZL_EEG_FILE_PATTERN,
                 eeg_stream_pattern: str = ZL_EEG_STREAM_PATTERN, 
                 marker_stream_pattern: str = ZL_MARKER_STREAM_PATTERN):
        """
        Initialize ZL_Dataset loader.
        
        Args:
            dataset_root: Root directory of the dataset
            eeg_pattern: Pattern to match EEG files (default from config)
            eeg_stream_pattern: Stream name pattern for EEG data (default from config)
            marker_stream_pattern: Stream name pattern for markers (default from config)
        """
        self.eeg_pattern = eeg_pattern
        self.eeg_stream_pattern = eeg_stream_pattern
        self.marker_stream_pattern = marker_stream_pattern
        self._subject_paths = {}  # Map subject_id to actual directory path
        super().__init__(dataset_root)
    
    def _discover_subjects(self):
        """
        Discover subjects by recursively searching for sub-* directories.
        Sessions are discovered within each subject directory.
        Handles nested folder structures (e.g., data/Zander Labs/sub-*).
        """
        print(f"Discovering subjects in {self.dataset_root}...")
        
        # Recursively find all subject directories (pattern: sub-*)
        subject_dirs = sorted([d for d in self.dataset_root.rglob('sub-*') 
                               if d.is_dir() and d.name.startswith('sub-')])
        
        # Deduplicate in case of weird nested structures
        seen = set()
        unique_subject_dirs = []
        for d in subject_dirs:
            if d.name not in seen:
                seen.add(d.name)
                unique_subject_dirs.append(d)
        
        for subject_dir in unique_subject_dirs:
            subject_id = subject_dir.name
            
            # Find all session directories (pattern: ses-*)
            session_dirs = sorted([d for d in subject_dir.iterdir() 
                                   if d.is_dir() and d.name.startswith('ses-')])
            
            if session_dirs:
                self.subjects.append(subject_id)
                self.sessions[subject_id] = [s.name for s in session_dirs]
                self._subject_paths[subject_id] = subject_dir  # Store the actual path
                print(f"  Found {subject_id}: {len(session_dirs)} session(s)")
                print(f"    Location: {subject_dir}")
        
        if not self.subjects:
            raise ValueError(f"No subjects found in {self.dataset_root}. Expected sub-* directories at any depth.")
        
        print(f"Total: {len(self.subjects)} subjects discovered\n")
    
    def get_eeg_file_path(self, subject_id: str, session_id: str) -> Path:
        """
        Get path to EEG file for a subject/session.
        
        Args:
            subject_id: Subject ID (e.g., 'sub-PD089')
            session_id: Session ID (e.g., 'ses-S001')
            
        Returns:
            Path to EEG file
            
        Raises:
            FileNotFoundError: If EEG file not found
        """
        # Use stored subject path (handles nested folder structures)
        if subject_id not in self._subject_paths:
            raise ValueError(f"Subject '{subject_id}' not found in discovered subjects")
        
        subject_dir = self._subject_paths[subject_id]
        eeg_dir = subject_dir / session_id / 'eeg'
        
        if not eeg_dir.exists():
            raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")
        
        # Find file matching pattern
        matching_files = list(eeg_dir.glob(self.eeg_pattern))
        
        if not matching_files:
            raise FileNotFoundError(
                f"No EEG files matching '{self.eeg_pattern}' in {eeg_dir}"
            )
        
        if len(matching_files) > 1:
            print(f"Warning: Multiple EEG files found, using first: {matching_files[0].name}")
        
        return matching_files[0]
    
    def load_subject_data(self, subject_id: str, session_id: str) -> Dict[str, Any]:
        """
        Load raw data for a subject/session using pyxdf.
        
        Args:
            subject_id: Subject ID (e.g., 'sub-PD089')
            session_id: Session ID (e.g., 'ses-S001')
            
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
        import pyxdf
        
        xdf_path = self.get_eeg_file_path(subject_id, session_id)
        
        if not xdf_path.exists():
            raise FileNotFoundError(f"XDF file not found: {xdf_path}")
        
        print(f"Loading {subject_id} / {session_id} from {xdf_path.name}...")
        
        # Load XDF file
        streams, header = pyxdf.load_xdf(str(xdf_path))
        
        # Extract EEG stream using specific pattern (actiCHamp)
        eeg_stream = self._find_stream(streams, pattern=self.eeg_stream_pattern)
        if eeg_stream is None:
            raise ValueError(f"Could not find EEG stream '{self.eeg_stream_pattern}' in {xdf_path}")
        
        # Extract marker stream using specific pattern (ZLT-markers)
        marker_stream = self._find_stream(streams, pattern=self.marker_stream_pattern)
        if marker_stream is None:
            raise ValueError(f"Could not find marker stream '{self.marker_stream_pattern}' in {xdf_path}")
        
        # Parse EEG data
        eeg_array = np.array(eeg_stream['time_series'])
        eeg_timestamps = np.array(eeg_stream['time_stamps'])
        sampling_rate = float(eeg_stream['info']['nominal_srate'][0])
        
        # Extract channel labels
        try:
            channel_labels = [
                ch['label'][0] 
                for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']
            ]
        except (KeyError, IndexError, TypeError):
            n_channels = eeg_array.shape[1]
            channel_labels = [f"CH{i+1}" for i in range(n_channels)]
        
        # Parse marker data
        markers = [
            item[0] if isinstance(item, (list, tuple)) else item 
            for item in marker_stream['time_series']
        ]
        marker_timestamps = np.array(marker_stream['time_stamps'])
        
        return {
            'eeg': eeg_array,
            'markers': markers,
            'eeg_timestamps': eeg_timestamps,
            'marker_timestamps': marker_timestamps,
            'sampling_rate': sampling_rate,
            'channel_labels': channel_labels,
            'subject_id': subject_id,
            'session_id': session_id,
        }
    
    @staticmethod
    def _find_stream(streams, pattern: str = 'EEG'):
        """
        Find a stream by name pattern.
        
        Args:
            streams: List of streams from pyxdf.load_xdf
            pattern: String pattern to match in stream name (case-insensitive)
            
        Returns:
            Stream dict or None if not found
        """
        for stream in streams:
            stream_name = stream['info'].get('name', [''])[0]
            if pattern.lower() in stream_name.lower():
                ts = stream.get('time_series')
                ts_len = ts.shape[0] if hasattr(ts, 'shape') else len(ts) if ts else 0
                if ts_len > 0:
                    return stream
        return None
