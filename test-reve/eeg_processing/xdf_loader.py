"""
XDF file loading and stream extraction utilities.
"""

import os
import pyxdf
import numpy as np
from config import DATA_ROOT, EEG_STREAM_PATTERN, MARKER_STREAM_PATTERN


class XDFLoader:
    """Load and parse XDF files."""
    
    def __init__(self, subject_id, session_id):
        """
        Initialize XDF loader.
        
        Args:
            subject_id: Subject ID (e.g., 'sub-PD089')
            session_id: Session ID (e.g., 'ses-S001')
        """
        self.subject_id = subject_id
        self.session_id = session_id
        self.xdf_path = self._construct_path()
        self.streams = None
        self.header = None
        
    def _construct_path(self):
        """Construct path to XDF file."""
        path = os.path.join(
            DATA_ROOT,
            self.subject_id,
            self.session_id,
            'eeg',
            f'{self.subject_id}_{self.session_id}_task-demo_workload_run-001_eeg.xdf'
        )
        return path
    
    def load(self):
        """
        Load XDF file.
        
        Returns:
            tuple: (streams, header)
        """
        if not os.path.exists(self.xdf_path):
            raise FileNotFoundError(f"XDF file not found: {self.xdf_path}")
        
        print(f"Loading XDF file: {self.xdf_path}")
        self.streams, self.header = pyxdf.load_xdf(self.xdf_path)
        print(f"Loaded {len(self.streams)} streams")
        return self.streams, self.header
    
    def get_eeg_stream(self):
        """
        Extract EEG data stream.
        Finds the stream by pattern matching on stream name (robust to different stream orders).
        
        Returns:
            dict: EEG stream with 'time_series', 'time_stamps', 'info' keys
        """
        if self.streams is None:
            self.load()
        
        # Search for EEG stream by name pattern (not by fixed index)
        for i, stream in enumerate(self.streams):
            stream_name = stream['info'].get('name', [''])[0]
            ts = stream.get('time_series')
            ts_shape = ts.shape if hasattr(ts, 'shape') else (len(ts) if ts else 0)
            ts_len = ts_shape[0] if isinstance(ts_shape, tuple) else ts_shape
            if EEG_STREAM_PATTERN in stream_name and ts_len > 0:
                print(f"  Found EEG stream '{stream_name}' at index {i}")
                return stream
        
        # Fallback: raise error if pattern not found
        raise ValueError(f"Could not find EEG stream with pattern '{EEG_STREAM_PATTERN}' in any of {len(self.streams)} streams")
    
    def get_marker_stream(self):
        """
        Extract marker/event stream.
        Finds the stream by pattern matching on stream name (robust to different stream orders).
        
        Returns:
            dict: Marker stream with 'time_series', 'time_stamps', 'info' keys
        """
        if self.streams is None:
            self.load()
        
        # Search for marker stream by name pattern (not by fixed index)
        for i, stream in enumerate(self.streams):
            stream_name = stream['info'].get('name', [''])[0]
            ts = stream.get('time_series')
            ts_len = len(ts) if isinstance(ts, list) else (ts.shape[0] if ts is not None else 0)
            if MARKER_STREAM_PATTERN in stream_name and ts_len > 0:
                print(f"  Found marker stream '{stream_name}' at index {i}")
                return stream
        
        # Fallback: raise error if pattern not found
        raise ValueError(f"Could not find marker stream with pattern '{MARKER_STREAM_PATTERN}' in any of {len(self.streams)} streams")
    
    def get_eeg_data(self):
        """
        Get parsed EEG data.
        
        Returns:
            tuple: (eeg_array, timestamps, sampling_rate)
                - eeg_array: np.ndarray of shape (samples, channels)
                - timestamps: np.ndarray of timestamps
                - sampling_rate: float sampling rate in Hz
        """
        stream = self.get_eeg_stream()
        
        eeg_array = np.array(stream['time_series'])
        timestamps = np.array(stream['time_stamps'])
        sampling_rate = float(stream['info']['nominal_srate'][0])
        
        print(f"EEG shape: {eeg_array.shape}")
        print(f"Sampling rate: {sampling_rate} Hz")
        
        return eeg_array, timestamps, sampling_rate
    
    def get_marker_data(self):
        """
        Get parsed marker data.
        
        Returns:
            tuple: (markers, timestamps)
                - markers: list of marker strings
                - timestamps: np.ndarray of timestamps
        """
        stream = self.get_marker_stream()
        
        # Flatten nested marker structure
        markers = [item[0] if isinstance(item, (list, tuple)) else item 
                   for item in stream['time_series']]
        timestamps = np.array(stream['time_stamps'])
        
        print(f"Total markers: {len(markers)}")
        # print(f"Unique markers: {set(markers)}")
        
        return markers, timestamps
    
    def get_channel_labels(self):
        """
        Extract channel labels from EEG stream info.
        
        Returns:
            list: Channel labels
        """
        stream = self.get_eeg_stream()
        eeg_info = stream['info']
        
        try:
            channel_labels = [ch['label'][0] for ch in eeg_info['desc'][0]['channels'][0]['channel']]
            return channel_labels
        except (KeyError, IndexError, TypeError):
            print("Warning: Could not extract channel labels from metadata")
            return [f"CH{i}" for i in range(99)]
