"""
Tests for ZLDataset data loader.

Tests cover:
- Subject discovery
- Data loading
- Data structure and integrity
- Error handling
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from preprocess_ZL.zl_dataset import ZL_SAMPLING_RATE, ZL_NUM_CHANNELS, ZL_TOTAL_CHANNELS


class TestZLDatasetDiscovery:
    """Tests for subject/session discovery."""
    
    def test_discover_subjects(self, zl_dataset):
        """Test that subjects are discovered correctly."""
        subjects = zl_dataset.get_all_subjects()
        
        assert subjects is not None, "Subjects list should not be None"
        assert len(subjects) > 0, "Should discover at least one subject"
        assert all(isinstance(s, str) for s in subjects), "All subjects should be strings"
        assert all(s.startswith('sub-') for s in subjects), "All subjects should start with 'sub-'"
    
    def test_discover_sessions(self, zl_dataset):
        """Test that sessions are discovered for subjects."""
        subjects = zl_dataset.get_all_subjects()
        
        for subject_id in subjects:
            sessions = zl_dataset.get_sessions_for_subject(subject_id)
            
            assert sessions is not None, f"Sessions for {subject_id} should not be None"
            assert len(sessions) > 0, f"Subject {subject_id} should have at least one session"
            assert all(s.startswith('ses-') for s in sessions), f"Sessions for {subject_id} should start with 'ses-'"
    
    def test_get_all_subject_sessions(self, zl_dataset):
        """Test that all subject/session pairs are returned."""
        pairs = zl_dataset.get_all_subject_sessions()
        
        assert pairs is not None, "Pairs list should not be None"
        assert len(pairs) > 0, "Should return at least one subject/session pair"
        assert all(len(p) == 2 for p in pairs), "Each pair should have 2 elements"
        assert all(isinstance(p[0], str) and isinstance(p[1], str) for p in pairs), "Pairs should be tuples of strings"


class TestZLDatasetLoading:
    """Tests for data loading."""
    
    def test_load_subject_data(self, zl_dataset, sample_subject_session):
        """Test that data can be loaded for a subject/session."""
        subject_id, session_id = sample_subject_session
        
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        assert data is not None, "Loaded data should not be None"
        assert isinstance(data, dict), "Loaded data should be a dictionary"
    
    def test_loaded_data_structure(self, zl_dataset, sample_subject_session):
        """Test that loaded data has the expected structure."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        # Check required keys
        required_keys = {
            'eeg', 'markers', 'eeg_timestamps', 'marker_timestamps',
            'sampling_rate', 'channel_labels', 'subject_id', 'session_id'
        }
        assert set(data.keys()) == required_keys, f"Data should have exactly these keys: {required_keys}"
    
    def test_eeg_data_shape_and_type(self, zl_dataset, sample_subject_session):
        """Test that EEG data has correct shape and type."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        eeg = data['eeg']
        
        assert isinstance(eeg, np.ndarray), "EEG should be a numpy array"
        assert eeg.ndim == 2, f"EEG should be 2D (samples, channels), got {eeg.ndim}D"
        assert eeg.shape[0] > 0, "EEG should have samples"
        assert eeg.shape[1] > 0, "EEG should have channels"
    
    def test_markers_data(self, zl_dataset, sample_subject_session):
        """Test that markers data is valid."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        markers = data['markers']
        
        assert isinstance(markers, list), "Markers should be a list"
        assert len(markers) > 0, "Should have at least one marker"
        assert all(isinstance(m, str) for m in markers), "All markers should be strings"
    
    def test_timestamps(self, zl_dataset, sample_subject_session):
        """Test that timestamps are valid."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        eeg_ts = data['eeg_timestamps']
        marker_ts = data['marker_timestamps']
        
        # Check EEG timestamps
        assert isinstance(eeg_ts, np.ndarray), "EEG timestamps should be numpy array"
        assert len(eeg_ts) == data['eeg'].shape[0], "EEG timestamps should match EEG sample count"
        assert np.all(np.diff(eeg_ts) >= 0), "EEG timestamps should be non-decreasing"
        
        # Check marker timestamps
        assert isinstance(marker_ts, np.ndarray), "Marker timestamps should be numpy array"
        assert len(marker_ts) == len(data['markers']), "Marker timestamps should match marker count"
    
    def test_sampling_rate(self, zl_dataset, sample_subject_session):
        """Test that sampling rate is valid."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        sr = data['sampling_rate']
        
        assert isinstance(sr, (int, float)), "Sampling rate should be numeric"
        assert sr > 0, "Sampling rate should be positive"
        assert sr == ZL_SAMPLING_RATE, f"ZL_Dataset should have {ZL_SAMPLING_RATE} Hz sampling rate"
    
    def test_channel_labels(self, zl_dataset, sample_subject_session):
        """Test that channel labels match EEG channels."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        labels = data['channel_labels']
        
        assert isinstance(labels, list), "Channel labels should be a list"
        assert len(labels) == data['eeg'].shape[1], "Channel labels should match EEG channel count"
        assert all(isinstance(l, str) for l in labels), "All channel labels should be strings"
    
    def test_subject_session_metadata(self, zl_dataset, sample_subject_session):
        """Test that subject and session IDs are stored in loaded data."""
        subject_id, session_id = sample_subject_session
        data = zl_dataset.load_subject_data(subject_id, session_id)
        
        assert data['subject_id'] == subject_id, "Subject ID should be preserved in loaded data"
        assert data['session_id'] == session_id, "Session ID should be preserved in loaded data"


class TestZLDatasetErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_subject(self, zl_dataset):
        """Test that invalid subject raises error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            zl_dataset.load_subject_data('sub-INVALID', 'ses-S001')
    
    def test_invalid_session(self, zl_dataset):
        """Test that invalid session raises error."""
        subjects = zl_dataset.get_all_subjects()
        if subjects:
            with pytest.raises((ValueError, FileNotFoundError)):
                zl_dataset.load_subject_data(subjects[0], 'ses-INVALID')
    
    def test_eeg_file_path_caching(self, zl_dataset, sample_subject_session):
        """Test that subject paths are cached after discovery."""
        subject_id, session_id = sample_subject_session
        
        # Get EEG file path twice
        path1 = zl_dataset.get_eeg_file_path(subject_id, session_id)
        path2 = zl_dataset.get_eeg_file_path(subject_id, session_id)
        
        assert path1 == path2, "EEG file path should be consistent"
        assert path1.exists(), "EEG file path should exist"


class TestZLDatasetConsistency:
    """Tests for consistency across multiple loads."""
    
    def test_consistent_data_loading(self, zl_dataset, sample_subject_session):
        """Test that loading the same data twice gives consistent results."""
        subject_id, session_id = sample_subject_session
        
        data1 = zl_dataset.load_subject_data(subject_id, session_id)
        data2 = zl_dataset.load_subject_data(subject_id, session_id)
        
        # Compare EEG data
        assert np.array_equal(data1['eeg'], data2['eeg']), "EEG data should be identical on reload"
        assert np.array_equal(data1['eeg_timestamps'], data2['eeg_timestamps']), "EEG timestamps should be identical"
        
        # Compare markers
        assert data1['markers'] == data2['markers'], "Markers should be identical on reload"
        assert np.array_equal(data1['marker_timestamps'], data2['marker_timestamps']), "Marker timestamps should be identical"
        
        # Compare metadata
        assert data1['sampling_rate'] == data2['sampling_rate'], "Sampling rate should be identical"
        assert data1['channel_labels'] == data2['channel_labels'], "Channel labels should be identical"
