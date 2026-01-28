"""
MNE-based EEG preprocessing pipeline for ZL_Dataset.

DATASET-SPECIFIC NOTE: This implementation is customized for the ZL_Dataset format,
which uses task/no-task markers with specific onset/offset filtering rules.

Implements preprocessing pipeline from test-reve/eeg_processing with:
- Channel filtering (exclude AUX and Markers)
- Bandpass filtering (FIR method)
- Downsampling to 200 Hz
- Epoch extraction around markers
- Per-epoch z-score normalization
- ZL-specific marker parsing and deduplication
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import mne
import json

from .preprocessing_config import (
    MNE_BANDPASS_LOW, MNE_BANDPASS_HIGH,
    EPOCH_TMIN, EPOCH_TMAX, NORMALIZE_METHOD, NORMALIZE_PER_EPOCH,
    EXCLUDE_CHANNELS, SKIP_MARKERS, TASK_MARKER_PREFIX, NO_TASK_MARKER_PREFIX,
    DOWNSAMPLE_RATE
)
from .electrode_handler import ElectrodePositionExtractor


class ZLMarkerHandler:
    """
    ZL_Dataset-specific marker parsing and classification logic.
    
    Handles:
    - Filtering non-experimental markers (Recording/Start, Break, Pause, Rest, etc.)
    - Filtering onset/offset markers
    - Binary classification (task vs no-task)
    - Deduplication of consecutive same-label markers
    """
    
    def __init__(self, 
                 task_marker_prefix: str = TASK_MARKER_PREFIX,
                 no_task_marker_prefix: str = NO_TASK_MARKER_PREFIX,
                 skip_markers: set = SKIP_MARKERS):
        """Initialize marker handler with ZL_Dataset configuration."""
        self.task_marker_prefix = task_marker_prefix.lower()
        self.no_task_marker_prefix = no_task_marker_prefix.lower()
        self.skip_markers = {m.lower() for m in skip_markers}
    
    def is_skip_marker(self, marker: str) -> bool:
        """Check if marker should be skipped (non-experimental)."""
        marker_lower = marker.lower()
        return any(skip in marker_lower for skip in self.skip_markers)
    
    def is_valid_marker(self, marker: str) -> bool:
        """Check if marker is valid task/no-task (excluding onset/offset)."""
        marker_lower = marker.lower()
        
        # Exclude onset/offset markers
        # Look for "onset task", "onset no task", "offset task", "offset no task"
        # (not just "onset" anywhere, since "(Def/PD, Onset)" appears on all markers)
        exclude_patterns = ['onset task', 'onset no task', 'offset task', 'offset no task']
        if any(pattern in marker_lower for pattern in exclude_patterns):
            return False
        
        # Check for task or no-task prefix
        return (self.task_marker_prefix in marker_lower or 
                self.no_task_marker_prefix in marker_lower)
    
    def extract_binary_label(self, marker: str) -> int:
        """Extract binary label (0=no task, 1=task)."""
        marker_lower = marker.lower()
        if marker_lower.startswith(self.no_task_marker_prefix):
            return 0
        elif marker_lower.startswith(self.task_marker_prefix):
            return 1
        return None
    
    def process_markers(self, 
                       markers: List[str],
                       marker_timestamps: np.ndarray
                       ) -> Tuple[List[Tuple[str, float, int]], Dict[str, int]]:
        """
        Process markers with ZL_Dataset-specific rules.
        
        Returns:
            tuple: (filtered_markers, skipped_marker_counts)
                filtered_markers: List of (marker, timestamp, label) tuples
                skipped_marker_counts: Dict of skipped marker names and their counts
        """
        skipped_markers = {}
        valid_markers_with_ts = []
        
        # Filter and classify valid markers
        for marker, marker_ts in zip(markers, marker_timestamps):
            if self.is_skip_marker(marker):
                skipped_markers[marker] = skipped_markers.get(marker, 0) + 1
                continue
            
            if not self.is_valid_marker(marker):
                continue
            
            label = self.extract_binary_label(marker)
            if label is None:
                continue
            
            valid_markers_with_ts.append((marker, marker_ts, label))
        
        # Deduplicate consecutive same-label markers
        filtered_markers = []
        i = 0
        while i < len(valid_markers_with_ts):
            marker, marker_ts, label = valid_markers_with_ts[i]
            j = i
            while j < len(valid_markers_with_ts) and valid_markers_with_ts[j][2] == label:
                j += 1
            
            count = j - i
            if count == 1:
                filtered_markers.append((marker, marker_ts, label))
            else:
                # Keep middle marker of duplicates
                middle_idx = i + count // 2
                filtered_markers.append(valid_markers_with_ts[middle_idx])
            
            i = j
        
        return filtered_markers, skipped_markers


class ZLPreprocessingPipeline:
    """
    MNE-based preprocessing pipeline for ZL_Dataset EEG data.
    
    DATASET-SPECIFIC: Customized for ZL_Dataset marker format and requirements.
    
    Pipeline (in order):
    1. Channel exclusion: Remove non-EEG channels (AUX_*, Markers) - FIRST STEP
    2. Apply FIR bandpass filter (0.5-99.5 Hz)
    3. Downsample from 500 Hz to 200 Hz
    4. Extract epochs around task/no-task markers (tmin=-1.5s, tmax=1.5s)
    5. Normalize each epoch with z-score
    6. Handle ZL-specific marker rules (onset/offset filtering, deduplication)
    """
    
    def __init__(self, eeg_data: np.ndarray, 
                 eeg_timestamps: np.ndarray,
                 markers: List[str],
                 marker_timestamps: np.ndarray,
                 channel_labels: List[str] = None,
                 sampling_rate: float = 500.0):
        """Initialize ZL preprocessing pipeline."""
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
        self.excluded_indices = []
        self.marker_handler = ZLMarkerHandler()
        self._create_raw_array()
    
    def _exclude_non_eeg_channels(self) -> Tuple[np.ndarray, List[str]]:
        """
        Exclude non-EEG channels from raw data.
        Removes AUX channels and Markers channel before any filtering.
        
        Returns:
            tuple: (filtered_eeg_data, filtered_channel_labels)
                - filtered_eeg_data: EEG data with only EEG channels (shape: samples, n_eeg_channels)
                - filtered_channel_labels: List of remaining EEG channel names
        """
        # Identify channels to exclude
        self.excluded_indices = []
        excluded_names = []
        for i, label in enumerate(self.channel_labels):
            for exclude_pattern in EXCLUDE_CHANNELS:
                if exclude_pattern.lower() in label.lower():
                    self.excluded_indices.append(i)
                    excluded_names.append(label)
                    break
        
        # Keep only EEG channels
        keep_indices = [i for i in range(len(self.channel_labels)) if i not in self.excluded_indices]
        eeg_data_filtered = self.eeg_data[:, keep_indices]
        eeg_ch_names = [self.channel_labels[i] for i in keep_indices]
        
        print(f"\nChannel Exclusion")
        print(f"  Input channels: {len(self.channel_labels)}")
        print(f"  Excluded {len(self.excluded_indices)} non-EEG channels:")
        for name in excluded_names:
            print(f"    - {name}")
        print(f"  Output channels: {len(eeg_ch_names)} EEG channels")
        print(f"  Data shape: {eeg_data_filtered.shape} (samples, channels)")
        
        return eeg_data_filtered, eeg_ch_names
    
    def _create_raw_array(self):
        """Create MNE RawArray with only EEG channels (channels already excluded)."""
        # Exclude non-EEG channels
        eeg_data_filtered, eeg_ch_names = self._exclude_non_eeg_channels()
        
        # Create channel info with only EEG channels
        ch_types = ['eeg'] * len(eeg_ch_names)
        info = mne.create_info(ch_names=eeg_ch_names, sfreq=self.sampling_rate, ch_types=ch_types)
        
        # Create RawArray with filtered data
        self.raw = mne.io.RawArray(eeg_data_filtered.T, info, verbose=False)
        
        print(f"  Created MNE RawArray: {self.raw.get_data().shape} (channels, samples)")
    
    def preprocess(self, l_freq=MNE_BANDPASS_LOW, h_freq=MNE_BANDPASS_HIGH, **kwargs) -> np.ndarray:
        """
        Apply MNE preprocessing pipeline (channels already excluded in __init__).
        
        Pipeline order:
        1. Channel exclusion (already done in _create_raw_array)
        2. FIR Bandpass filter
        3. Downsample to target rate
        
        IMPORTANT: Markers/annotations are tied to absolute time (seconds), not sample indices.
        When raw.resample() is called, MNE automatically updates raw.info['sfreq'] and 
        re-indexes all annotations. In extract_epochs(), use mne.events_from_annotations()
        to get correct sample indices based on the current sampling rate.
        
        Args:
            l_freq: Low frequency for bandpass (Hz)
            h_freq: High frequency for bandpass (Hz)
            
        Returns:
            np.ndarray: Preprocessed EEG data (only EEG channels, filtered, downsampled)
        """
        if self.raw is None:
            raise RuntimeError("RawArray not created. Call _create_raw_array() first.")
        
        print(f"\n Bandpass Filtering")
        print(f"  FIR Bandpass filter: {l_freq}-{h_freq} Hz")
        
        # Apply FIR bandpass filter (default method in MNE)
        self.raw.filter(l_freq, h_freq, verbose=False)
        
        # Resample
        print(f"\n Downsampling")
        if DOWNSAMPLE_RATE is not None and DOWNSAMPLE_RATE < self.sampling_rate:
            print(f"  From {self.sampling_rate} Hz to {DOWNSAMPLE_RATE} Hz...")
            self.raw.resample(DOWNSAMPLE_RATE, verbose=False)
            self.sampling_rate = DOWNSAMPLE_RATE
            print(f"  New sampling rate: {self.sampling_rate} Hz")
        
        preprocessed_data = self.raw.get_data()
        print(f"  Output shape: {preprocessed_data.shape} (channels, samples)")
        
        return preprocessed_data
    
    def extract_epochs(self, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, **kwargs) -> Tuple[np.ndarray, List, List]:
        """
        Extract epochs around marker events.
        
        Args:
            tmin: Start of epoch relative to marker (seconds)
            tmax: End of epoch relative to marker (seconds)
            
        Returns:
            tuple: (epochs_array, labels, metadata)
        """
        print(f"\nExtracting epochs: [{tmin}, {tmax}] seconds around markers...")
        
        # Create events array from markers
        events, event_id = self._create_events_array()
        
        if len(events) == 0:
            print("WARNING: No valid task/no-task markers found!")
            self.epochs = np.array([])
            self.epoch_labels = []
            return self.epochs, self.epoch_labels, []
        
        print(f"Found {len(events)} valid events (after filtering onset/offset and deduplicating)")
        
        # Analyze overlapping epochs
        self._analyze_epoch_overlaps(events, tmin, tmax)
        
        # Create epochs using MNE
        epochs = mne.Epochs(
            self.raw, events, event_id,
            tmin=tmin, tmax=tmax,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            event_repeated='merge'
        )
        
        # Convert to numpy array
        epochs_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
        actual_events = epochs.events
        
        # Map event IDs to binary labels
        labels = []
        for e in actual_events:
            event_id_val = int(e[2])
            if event_id_val == event_id['no task']:
                labels.append(0)
            elif event_id_val == event_id['task']:
                labels.append(1)
        
        # Validate
        if len(labels) != epochs_data.shape[0]:
            print(f"WARNING: Event/epoch mismatch! Using actual epoch count")
            labels = labels[:epochs_data.shape[0]]
        
        # Report dropped epochs
        n_dropped = len(events) - epochs_data.shape[0]
        if n_dropped > 0:
            print(f"\nEpochs dropped during extraction: {n_dropped}")
            print(f"  Original events: {len(events)}")
            print(f"  Extracted epochs: {epochs_data.shape[0]}")
            print(f"  Note: Epochs with overlapping time windows are merged/dropped by MNE")
        
        # Create metadata
        metadata = []
        for i, (event, label) in enumerate(zip(actual_events, labels)):
            label_str = 'no task' if label == 0 else 'task'
            metadata.append({
                'label': label,
                'label_str': label_str,
                'marker_timestamp': self.marker_timestamps[int(event[0])] if int(event[0]) < len(self.marker_timestamps) else None,
                'sample_index': int(event[0]),
            })
        
        print(f"Extracted {epochs_data.shape[0]} epochs")
        print(f"Epochs shape: {epochs_data.shape}")
        
        self.epochs = epochs_data
        self.epoch_labels = labels
        self.epoch_metadata = metadata
        
        return epochs_data, labels, metadata
    
    def _create_events_array(self) -> Tuple[np.ndarray, Dict]:
        """
        Create MNE events array from ZL_Dataset markers.
        
        Uses ZLMarkerHandler to process markers with dataset-specific rules:
        - Filters non-experimental markers (Recording/Start, Break, Pause, Rest)
        - Filters onset/offset markers
        - Deduplicates consecutive same-label markers
        
        IMPORTANT: Markers are linked to absolute time (seconds), not sample indices.
        When raw data is resampled, MNE automatically re-indexes by updating raw.info['sfreq'].
        We create mne.Annotations tied to time, and MNE will convert them to correct sample indices.
        """
        event_id = {'no task': 1, 'task': 2}
        
        # Process markers using ZL_Dataset-specific handler
        filtered_markers, skipped_markers = self.marker_handler.process_markers(
            self.markers, self.marker_timestamps
        )
        
        # Create annotations (tied to relative time, not sample indices)
        # MNE will automatically handle re-indexing when extracting events
        # NOTE: Timestamps must be relative to start of recording, not absolute time
        annotations_onsets = []
        annotations_durations = []
        annotations_descriptions = []
        
        # Get the start time of the recording
        t_start = self.eeg_timestamps[0]
        
        for marker, marker_ts, label in filtered_markers:
            # Convert absolute marker timestamp to relative time
            relative_marker_time = marker_ts - t_start
            
            label_str = 'no task' if label == 0 else 'task'
            annotations_onsets.append(relative_marker_time)
            annotations_durations.append(0)  # Point event (no duration)
            annotations_descriptions.append(label_str)
        
        # Add annotations to raw data
        # These are automatically re-indexed when data is resampled
        annotations = mne.Annotations(
            onset=annotations_onsets,
            duration=annotations_durations,
            description=annotations_descriptions,
            orig_time=None  # Already in relative time
        )
        self.raw.set_annotations(annotations)
        
        # Extract events from annotations
        # MNE automatically handles sample index conversion based on current sampling rate
        events, event_id_from_mne = mne.events_from_annotations(self.raw, event_id={'no task': 1, 'task': 2})
        
        print(f"\nZL_Dataset Binary Event ID mapping:")
        print(f"  no task: ID={event_id['no task']} (count={np.sum(events[:, 2] == event_id['no task']) if len(events) > 0 else 0})")
        print(f"  task: ID={event_id['task']} (count={np.sum(events[:, 2] == event_id['task']) if len(events) > 0 else 0})")
        
        if skipped_markers:
            print(f"\nSkipped non-experimental markers (ZL_Dataset specific):")
            for marker, count in sorted(skipped_markers.items()):
                print(f"  {marker}: {count}")
        
        return events, event_id
    
    
    def _analyze_epoch_overlaps(self, events: np.ndarray, tmin: float, tmax: float):
        """Analyze overlapping epoch windows."""
        if len(events) < 2:
            return
        
        epoch_samples_before = int(abs(tmin) * self.sampling_rate)
        epoch_samples_after = int(tmax * self.sampling_rate)
        
        overlaps = 0
        for i in range(len(events) - 1):
            curr_sample = events[i][0]
            next_sample = events[i + 1][0]
            curr_end = curr_sample + epoch_samples_after
            next_start = next_sample - epoch_samples_before
            
            if curr_end >= next_start:
                overlaps += 1
        
        if overlaps > 0:
            pct = (overlaps / (len(events) - 1)) * 100
            print(f"  {overlaps} consecutive events have overlapping epoch windows ({pct:.1f}%)")
    
    def normalize_epochs(self, method=NORMALIZE_METHOD, **kwargs) -> np.ndarray:
        """
        Normalize extracted epochs.
        
        Args:
            method: Normalization method ('zscore')
            
        Returns:
            np.ndarray: Normalized epochs
        """
        if self.epochs is None or len(self.epochs) == 0:
            print("WARNING: No epochs to normalize")
            return self.epochs
        
        print(f"\nNormalizing epochs with {method} method...")
        
        if method == 'zscore':
            # Z-score normalization per epoch
            normalized = np.zeros_like(self.epochs)
            for i in range(len(self.epochs)):
                epoch = self.epochs[i]
                mean = np.mean(epoch, axis=1, keepdims=True)
                std = np.std(epoch, axis=1, keepdims=True)
                std[std == 0] = 1  # Avoid division by zero
                normalized[i] = (epoch - mean) / std
            return normalized
        
        else:
            print(f"Unknown normalization method: {method}")
            return self.epochs
    
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
    
    @staticmethod
    def extract_electrode_positions_from_xdf(xdf_path: str) -> np.ndarray:
        """
        Extract electrode positions from XDF file CapTrak metadata.
        
        Delegates to ElectrodePositionExtractor for electrode handling.
        
        Args:
            xdf_path: Path to XDF file
            
        Returns:
            np.ndarray: Electrode positions of shape (n_electrodes, 3) with x, y, z coordinates
        """
        return ElectrodePositionExtractor.extract_from_xdf(xdf_path)


# Backward compatibility aliases
MNEPreprocessor = ZLPreprocessingPipeline
MNEPreprocessorZLDataset = ZLPreprocessingPipeline
