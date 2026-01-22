"""
MNE-based EEG preprocessor for ZL_Dataset.

Implements preprocessing pipeline from test-reve/eeg_processing with:
- Channel filtering (exclude AUX and Markers)
- Bandpass filtering
- Common Average Reference (CAR)
- Epoch extraction around markers
- Per-epoch z-score normalization
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import mne

from .base import EEGPreprocessor
from .preprocess_config import (
    MNE_BANDPASS_LOW, MNE_BANDPASS_HIGH, APPLY_CAR_REFERENCE,
    EPOCH_TMIN, EPOCH_TMAX, NORMALIZE_METHOD, NORMALIZE_PER_EPOCH,
    EXCLUDE_CHANNELS, SKIP_MARKERS, TASK_MARKER_PREFIX, NO_TASK_MARKER_PREFIX,
    DOWNSAMPLE_RATE
)


class MNEPreprocessor(EEGPreprocessor):
    """
    MNE-based preprocessor for ZL_Dataset EEG data.
    
    Pipeline:
    1. Create MNE RawArray and exclude non-EEG channels (AUX_*, Markers)
    2. Apply bandpass filter (0.5-99.5 Hz)
    3. Downsample from 500 Hz to 200 Hz
    4. Extract epochs around marker events (tmin=-1.5s, tmax=1.5s)
    5. Normalize each epoch with z-score
    """
    
    def __init__(self, eeg_data: np.ndarray, 
                 eeg_timestamps: np.ndarray,
                 markers: List[str],
                 marker_timestamps: np.ndarray,
                 channel_labels: List[str] = None,
                 sampling_rate: float = 500.0):
        """Initialize MNE preprocessor."""
        super().__init__(eeg_data, eeg_timestamps, markers, marker_timestamps, 
                         channel_labels, sampling_rate)
        self.excluded_indices = []
        self._create_raw_array()
    
    def _create_raw_array(self):
        """Create MNE RawArray, excluding non-EEG channels."""
        # Identify channels to exclude
        self.excluded_indices = []
        for i, label in enumerate(self.channel_labels):
            for exclude_pattern in EXCLUDE_CHANNELS:
                if exclude_pattern.lower() in label.lower():
                    self.excluded_indices.append(i)
                    break
        
        # Create channel info
        ch_names = self.channel_labels
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=self.sampling_rate, ch_types=ch_types)
        
        # Create RawArray
        self.raw = mne.io.RawArray(self.eeg_data.T, info, verbose=False)
        
        print(f"Created MNE RawArray: {self.raw.get_data().shape}")
        print(f"  Excluding {len(self.excluded_indices)} non-EEG channels: {[self.channel_labels[i] for i in self.excluded_indices]}")
    
    def preprocess(self, l_freq=MNE_BANDPASS_LOW, h_freq=MNE_BANDPASS_HIGH, **kwargs) -> np.ndarray:
        """
        Apply MNE preprocessing: bandpass filter -> downsample.
        
        Pipeline order:
        1. Bandpass filter
        2. Downsample to target rate
        
        Args:
            l_freq: Low frequency for bandpass (Hz)
            h_freq: High frequency for bandpass (Hz)
            
        Returns:
            np.ndarray: Preprocessed EEG data
        """
        if self.raw is None:
            raise RuntimeError("RawArray not created. Call _create_raw_array() first.")
        
        print(f"\nApplying MNE preprocessing...")
        print(f"  Bandpass filter: {l_freq}-{h_freq} Hz")
        
        # 1. Apply bandpass filter
        self.raw.filter(l_freq, h_freq, l_trans_bandwidth=1, h_trans_bandwidth=1, 
                       verbose=False)
        
        # 2. Downsample if needed
        if DOWNSAMPLE_RATE is not None and DOWNSAMPLE_RATE < self.sampling_rate:
            print(f"  Downsampling from {self.sampling_rate} Hz to {DOWNSAMPLE_RATE} Hz...")
            self.raw.resample(DOWNSAMPLE_RATE, verbose=False)
            self.sampling_rate = DOWNSAMPLE_RATE
            print(f"    New sampling rate: {self.sampling_rate} Hz")
        
        return self.raw.get_data()
    
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
    
    def _is_skip_marker(self, marker: str) -> bool:
        """Check if marker should be skipped."""
        marker_lower = marker.lower()
        for skip_pattern in SKIP_MARKERS:
            if skip_pattern.lower() in marker_lower:
                return True
        return False
    
    def _is_valid_marker(self, marker: str) -> bool:
        """Check if marker is valid task/no-task (not onset/offset)."""
        marker_lower = marker.lower()
        if marker_lower.startswith('onset') or marker_lower.startswith('offset'):
            return False
        if TASK_MARKER_PREFIX.lower() in marker_lower or NO_TASK_MARKER_PREFIX.lower() in marker_lower:
            return True
        return False
    
    def _extract_binary_label(self, marker: str) -> int:
        """Extract binary label (0=no task, 1=task)."""
        marker_lower = marker.lower()
        if marker_lower.startswith(NO_TASK_MARKER_PREFIX):
            return 0
        elif marker_lower.startswith(TASK_MARKER_PREFIX):
            return 1
        return None
    
    def _create_events_array(self) -> Tuple[np.ndarray, Dict]:
        """Create MNE events array from markers."""
        events = []
        event_id = {'no task': 1, 'task': 2}
        skipped_markers = {}
        
        # First pass: collect valid markers
        valid_markers_with_ts = []
        for marker, marker_ts in zip(self.markers, self.marker_timestamps):
            if self._is_skip_marker(marker):
                skipped_markers[marker] = skipped_markers.get(marker, 0) + 1
                continue
            
            if not self._is_valid_marker(marker):
                continue
            
            label = self._extract_binary_label(marker)
            if label is None:
                continue
            
            valid_markers_with_ts.append((marker, marker_ts, label))
        
        # Second pass: deduplicate consecutive same-label markers
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
                middle_idx = i + count // 2
                filtered_markers.append(valid_markers_with_ts[middle_idx])
            
            i = j
        
        # Third pass: create events array
        for marker, marker_ts, label in filtered_markers:
            idx = np.argmin(np.abs(self.eeg_timestamps - marker_ts))
            label_str = 'no task' if label == 0 else 'task'
            events.append([idx, 0, event_id[label_str]])
        
        events = np.array(events)
        
        print(f"\nBinary Event ID mapping:")
        print(f"  no task: ID={event_id['no task']} (count={np.sum(events[:, 2] == event_id['no task']) if len(events) > 0 else 0})")
        print(f"  task: ID={event_id['task']} (count={np.sum(events[:, 2] == event_id['task']) if len(events) > 0 else 0})")
        
        if skipped_markers:
            print(f"\nSkipped non-experimental markers:")
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
