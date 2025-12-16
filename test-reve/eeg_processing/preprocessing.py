"""
EEG data preprocessing and epoch extraction using MNE-Python.
"""

import numpy as np
import mne
from config import (
    SAMPLING_RATE, TMIN, TMAX,
    NORMALIZE_DATA, Z_SCORE_NORMALIZE,
    APPLY_BANDPASS_FILTER, BANDPASS_LOW, BANDPASS_HIGH,
    EXCLUDE_CHANNELS, SKIP_MARKERS, TARGET_MARKER_PREFIX,
    DOWNSAMPLE_RATE
)


class EEGPreprocessor:
    """Preprocess EEG data and extract epochs using MNE-Python."""
    
    def __init__(self, eeg_data, timestamps, markers, marker_timestamps, 
                 channel_labels=None, sampling_rate=SAMPLING_RATE):
        """
        Initialize EEG preprocessor.
        
        Args:
            eeg_data: np.ndarray of shape (samples, channels)
            timestamps: np.ndarray of EEG timestamps
            markers: list of marker strings
            marker_timestamps: np.ndarray of marker timestamps
            channel_labels: list of channel labels
            sampling_rate: sampling rate in Hz
        """
        self.eeg_data = eeg_data
        self.timestamps = timestamps
        self.markers = markers
        self.marker_timestamps = marker_timestamps
        self.channel_labels = channel_labels or [f"CH{i}" for i in range(eeg_data.shape[1])]
        self.sampling_rate = sampling_rate
        
        self.raw = None
        self.epochs = None
        self.epoch_labels = []
        self.epoch_metadata = []
        
        # Create MNE RawArray
        self._create_raw_array()
        
    def _create_raw_array(self):
        """Create MNE RawArray from EEG data, excluding non-EEG channels."""
        # Identify channels to exclude
        self.excluded_indices = []
        self.kept_channel_labels = []
        self.kept_eeg_data = []
        
        for idx, ch_name in enumerate(self.channel_labels):
            # Check if channel should be excluded
            should_exclude = any(exclude_str.lower() in ch_name.lower() for exclude_str in EXCLUDE_CHANNELS)
            
            if should_exclude:
                self.excluded_indices.append(idx)
                print(f"  Excluding channel: {ch_name}")
            else:
                self.kept_channel_labels.append(ch_name)
                self.kept_eeg_data.append(self.eeg_data[:, idx])
        
        # Update channel labels and data
        self.channel_labels = self.kept_channel_labels
        self.eeg_data = np.column_stack(self.kept_eeg_data)
        
        print(f"\nChannel filtering:")
        print(f"  Original channels: {len(self.channel_labels) + len(self.excluded_indices)}")
        print(f"  Kept EEG channels: {len(self.channel_labels)}")
        print(f"  Excluded channels: {len(self.excluded_indices)}")
        print(f"  Channel names (first 10): {self.channel_labels[:10]}")
        print(f"  Channel names (last 10): {self.channel_labels[-10:]}")
        
        # Create channel info for MNE
        ch_types = ['eeg'] * len(self.channel_labels)
        info = mne.create_info(ch_names=self.channel_labels, sfreq=self.sampling_rate, ch_types=ch_types)
        
        # Create RawArray (MNE expects shape (channels, samples))
        self.raw = mne.io.RawArray(self.eeg_data.T, info, verbose=False)
        
        print(f"\nCreated MNE RawArray: {self.raw}")
    
    def _is_workload_marker(self, marker):
        """Check if marker is a workload event."""
        return TARGET_MARKER_PREFIX.lower() in marker.lower()
    
    def _extract_workload_level(self, marker):
        """
        Extract workload level from marker name.
        Maps full marker names to simplified workload levels: original, low, medium, high
        
        Args:
            marker: Full marker string
            
        Returns:
            str: Simplified workload level (e.g., 'original workload')
        """
        marker_lower = marker.lower()
        
        # Map workload levels (order matters: check most specific first)
        workload_levels = {
            'high workload': 'high workload',
            'medium workload': 'medium workload',
            'low workload': 'low workload',
            'original workload': 'original workload',
        }
        
        for level_pattern, level_name in workload_levels.items():
            if level_pattern in marker_lower:
                return level_name
        
        # Fallback: return original marker if no pattern matched
        print(f"Warning: Could not extract workload level from marker: {marker}")
        return marker
    
    def _is_skip_marker(self, marker):
        """Check if marker should be skipped (non-experimental data)."""
        # Check exact matches and case-insensitive partial matches for known skip patterns
        marker_lower = marker.lower()
        for skip_pattern in SKIP_MARKERS:
            if skip_pattern.lower() in marker_lower:
                return True
        return False
    
    def _analyze_epoch_overlaps(self, events, tmin, tmax):
        """
        Analyze how many epochs will have overlapping time windows.
        This is informational to help understand why epochs are dropped.
        
        Args:
            events: Array of event times (sample indices)
            tmin: Start of epoch relative to marker (seconds)
            tmax: End of epoch relative to marker (seconds)
        """
        if len(events) < 2:
            return
        
        # Calculate epoch window size in samples
        epoch_samples_before = int(abs(tmin) * self.sampling_rate)
        epoch_samples_after = int(tmax * self.sampling_rate)
        
        # Count overlapping windows
        overlaps = 0
        for i in range(len(events) - 1):
            curr_sample = events[i][0]
            next_sample = events[i + 1][0]
            
            # Current epoch ends at: curr_sample + epoch_samples_after
            # Next epoch starts at: next_sample - epoch_samples_before
            
            curr_end = curr_sample + epoch_samples_after
            next_start = next_sample - epoch_samples_before
            
            if curr_end >= next_start:
                overlaps += 1
        
        if overlaps > 0:
            pct = (overlaps / (len(events) - 1)) * 100
            print(f" {overlaps} consecutive events have overlapping epoch windows ({pct:.1f}%)")
    
    def _create_events_array(self):
        """Create MNE events array from markers."""
        events = []
        event_id = {}
        current_event_id = 1
        skipped_markers = {}
        seen_sample_indices = set()  # Track duplicate sample indices
        
        for marker, marker_ts in zip(self.markers, self.marker_timestamps):
            # Skip non-experimental markers (breaks, pauses, etc.)
            if self._is_skip_marker(marker):
                skipped_markers[marker] = skipped_markers.get(marker, 0) + 1
                continue
            
            # Filter for workload markers only
            if not self._is_workload_marker(marker):
                continue
            
            # Extract simplified workload level from marker
            simplified_marker = self._extract_workload_level(marker)
            
            # Find closest sample index to marker timestamp (minimum distance)
            idx = np.argmin(np.abs(self.timestamps - marker_ts))
            
            # Check alignment quality
            time_diff = np.abs(self.timestamps[idx] - marker_ts)
            if time_diff > (1.0 / self.sampling_rate):
                print(f"  Warning: Marker '{marker}' misaligned by {time_diff:.6f}s (>{1.0/self.sampling_rate:.6f}s)")
            
            # # Skip duplicate sample indices (multiple markers at exact same timestamp)
            # if idx in seen_sample_indices:
            #     print(f"    Skipping duplicate marker '{marker}' at sample {idx}")
            #     continue
            # seen_sample_indices.add(idx)
            
            # Create unique event ID for this marker type
            if simplified_marker not in event_id:
                event_id[simplified_marker] = current_event_id
                current_event_id += 1
            
            # events array: [sample_idx, 0, event_id]
            events.append([idx, 0, event_id[simplified_marker]])
        
        events = np.array(events)
        
        # Print event ID mapping
        print(f"\nEvent ID mapping:")
        for marker, event_num in sorted(event_id.items(), key=lambda x: x[1]):
            count = np.sum(events[:, 2] == event_num)
            print(f"  {marker}: ID={event_num} (count={count})")
        
        # Print skipped markers
        if skipped_markers:
            print(f"\nSkipped non-experimental markers:")
            for marker, count in sorted(skipped_markers.items()):
                print(f"  {marker}: {count}")
        
        return events, event_id
    
    def extract_epochs(self, tmin=TMIN, tmax=TMAX):
        """
        Extract epochs around marker events using MNE.
        
        Args:
            tmin: Start time of epoch relative to marker (seconds)
            tmax: End time of epoch relative to marker (seconds)
            
        Returns:
            tuple: (epochs_array, labels, metadata)
                - epochs_array: np.ndarray of shape (num_epochs, num_channels, epoch_samples)
                - labels: list of epoch labels
                - metadata: list of dicts with epoch info
        """
        print(f"\nExtracting epochs: [{tmin}, {tmax}] seconds around markers...")
        
        # Create events array
        events, event_id = self._create_events_array()
        
        if len(events) == 0:
            print("WARNING: No workload markers found!")
            self.epochs = np.array([])
            self.epoch_labels = []
            return self.epochs, self.epoch_labels, []
        
        print(f"Found {len(events)} workload events (after removing duplicates)")
        
        # Analyze overlapping epochs (informational)
        self._analyze_epoch_overlaps(events, tmin, tmax)
        
        # print(f"Event types: {event_id}")
        
        # Create epochs using MNE
        epochs = mne.Epochs(
            self.raw, events, event_id,
            tmin=tmin, tmax=tmax,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            event_repeated='merge'  # Handle any remaining duplicate timestamps
        )
        
        # Convert to numpy array and get labels
        epochs_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
        
        # Create reverse mapping from event_id to marker names
        id_to_marker = {v: k for k, v in event_id.items()}
        
        # Get the actual events that were used (MNE may reject/merge some)
        actual_events = epochs.events
        labels = [id_to_marker[int(e[2])] for e in actual_events if int(e[2]) in id_to_marker]
        
        # Validate label count matches epoch count
        if len(labels) != epochs_data.shape[0]:
            print(f"WARNING: Event/epoch mismatch!")
            print(f"  Events: {len(labels)}")
            print(f"  Actual epochs: {epochs_data.shape[0]}")
            print(f"  Using actual epoch count")
            labels = labels[:epochs_data.shape[0]]
        
        # Report how many epochs were dropped due to overlaps
        n_dropped = len(events) - epochs_data.shape[0]
        if n_dropped > 0:
            print(f"\nEpochs dropped during extraction: {n_dropped}")
            print(f"  Original events: {len(events)}")
            print(f"  Extracted epochs: {epochs_data.shape[0]}")
            print(f"  Reason: Overlapping epoch windows (events too close together)")
            print(f"  Note: Epochs with overlapping time windows are merged/dropped by MNE")
        
        # Create metadata
        metadata = []
        for i, (event, label) in enumerate(zip(actual_events, labels)):
            metadata.append({
                'marker': label,
                'marker_timestamp': self.timestamps[int(event[0])] if int(event[0]) < len(self.timestamps) else None,
                'sample_index': int(event[0]),
                'epoch_num': i
            })
        
        self.epochs = epochs_data
        self.epoch_labels = labels
        self.epoch_metadata = metadata
        
        print(f"Extracted {len(epochs_data)} epochs")
        print(f"Epochs shape: {epochs_data.shape}")
        print(f"Label distribution: {self._label_distribution()}")
        
        return self.epochs, self.epoch_labels, self.epoch_metadata
    
    def _label_distribution(self):
        """Get distribution of labels."""
        from collections import Counter
        return dict(Counter(self.epoch_labels))
    
    def preprocess_with_mne(self, l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, 
                           normalize=NORMALIZE_DATA):
        """
        Apply MNE preprocessing to raw data (before epoching).
        
        Args:
            l_freq: Low-frequency cutoff (Hz)
            h_freq: High-frequency cutoff (Hz)
            normalize: Apply normalization after filtering
            
        Returns:
            mne.io.Raw: Preprocessed raw data
        """
        print(f"\nApplying MNE preprocessing...")
        
        # Bandpass filter
        print(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
        self.raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        
        # Optional: Apply common average reference
        print("Applying common average reference...")
        self.raw.set_eeg_reference('average', verbose=False)
        
        # Optional: Downsample
        if DOWNSAMPLE_RATE is not None and DOWNSAMPLE_RATE < self.sampling_rate:
            print(f"Downsampling from {self.sampling_rate} Hz to {DOWNSAMPLE_RATE} Hz...")
            self.raw.resample(DOWNSAMPLE_RATE, verbose=False)
            self.sampling_rate = DOWNSAMPLE_RATE
            print(f"  New sampling rate: {self.sampling_rate} Hz")
        
        return self.raw
    
    def normalize_epochs(self, method='zscore'):
        """
        Normalize epochs after extraction.
        
        Args:
            method: 'zscore' or 'minmax'
            
        Returns:
            np.ndarray: Normalized epochs
        """
        if self.epochs is None or len(self.epochs) == 0:
            raise ValueError("No epochs to normalize. Call extract_epochs() first.")
        
        print(f"Normalizing epochs using {method}...")
        
        epochs_normalized = self.epochs.copy()
        
        if method == 'zscore':
            # Normalize per channel per epoch
            for i in range(epochs_normalized.shape[0]):
                for ch in range(epochs_normalized.shape[1]):
                    data = epochs_normalized[i, ch, :]
                    mean = np.mean(data)
                    std = np.std(data)
                    if std > 0:
                        epochs_normalized[i, ch, :] = (data - mean) / std
                        
        elif method == 'minmax':
            # Normalize per channel per epoch to [0, 1]
            for i in range(epochs_normalized.shape[0]):
                for ch in range(epochs_normalized.shape[1]):
                    data = epochs_normalized[i, ch, :]
                    data_min = np.min(data)
                    data_max = np.max(data)
                    if data_max > data_min:
                        epochs_normalized[i, ch, :] = (data - data_min) / (data_max - data_min)
        
        self.epochs = epochs_normalized
        return epochs_normalized
    
    def get_processed_epochs(self, preprocess=APPLY_BANDPASS_FILTER, 
                            normalize=NORMALIZE_DATA, tmin=TMIN, tmax=TMAX):
        """
        Get fully processed epochs with filtering and normalization.
        
        Args:
            preprocess: Apply filtering and CAR reference to raw data first
            normalize: Apply normalization to epochs
            tmin: Start of epoch relative to marker (seconds)
            tmax: End of epoch relative to marker (seconds)
            
        Returns:
            tuple: (epochs, labels, metadata)
        """
        # Preprocess raw data if requested
        if preprocess:
            self.preprocess_with_mne(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH)
        
        # Extract epochs
        epochs, labels, metadata = self.extract_epochs(tmin=tmin, tmax=tmax)
        
        # Normalize epochs if requested
        if normalize:
            epochs = self.normalize_epochs(method='zscore')
        
        return epochs, labels, metadata
    
    def get_label_indices(self, label):
        """Get indices of epochs with specific label."""
        return [i for i, l in enumerate(self.epoch_labels) if l == label]
