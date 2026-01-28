"""
Configuration for ZL_Dataset preprocessing pipeline.
"""

# MNE Preprocessing
MNE_BANDPASS_LOW = 0.5   # Hz
MNE_BANDPASS_HIGH = 99.5  # Hz
DOWNSAMPLE_RATE = 200  # Downsample to this frequency (Hz), or None to keep original 500 Hz

# Epoch extraction
EPOCH_TMIN = -1.5   # seconds before marker
EPOCH_TMAX = 1.5    # seconds after marker

# Normalization
NORMALIZE_METHOD = 'zscore'  # 'zscore', 'minmax', etc.
NORMALIZE_PER_EPOCH = True   # Normalize each epoch independently

# Channel filtering
EXCLUDE_CHANNELS = ['AUX_1', 'AUX_2', 'Markers']  # Non-EEG channels to exclude

# Marker filtering (ZL_Dataset specific)
SKIP_MARKERS = {
    'Recording/Start',
    'Recording/End',
    'Break',
    'break',
    'Pause',
    'pause',
    'Rest',
    'rest',
}

# Binary classification markers
TASK_MARKER_PREFIX = 'task'
NO_TASK_MARKER_PREFIX = 'no task'
