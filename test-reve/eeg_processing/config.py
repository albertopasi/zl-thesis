"""
Configuration for EEG processing pipeline.
"""

import os

# Data paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# EEG stream indices (from data exploration)
EEG_STREAM_INDEX = 4       # 'actiCHamp-24020239' - Primary EEG data
MARKER_STREAM_INDEX = 5    # 'ZLT-markers' - Primary event markers

# EEG parameters
SAMPLING_RATE = 500  # Hz
NUM_EEG_CHANNELS = 96  # Keep only EEG channels (exclude AUX_1, AUX_2, Markers)
EXCLUDE_CHANNELS = ['AUX_1', 'AUX_2', 'Markers']  # Non-EEG auxiliary channels to remove

# Downsampling (optional: reduce computational load)
# Set to None to disable downsampling, or set to target Hz (e.g., 250 Hz)
DOWNSAMPLE_RATE = None  # Downsample to this frequency (Hz), or None to keep original
# DOWNSAMPLE_RATE = 200  # Downsample from 500 Hz to 200 Hz
# DOWNSAMPLE_RATE = 250  # Example: downsample from 500 Hz to 250 Hz

# Epoch extraction - Time window around markers
TMIN = -1.5  # Start of epoch relative to marker (seconds)
TMAX = 1.5   # End of epoch relative to marker (seconds)
# Note: EPOCH_SAMPLES will be calculated after downsampling is applied
EPOCH_SAMPLES = int(SAMPLING_RATE * (TMAX - TMIN))

# Preprocessing with MNE
NORMALIZE_DATA = True
Z_SCORE_NORMALIZE = True  # Normalize by channel statistics per epoch
APPLY_BANDPASS_FILTER = True
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 99.5

# Markers to classify - Target: Workload events
# Keep all markers containing 'workload' (regardless of task/no-task, sparkles, count, etc.)
TARGET_MARKER_PREFIX = 'workload'

# Markers to skip (non-experimental data: breaks, pauses, etc.)
# These events should be excluded from epochs
SKIP_MARKERS = {
    'Recording/Start',
    'Recording/End',
    'Break',
    'break',
    'Pause',
    'pause',
    'Rest',
    'rest',
    'Baseline',
    'baseline'
}

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
SAVE_PROCESSED_DATA = True
SAVE_FEATURES = True

# REVE model parameters
REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITIONS_MODEL_ID = "brain-bzh/reve-positions"

# Model architecture parameters
HIDDEN_DIM = 512
NUM_CHANNELS = 96 
SAMPLE_LENGTH = EPOCH_SAMPLES  # Dynamic: calculated from TMIN, TMAX, SAMPLING_RATE
NUM_CLASSES = None  # Determined dynamically from actual markers in data (set during preprocessing)

# Training parameters (for future use)
BATCH_SIZE = 32
N_EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
