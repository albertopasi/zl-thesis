"""
Configuration for EEG processing pipeline.
"""

import os

# Data paths
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Stream detection patterns (identified by name/type, not fixed indices)
EEG_STREAM_PATTERN = 'actiCHamp'
MARKER_STREAM_PATTERN = 'ZLT-markers'

# EEG parameters
SAMPLING_RATE = 500  # Hz
NUM_EEG_CHANNELS = 96  # Keep only EEG channels
EXCLUDE_CHANNELS = ['AUX_1', 'AUX_2', 'Markers']  # Non-EEG channels to remove

# Downsampling
# Set to None to disable downsampling, or set to target Hz
# DOWNSAMPLE_RATE = None  # Downsample to this frequency (Hz), or None to keep original
# DOWNSAMPLE_RATE = 200
DOWNSAMPLE_RATE = 250

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

# Markers to classify - Target: Binary task classification (task vs no task)
# We filter for markers starting with 'task' or 'no task' (skip 'onset' and 'offset')
TARGET_MARKER_PREFIX = 'task'  # Used to verify marker validity

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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output250')
SAVE_PROCESSED_DATA = True
SAVE_FEATURES = True

# REVE model parameters
REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITIONS_MODEL_ID = "brain-bzh/reve-positions"

# Model architecture parameters
HIDDEN_DIM = 512
NUM_CHANNELS = 96 
SAMPLE_LENGTH = EPOCH_SAMPLES  # Dynamic: calculated from TMIN, TMAX, SAMPLING_RATE
NUM_CLASSES = 2  # Binary classification: 0 (no task) vs 1 (task)

# Training parameters
BATCH_SIZE = 32
N_EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Comprehensive evaluation hyperparameters
EVAL_LEARNING_RATES = [0.001, 0.002, 0.005]
EVAL_SEEDS = [42, 123, 456]
EVAL_NUM_EPOCHS = 20
