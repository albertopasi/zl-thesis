"""
Configuration for ZL_Dataset data loader.

Contains dataset-specific parameters like stream names, expected values, and patterns.
"""

# ============================================================================
# ZL_DATASET CONFIGURATION
# ============================================================================

# Stream detection patterns (must match XDF stream names)
ZL_EEG_STREAM_PATTERN = 'actiCHamp'
ZL_MARKER_STREAM_PATTERN = 'ZLT-markers'

# File matching pattern
ZL_EEG_FILE_PATTERN = '*.xdf'

# Expected dataset properties (for validation/assertions)
ZL_SAMPLING_RATE = 500.0  # Hz
ZL_NUM_CHANNELS = 96  # Number of EEG channels (excludes AUX and Markers)
ZL_TOTAL_CHANNELS = 99  # Total channels including AUX and Markers
