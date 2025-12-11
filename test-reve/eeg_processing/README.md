# EEG Processing Module

Complete pipeline for preprocessing EEG data and extracting features using the REVE model.

## Overview

The `eeg_processing` module provides:

1. **XDF Loading** (`xdf_loader.py`) - Load EEG data from XDF files
2. **MNE-based Preprocessing** (`preprocessing.py`) - Epoch extraction, filtering, and normalization  
3. **REVE Feature Extraction** (`feature_extraction.py`) - Extract features using the REVE model
4. **Complete Pipeline** (`main.py`) - End-to-end processing

## Key Features

### Workload Classification Focus
- Extracts only workload-related markers (filters by `'workload'` string)
- Ignores task/no-task variants, sparkles, count, and other attributes
- All workload events are treated as the target classification events

### Time Window: [-1.5, +1.5] seconds
- Fixed 3-second epoch around each marker
- Symmetric window centered on event
- Total of 1500 samples per epoch (at 500 Hz sampling rate)

### MNE-Python Preprocessing
- Bandpass filter: 0.5-50 Hz
- Notch filter: 50 Hz (line noise)
- Common Average Reference (CAR)
- Per-epoch z-score normalization

## Usage

### Basic Usage

```python
from eeg_processing.main import process_subject

# Process a single subject
results = process_subject(
    subject_id='sub-PD089',
    session_id='ses-S001',
    extract_features=True,
    save_data=True
)

# Access results
epochs = results['epochs']           # Shape: (n_epochs, 99 channels, 1500 samples)
labels = results['labels']           # Marker names
features = results['features']       # REVE features (if extract_features=True)
metadata = results['metadata']       # Epoch metadata
```

### Using the Preprocessing Class Directly

```python
from eeg_processing.xdf_loader import XDFLoader
from eeg_processing.preprocessing import EEGPreprocessor

# Load XDF data
loader = XDFLoader('sub-PD089', 'ses-S001')
eeg_data, eeg_timestamps, sfreq = loader.get_eeg_data()
markers, marker_timestamps = loader.get_marker_data()
channel_labels = loader.get_channel_labels()

# Preprocess and extract epochs
preprocessor = EEGPreprocessor(
    eeg_data, eeg_timestamps, markers, marker_timestamps,
    channel_labels=channel_labels, sampling_rate=sfreq
)

# Get fully processed epochs with MNE filtering
epochs, labels, metadata = preprocessor.get_processed_epochs(
    preprocess=True,  # Apply MNE filters
    normalize=True,    # Apply z-score normalization
    tmin=-1.5,
    tmax=1.5
)
```

### Configuration

Edit [config.py](config.py) to customize:

- `TMIN` / `TMAX` - Epoch time window
- `SAMPLING_RATE` - EEG sampling rate
- `BANDPASS_LOW` / `BANDPASS_HIGH` - Filter frequencies
- `TARGET_MARKER_PREFIX` - Which markers to extract (`'workload'` by default)
- `SKIP_MARKERS` - Markers to exclude
- `OUTPUT_DIR` - Where to save results

## Output Files

When `save_data=True`, the following files are created:

- `{subject}_{session}_epochs.npy` - Preprocessed epochs array (n_epochs, 99, 1500)
- `{subject}_{session}_labels.npy` - Epoch labels (marker names)
- `{subject}_{session}_features.npy` - REVE features (if extract_features=True)
- `{subject}_{session}_label_features.npy` - Features organized by label
- `{subject}_{session}_metadata.csv` - Epoch metadata

## Example Notebook

See [eeg_preprocessing_mne.ipynb](../eeg_preprocessing_mne.ipynb) for a complete walkthrough:
1. Load and explore markers
2. Create MNE Raw object
3. Extract workload epochs with [-1.5, +1.5] window
4. Apply MNE preprocessing filters
5. Visualize and validate results
6. Save for downstream analysis

## Dependencies

- pyxdf
- numpy
- mne
- torch (for REVE features)
- transformers (for REVE model)

## File Structure

```
eeg_processing/
├── __init__.py                 # Package info
├── config.py                   # Configuration parameters
├── xdf_loader.py              # XDF file loading
├── preprocessing.py           # MNE-based preprocessing
├── feature_extraction.py      # REVE feature extraction  
├── main.py                    # Main pipeline
└── output/                    # Saved results
    ├── sub-PD089_ses-S001_epochs.npy
    ├── sub-PD089_ses-S001_labels.npy
    ├── sub-PD089_ses-S001_features.npy
    └── ...
```

## Notes

- All workload markers (regardless of modifiers) are included in epochs
- Control markers (Recording/Start, Recording/End) are excluded
- Epochs are normalized per-channel using z-score
- MNE provides robust filtering and artifact handling
- REVE features are optional but recommended for classification

