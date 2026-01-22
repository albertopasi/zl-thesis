# ZL_Dataset Preprocessing Pipeline

## Overview
The preprocessing pipeline in `src/preprocess/mne_preprocessor.py` implements the same processing steps as `test-reve/eeg_processing/preprocessing.py` with full configuration through `src/preprocess/preprocess_config.py`.

## Pipeline Steps (in order)

### 1. **Channel Filtering**
   - **Purpose**: Remove non-EEG channels before processing
   - **Configuration**: `EXCLUDE_CHANNELS = ['AUX_1', 'AUX_2', 'Markers']`
   - **Location**: `_create_raw_array()` method
   - **Output**: MNE RawArray with only EEG channels (96 of 99 channels)

### 2. **Bandpass Filter**
   - **Purpose**: Remove low-frequency noise and high-frequency artifacts
   - **Frequency Range**: 0.5 - 99.5 Hz
   - **Configuration**: `MNE_BANDPASS_LOW = 0.5` Hz, `MNE_BANDPASS_HIGH = 99.5` Hz
   - **Location**: `preprocess()` method, step 1
   - **Implementation**: MNE's `raw.filter()` with default FIR design

### 3. **Downsampling**
   - **Purpose**: Reduce computational load and file size
   - **Target Sampling Rate**: 200 Hz (from original 500 Hz)
   - **Configuration**: `DOWNSAMPLE_RATE = 200` Hz
   - **Location**: `preprocess()` method, step 2
   - **Implementation**: MNE's `raw.resample()`
   - **Note**: Updates internal sampling_rate after resampling

### 4. **Event Creation & Marker Filtering**
   - **Purpose**: Identify task/no-task boundary markers in the signal
   - **Configuration**:
     - `TASK_MARKER_PREFIX = 'task'` - identifies task events
     - `NO_TASK_MARKER_PREFIX = 'no task'` - identifies no-task events
     - `SKIP_MARKERS` - set of marker names to exclude (Recording/Start, Recording/End, breaks, etc.)
   - **Location**: `_create_events_array()` method
   - **Output**: MNE events array with event IDs (0 = no task, 1 = task)

### 5. **Epoch Extraction**
   - **Purpose**: Extract time windows around each marker event
   - **Time Window**: [`EPOCH_TMIN seconds`, `EPOCH_TMIN seconds`] around each marker
   - **Configuration** (from config): `EPOCH_TMIN = -1.5`, `EPOCH_TMAX = 1.5`
   - **Location**: `extract_epochs()` method
   - **Implementation**: MNE's `mne.Epochs()`
   - **Output**: Array of shape (n_epochs, n_channels, n_samples)
   - **Note**: Handles overlapping epochs with `event_repeated='merge'`

### 6. **Per-Epoch Z-Score Normalization**
   - **Purpose**: Normalize each epoch independently to zero mean and unit variance
   - **Configuration**: `NORMALIZE_METHOD = 'zscore'`, `NORMALIZE_PER_EPOCH = True`
   - **Location**: `normalize()` method
   - **Formula**: $z = \frac{x - \mu}{\sigma}$ (computed per-channel per-epoch)
   - **Implementation**: Triple-nested loop (n_epochs × n_channels × sample scaling)

## Summary Table

| Step | Component | Input | Output | Key Config |
|------|-----------|-------|--------|------------|
| 1 | Channel Filter | 99 channels | 96 EEG channels | `EXCLUDE_CHANNELS` |
| 2 | Bandpass Filter | Raw 500 Hz | Filtered | `0.5-99.5 Hz` |
| 3 | Downsample | 500 Hz | **200 Hz** | `DOWNSAMPLE_RATE=200` |
| 4 | Event Creation | Markers + config | Event array | `TASK_MARKER_PREFIX`, `NO_TASK_MARKER_PREFIX` |
| 5 | Epoch Extraction | Preprocessed + events | Epochs array | `[-1.5, +1.5]s` |
| 6 | Z-Score Normalize | Epochs | Normalized epochs | `NORMALIZE_METHOD='zscore'` |

## Files Modified

### `src/preprocess/preprocess_config.py`
- **Added**: `DOWNSAMPLE_RATE = 200`
- **Fixed**: Filter frequencies already correct (0.5-99.5 Hz)
- **Status**: ✅ Complete

### `src/preprocess/mne_preprocessor.py`
- **Added**: Import of `DOWNSAMPLE_RATE`
- **Fixed**: Pipeline docstring to show correct order
- **Updated**: `preprocess()` method to include downsampling step
- **Status**: ✅ Complete

## Verification Checklist

- [x] Filter: 0.5-99.5 Hz
- [x] Downsample: 500 Hz → 200 Hz (step 3, after filtering, before epoch extraction)
- [x] Epoch extraction: [-1.5s, +1.5s] around markers
- [x] Z-score normalization: Per-channel per-epoch after extraction
- [x] Channel exclusion: AUX_1, AUX_2, Markers removed before processing
- [x] Marker filtering: SKIP_MARKERS excludes Recording/Start, Recording/End, breaks
- [x] CAR reference: Disabled (not applied)

## Preprocessing Configuration

The pipeline is fully configured via `src/preprocess/preprocess_config.py`:

```python
MNE_BANDPASS_LOW = 0.5        # Hz
MNE_BANDPASS_HIGH = 99.5      # Hz
APPLY_CAR_REFERENCE = False   # Disabled
DOWNSAMPLE_RATE = 200         # Hz
EPOCH_TMIN = -1.5             # seconds
EPOCH_TMAX = 1.5              # seconds
NORMALIZE_METHOD = 'zscore'
NORMALIZE_PER_EPOCH = True
```
