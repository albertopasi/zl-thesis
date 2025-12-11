# EEG Processing Pipeline - Complete Flow Documentation

## Overview
The pipeline processes raw EEG data from XDF files into preprocessed epochs ready for REVE feature extraction and workload classification.

**Entry Point**: `main.py` → `process_subject()` or `process_multiple_subjects()`

---

## Pipeline Flow (Step-by-Step)

### Step 0: Configuration Loading
**File**: `config.py`
- Loads all pipeline parameters (sampling rate, time windows, filter settings, etc.)
- Defines data paths, channel exclusions, marker filters
- All downstream modules import from this single source of truth

**Key Parameters**:
- `DATA_ROOT`: Path to data directory
- `SAMPLING_RATE`: 500 Hz
- `TMIN`, `TMAX`: [-1.5, +1.5] seconds
- `BANDPASS_LOW`, `BANDPASS_HIGH`: 0.5-100 Hz
- `EXCLUDE_CHANNELS`: ['AUX1', 'AUX2', 'Markers']
- `TARGET_MARKER_PREFIX`: 'workload'
- `SKIP_MARKERS`: Non-experimental events (breaks, pauses, etc.)

---

## Pipeline Step 1: Data Loading

### 1a. Entry Point: `main.py` → `process_subject()`

```python
def process_subject(subject_id, session_id, extract_features=True, save_data=True):
```

**Input Parameters**:
- `subject_id`: e.g., 'sub-PD089'
- `session_id`: e.g., 'ses-S001'
- `extract_features`: Boolean (default: True)
- `save_data`: Boolean (default: True)

**Output**: Dictionary with keys:
- `epochs`, `labels`, `metadata`
- `features`, `label_features` (if extract_features=True)

---

### 1b. XDF File Loading: `xdf_loader.py` → `XDFLoader` class

**Call**: `loader = XDFLoader(subject_id, session_id)` in `main.py:47`

#### 1b-i: Initialization (`__init__`)
- Constructs file path from subject/session IDs
- **Path format**: `{DATA_ROOT}/{subject}/{session}/eeg/{subject}_{session}_task-demo_workload_run-001_eeg.xdf`
- **Example**: `c:\Users\...\data\sub-PD089\ses-S001\eeg\sub-PD089_ses-S001_task-demo_workload_run-001_eeg.xdf`

#### 1b-ii: Get EEG Data: `loader.get_eeg_data()`
**Call**: `main.py:50`

**Execution flow**:
1. Calls `load()` if not already loaded
2. Loads XDF file using `pyxdf.load_xdf()`
3. Extracts stream at index 4 (EEG data stream: 'actiCHamp-24020239')
4. Returns:
   - `eeg_data`: np.ndarray shape (912513, 99) [samples, channels]
   - `eeg_timestamps`: np.ndarray of LSL timestamps
   - `sampling_rate`: 500 Hz

#### 1b-iii: Get Marker Data: `loader.get_marker_data()`
**Call**: `main.py:51`

**Execution flow**:
1. Extracts stream at index 5 (Marker stream: 'ZLT-markers')
2. Flattens nested marker structure: `[['workload_low']]` → `'workload_low'`
3. Returns:
   - `markers`: List of marker strings (e.g., ['Recording/Start', 'workload_low', ...])
   - `marker_timestamps`: np.ndarray of LSL timestamps

#### 1b-iv: Get Channel Labels: `loader.get_channel_labels()`
**Call**: `main.py:52`

**Execution flow**:
1. Extracts channel info from EEG stream metadata
2. Parses nested structure: `eeg_info['desc'][0]['channels'][0]['channel']`
3. Returns list of 99 channel names: `['Fp1', 'Fpz', 'Fp2', ..., 'AUX1', 'AUX2', 'Markers']`

---

## Pipeline Step 2: Preprocessing & Epoch Extraction

### 2a. Initialize Preprocessor: `preprocessing.py` → `EEGPreprocessor` class

**Call**: `main.py:57-61`

```python
preprocessor = EEGPreprocessor(
    eeg_data, eeg_timestamps, markers, marker_timestamps,
    channel_labels=channel_labels, sampling_rate=sfreq
)
```

#### 2a-i: Initialization (`__init__`)
- Stores all data references
- **Automatically calls**: `_create_raw_array()`

#### 2a-ii: Channel Filtering: `_create_raw_array()`

**Execution flow**:
1. **Identify excluded channels**: Loops through channel_labels
   - Checks if channel name contains 'AUX1', 'AUX2', or 'Markers' (case-insensitive)
   - Excludes: AUX1, AUX2, Markers (3 channels)
   - **Result**: Keeps 96 EEG channels
   
   **Console Output**:
   ```
   Channel filtering:
     Original channels: 99
     Kept EEG channels: 96
     Excluded channels: 3
     Excluding channel: AUX1
     Excluding channel: AUX2
     Excluding channel: Markers
   ```

2. **Filter EEG data**: Keeps only columns for kept channels
   - Input: (912513, 99)
   - Output: (912513, 96)

3. **Create MNE Info object**: `mne.create_info()`
   - Channel names: 96 EEG channel labels
   - Channel types: all 'eeg'
   - Sampling rate: 500 Hz

4. **Create MNE RawArray**: `mne.io.RawArray()`
   - Converts data from (samples, channels) → (channels, samples)
   - Input shape: (912513, 96) → transpose → (96, 912513)
   - Creates MNE Raw object with info and data

---

### 2b. Get Processed Epochs: `preprocessing.get_processed_epochs()`

**Call**: `main.py:64-69`

```python
epochs, epoch_labels, epoch_metadata = preprocessor.get_processed_epochs(
    preprocess=True,
    normalize=True,
    tmin=-1.5,
    tmax=1.5
)
```

#### 2b-i: Preprocess Raw Data (if `preprocess=True`)

**Function**: `preprocess_with_mne()`

1. **Bandpass Filter**: `raw.filter(0.5, 100.0)`
   - High-pass: 0.5 Hz (remove DC drift)
   - Low-pass: 100.0 Hz (remove high-frequency noise)
   - Design: FIR (finite impulse response)
   - **Output**: Filtered raw data

2. **Common Average Reference (CAR)**: `raw.set_eeg_reference('average')`
   - Computes average across all channels
   - Subtracts from each channel
   - Reduces common noise (environmental interference)

**Console Output**:
```
Applying MNE preprocessing...
Applying bandpass filter: 0.5-100 Hz
Applying common average reference...
```

#### 2b-ii: Create Events Array: `_create_events_array()`

**Execution flow**:

1. **Loop through all markers**:
   - For each (marker, marker_timestamp) pair:

2. **Skip non-experimental markers**:
   - Check if marker matches SKIP_MARKERS (case-insensitive substring matching)
   - Skip: 'Recording/Start', 'Recording/End', 'Break', 'Pause', 'Rest', 'Baseline'
   - Track skipped markers with counts
   
   **Console Output**:
   ```
   Skipped non-experimental markers:
     Recording/Start: 1
     Break: 5
     Recording/End: 1
   ```

3. **Filter for workload markers**:
   - Check if 'workload' is in marker name (case-insensitive)
   - Only workload events continue

4. **Convert timestamps to sample indices**:
   - `idx = np.searchsorted(eeg_timestamps, marker_ts)`
   - Finds nearest sample index matching marker timestamp

5. **Create event IDs**:
   - Assign unique numeric ID to each unique marker type
   - Example: 'workload_low' → ID 1, 'workload_high' → ID 2
   - **Returns**: Dictionary mapping marker → event_id

6. **Build events array**:
   - Shape: (num_workload_events, 3)
   - Columns: [sample_index, 0, event_id]
   - Example: [[50000, 0, 1], [75000, 0, 2], ...]

**Console Output**:
```
Found 50 workload events
Event types: {'workload_low_task': 1, 'workload_high_task': 2, 'workload_low_no_task': 3, 'workload_high_no_task': 4}
```

#### 2b-iii: Create Epochs: `extract_epochs()`

**Execution flow**:

1. **MNE Epochs creation**: `mne.Epochs()`
   - **Input**:
     - Raw data: (96, 912513)
     - Events: array of [sample, 0, event_id]
     - Event ID mapping: {'workload_low': 1, ...}
     - Time window: tmin=-1.5, tmax=1.5 seconds
   
   - **Processing**:
     - For each event at sample `idx`:
       - Extract: `[idx + tmin*sfreq : idx + tmax*sfreq]`
       - Time range: [-1.5, +1.5] seconds = 1500 samples
     - Preload all data into memory
   
   - **Output**: MNE Epochs object
     - Shape: (num_epochs, 96, 1500)

2. **Extract epoch data**:
   - `epochs_data = epochs.get_data()`
   - Returns: np.ndarray (num_epochs, 96, 1500)

3. **Extract labels**:
   - Maps event IDs back to marker names
   - Returns: List of labels matching epochs

4. **Create metadata**:
   - For each epoch: marker name, timestamp, sample index, epoch number
   - Returns: List of dictionaries

**Console Output**:
```
Extracted 50 epochs
Epochs shape: (50, 96, 1500)
Label distribution: {'workload_low_task': 12, 'workload_high_task': 13, ...}
```

#### 2b-iv: Normalize Epochs (if `normalize=True`)

**Function**: `normalize_epochs(method='zscore')`

**Execution flow**:

1. **For each epoch** (50 epochs):
   - **For each channel** (96 channels):
     - Extract data: `data = epoch[channel, :]`
     - Calculate: `mean` and `std` of this channel's data
     - Normalize: `(data - mean) / std`

2. **Result**: Per-channel z-score normalized epochs
   - Mean ≈ 0, Std ≈ 1 for each channel in each epoch
   - **Output shape**: (50, 96, 1500) (unchanged)

**Console Output**:
```
Normalizing epochs using zscore...
```

#### 2b-v: Return Processed Data

**Returns**:
- `epochs`: np.ndarray (50, 96, 1500) - preprocessed, filtered, normalized
- `epoch_labels`: List of 50 marker names
- `epoch_metadata`: List of 50 metadata dicts

---

## Pipeline Step 3: REVE Feature Extraction (Optional)

### 3a. Initialize Feature Extractor: `feature_extraction.py` → `REVEFeatureExtractor` class

**Call** (if `extract_features=True`): `main.py:75-76`

```python
feature_extractor = REVEFeatureExtractor()
feature_extractor.load_model()
```

#### 3a-i: Initialization (`__init__`)
- Detects device (GPU if available, else CPU)
- Prints device info

#### 3a-ii: Load REVE Model: `load_model()`

**Execution flow**:

1. **Load base model**: `AutoModel.from_pretrained("brain-bzh/reve-base-model")`
   - Downloads from Hugging Face (requires authentication)
   - Loads REVE neural network
   
2. **Load position bank**: `AutoModel.from_pretrained("brain-bzh/reve-positions-model")`
   - Electrode position embeddings
   
3. **Setup model with classification head**: `setup_model()`
   - Replaces final layer with classification head
   - Final layer: Flatten → RMSNorm → Dropout(0.1) → Linear(final_dim, num_classes)

4. **Move to device**: `model.to(device)`
   - GPU or CPU

5. **Set evaluation mode**: `model.eval()`
   - Disables dropout, batch norm updates

**Console Output**:
```
Loading REVE model from brain-bzh/reve-base-model...
Loading position bank from brain-bzh/reve-positions-model...
REVE model loaded successfully
```

---

### 3b: Extract Features: `feature_extractor.extract_features_for_labels()`

**Call**: `main.py:82-85`

```python
label_features = feature_extractor.extract_features_for_labels(
    epochs, epoch_labels, batch_size=4
)
```

#### 3b-i: Batch Processing: `extract_features()`

**Execution flow**:

1. **Convert to tensor**:
   - Input: np.ndarray (50, 96, 1500)
   - Convert to torch.Tensor, float32

2. **Process in batches** (batch_size=4):
   - Batch 1: epochs 0-3
   - Batch 2: epochs 4-7
   - Batch 3: epochs 8-11
   - ... and so on

3. **For each batch**:
   - Move to device (GPU/CPU)
   - Prepare position embeddings
   - **Forward pass**: `model(batch, positions)`
   - Extract output features
   - Move results back to CPU

4. **Combine all batches**:
   - Stack feature arrays: (num_epochs, feature_dim)
   - Typical feature_dim: ~2048 or as defined by REVE

**Console Output**:
```
Extracting features from 50 epochs...
  Processed 50/50 epochs
Features extracted: shape (50, 2048)
```

#### 3b-ii: Organize by Label: `extract_features_for_labels()`

**Execution flow**:

1. **Get all features**: Shape (50, 2048)

2. **Group by label**:
   - For each unique marker ('workload_low_task', 'workload_high_task', ...):
     - Find indices where label matches
     - Extract features for those indices
   - Return dictionary: {'workload_low_task': (12, 2048), 'workload_high_task': (13, 2048), ...}

**Console Output**:
```
Features by label:
  workload_low_task: (12, 2048)
  workload_high_task: (13, 2048)
  workload_low_no_task: (12, 2048)
  workload_high_no_task: (13, 2048)
```

---

## Pipeline Step 4: Save Results

### 4a. Save Function: `main.py` → `_save_results()`

**Call** (if `save_data=True`): `main.py:90`

```python
_save_results(results)
```

#### 4a-i: Create Output Directory
- Path: `eeg_processing/output/`
- Creates if doesn't exist

#### 4a-ii: Save Epochs Array
- **File**: `sub-PD089_ses-S001_epochs.npy`
- **Shape**: (50, 96, 1500)
- **Type**: np.ndarray (float32)

#### 4a-iii: Save Labels
- **File**: `sub-PD089_ses-S001_labels.npy`
- **Type**: np.ndarray of strings
- **Content**: ['workload_low_task', 'workload_high_task', ...]

#### 4a-iv: Save Features (if extracted)
- **File**: `sub-PD089_ses-S001_features.npy`
- **Shape**: (50, 2048)
- **Type**: np.ndarray (float32)

#### 4a-v: Save Label-Organized Features (if extracted)
- **File**: `sub-PD089_ses-S001_label_features.npy`
- **Type**: Dictionary of np.ndarrays
- **Content**: {'workload_low_task': (12, 2048), ...}

**Console Output**:
```
[Step 4] Saving results...
  Saved epochs: eeg_processing/output/sub-PD089_ses-S001_epochs.npy
  Saved labels: eeg_processing/output/sub-PD089_ses-S001_labels.npy
  Saved features: eeg_processing/output/sub-PD089_ses-S001_features.npy
  Saved label features: eeg_processing/output/sub-PD089_ses-S001_label_features.npy
```

---

## Pipeline Step 5: Return Results

### 5a: Return Dictionary

**Call**: `main.py:92`

```python
return results
```

**Dictionary structure**:
```python
{
    'subject_id': 'sub-PD089',
    'session_id': 'ses-S001',
    'epochs': np.ndarray (50, 96, 1500),
    'labels': ['workload_low_task', 'workload_high_task', ...],
    'metadata': [
        {'marker': 'workload_low_task', 'marker_timestamp': ..., ...},
        ...
    ],
    'features': np.ndarray (50, 2048),  # if extract_features=True
    'label_features': {                   # if extract_features=True
        'workload_low_task': np.ndarray (12, 2048),
        'workload_high_task': np.ndarray (13, 2048),
        ...
    }
}
```

---

## End-to-End Example Call

### Single Subject Processing

```python
from eeg_processing.main import process_subject

# Run pipeline
results = process_subject(
    subject_id='sub-PD089',
    session_id='ses-S001',
    extract_features=True,
    save_data=True
)

# Access results
epochs = results['epochs']              # (50, 96, 1500)
labels = results['labels']              # ['workload_low_task', ...]
features = results['features']          # (50, 2048)
```

### Multiple Subjects Processing

```python
from eeg_processing.main import process_multiple_subjects

results_all = process_multiple_subjects(
    subject_ids=['sub-PD089', 'sub-PD094'],
    extract_features=True,
    save_data=True
)
```

### Command Line Usage

```bash
cd test-reve/eeg_processing
python main.py
```

**Default behavior**:
- Processes subjects: ['sub-PD089', 'sub-PD094']
- Extracts REVE features
- Saves all results

---

## Key Numbers Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Input Channels** | 99 | AUX1, AUX2, Markers |
| **Output Channels** | 96 | EEG channels only |
| **Sampling Rate** | 500 Hz | From XDF metadata |
| **Epoch Duration** | 3 seconds | [-1.5, +1.5] |
| **Samples per Epoch** | 1500 | 500 Hz × 3 sec |
| **Filter Range** | 0.5-100 Hz | Bandpass (high-pass + low-pass) |
| **Example: # Epochs** | ~50 | Workload events only |
| **Feature Dimension** | ~2048 | REVE model output |

---

## Status: ✅ READY TO RUN

**All requirements met**:
- ✅ XDF loading and parsing
- ✅ Channel filtering (keep 96 EEG, exclude AUX1/AUX2/Markers)
- ✅ Marker filtering (workload only, skip non-experimental)
- ✅ Epoch extraction [-1.5, +1.5] seconds
- ✅ MNE preprocessing (bandpass 0.5-100 Hz, CAR reference)
- ✅ Per-epoch normalization (z-score)
- ✅ REVE feature extraction (optional)
- ✅ Results saving (numpy arrays + metadata)

**To run**:
```bash
python main.py
```

or

```python
from eeg_processing.main import process_subject
results = process_subject('sub-PD089', 'ses-S001')
```
