# MNE Preprocessing Concepts: A Comprehensive Guide

## Overview

This document explains the key concepts used in MNE-based EEG preprocessing, with particular focus on how timestamps, events, and epochs interact with downsampling and data transformations.

---

## 1. Sampling Rate and Time Representation

### 1.1 Sampling Rate (`sfreq`)

The **sampling frequency** determines how many samples are recorded per second. In our ZL_Dataset:
- **Original sampling rate**: 500 Hz (500 samples/second)
- **Target sampling rate**: 200 Hz (200 samples/second after downsampling)

Each sample represents a voltage measurement at a specific point in time.

### 1.2 Time vs. Sample Index

There are two ways to reference a point in time in MNE data:

| Representation | Units | Example | When Used |
|---|---|---|---|
| **Absolute time** | Seconds from recording start | 3.5 seconds | Annotations, marker timestamps |
| **Sample index** | Sample number from start | Sample 1750 | Events array, raw data indexing |

**Conversion formula:**
```
sample_index = time_in_seconds × sampling_rate
time_in_seconds = sample_index / sampling_rate
```

**Example at 500 Hz:**
- 2 seconds = 2 × 500 = 1000 samples
- Sample 2500 = 2500 / 500 = 5 seconds

---

## 2. Timestamps and Recording Alignment

### 2.1 Absolute vs. Relative Timestamps

EEG recordings typically come with timestamps in two forms:

**Absolute timestamps** (wall-clock time):
```python
eeg_timestamps = [1674325600.123, 1674325600.125, 1674325600.127, ...]  # Unix time
marker_timestamps = [1674325603.456, 1674325608.789, ...]  # Unix time
```

**Relative timestamps** (relative to recording start):
```python
t_start = eeg_timestamps[0]  # Recording start time
relative_time = eeg_timestamps[i] - t_start  # Time since start
# Result: [0.0, 0.002, 0.004, ...] seconds
```

### 2.2 Why MNE Requires Relative Timestamps

MNE RawArray expects data and annotations to be indexed relative to the recording start, not absolute wall-clock time:

```python
# ❌ WRONG: Using absolute timestamps
self.raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
annotations = mne.Annotations(onset=[1674325603.456], ...)  # Won't align!

# ✅ CORRECT: Using relative timestamps
t_start = eeg_timestamps[0]
relative_marker_time = marker_timestamps[0] - t_start
annotations = mne.Annotations(onset=[relative_marker_time], ...)  # Aligns correctly
```

The key insight: **MNE annotations are tied to data indices, not absolute time**. When you create a RawArray, time 0.0 corresponds to the first sample, time 0.002s to the second sample (at 500 Hz), etc.

---

## 3. Events Array: Structure and Generation

### 3.1 Event Array Format

An **event** is a discrete point in time where something noteworthy happened (e.g., a task started, stimulus presented, marker recorded).

MNE represents events as a 2D numpy array with shape `(n_events, 3)`:

```python
events = np.array([
    [500,   0, 1],      # Row 0: Sample 500, duration 0, event_id 1
    [1200,  0, 2],      # Row 1: Sample 1200, duration 0, event_id 2
    [1800,  0, 1],      # Row 2: Sample 1800, duration 0, event_id 1
])
```

**Columns:**
- **Column 0**: Sample index (integer) where the event occurs
- **Column 1**: Duration in samples (0 for instantaneous events like markers)
- **Column 2**: Event ID (identifies the event type, must be integer)

### 3.2 Event ID Mapping

Event IDs are arbitrary integers that label event types:

```python
event_id = {
    'no task': 1,  # Marker ID 1 = no-task event
    'task': 2      # Marker ID 2 = task event
}
```

This mapping tells MNE which samples correspond to which events.

### 3.3 Creating Events from Annotations

In the preprocessing pipeline, we create events in two stages:

**Stage 1: Create Annotations (time-based)**
```python
# Annotations are tied to absolute time (seconds from recording start)
filtered_markers = [
    ('Task Marker', 3.5, 1),   # (name, time, label)
    ('No-Task Marker', 7.2, 0)
]

for marker, marker_ts, label in filtered_markers:
    t_start = eeg_timestamps[0]
    relative_time = marker_ts - t_start  # Convert to relative time
    annotations_onsets.append(relative_time)
    annotations_descriptions.append('task' if label == 1 else 'no task')

annotations = mne.Annotations(
    onset=annotations_onsets,         # Times in seconds [3.5, 7.2]
    duration=[0, 0],
    description=['task', 'no task']
)
self.raw.set_annotations(annotations)
```

**Stage 2: Extract Events (sample-based)**
```python
# MNE converts annotations to events, adjusting for current sampling rate
events, event_id = mne.events_from_annotations(
    self.raw,
    event_id={'no task': 1, 'task': 2}
)

# Result: events array with sample indices
# events = [[1750, 0, 2], [3600, 0, 1]]
# (assuming 500 Hz: 3.5s × 500 = 1750 samples, 7.2s × 500 = 3600 samples)
```

---

## 4. Downsampling and Automatic Re-indexing

### 4.1 What Downsampling Does

Downsampling reduces the number of samples per second from 500 Hz to 200 Hz:

```
Original:   [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] at 500 Hz
Downsampled: [s0, s2, s5, s7, s9] at 200 Hz
```

**Key metric**: Downsampling by factor 2.5 means we keep every 2.5th sample.

### 4.2 Automatic Annotation Re-indexing

**Critical MNE behavior**: When you call `raw.resample(new_rate)`, MNE **automatically updates all annotations** to maintain correct alignment:

```python
# Initial state (500 Hz):
raw.info['sfreq'] = 500
annotations.onset = [3.5]  # 3.5 seconds from start

# After resampling to 200 Hz:
raw.resample(200, verbose=False)
raw.info['sfreq'] = 200    # ← Updated

# Annotations are automatically adjusted:
# The annotation still represents the same moment in time (3.5 seconds)
# But MNE knows the sampling rate changed, so sample indices will differ
```

### 4.3 Sample Index Changes During Downsampling

When downsampling from 500 Hz to 200 Hz, sample indices are scaled accordingly:

```python
# At 500 Hz:
# Marker at 3.5 seconds = sample index 1750 (3.5 × 500)

# After downsampling to 200 Hz:
# Same marker at 3.5 seconds = sample index 700 (3.5 × 200)

# When MNE extracts events from annotations AFTER resampling,
# it uses the new sampling rate to calculate sample indices
events, _ = mne.events_from_annotations(self.raw)
# Sample indices in events array are now based on 200 Hz rate
```

**Why this matters**: If you extract events BEFORE resampling vs AFTER, you get different sample indices but they represent the same temporal location.

### 4.4 Workflow in Preprocessing Pipeline

```python
# Step 1: Create RawArray (500 Hz)
self.raw = mne.io.RawArray(eeg_data_filtered.T, info, verbose=False)
# raw.info['sfreq'] = 500

# Step 2: Apply filter (doesn't change sfreq)
self.raw.filter(0.5, 99.5, verbose=False)
# raw.info['sfreq'] = 500

# Step 3: Downsample (changes sfreq)
self.raw.resample(200, verbose=False)
# raw.info['sfreq'] = 200  ← Updated automatically
# All annotations preserved and re-indexed

# Step 4: Extract events (uses current sampling rate)
events, event_id = mne.events_from_annotations(self.raw)
# Sample indices now use 200 Hz rate
```

---

## 5. Epochs: Extracting Time Windows

### 5.1 Epoch Definition

An **epoch** is a fixed-duration time window of EEG data centered around an event:

```
Raw EEG:     [====== event (sample 1000) ======]
             t=-1.5s          t=0s          t=1.5s

Epoch window: ├─ tmin=-1.5s ─┤├─ tmax=1.5s ─┤
             Sample: [700 ... 1000 ... 1300]
             Duration: 3 seconds = 600 samples (at 200 Hz)
```

**Parameters:**
- `tmin`: Time relative to event where epoch starts (negative = before event)
- `tmax`: Time relative to event where epoch ends (positive = after event)
- Duration: `(tmax - tmin) × sampling_rate` = `(1.5 - (-1.5)) × 200` = 600 samples

### 5.2 Creating Epochs from Events

```python
epochs = mne.Epochs(
    self.raw,                    # RawArray (channels, samples)
    events,                      # Events array (n_events, 3)
    event_id,                    # Mapping {'no task': 1, 'task': 2}
    tmin=-1.5,                   # 1.5 seconds before event
    tmax=1.5,                    # 1.5 seconds after event
    baseline=None,               # No baseline correction
    preload=True,                # Load all data into memory
    reject_by_annotation=False,  # Don't reject bad segments
    event_repeated='merge'       # Merge overlapping epochs
)
```

**Output**: Epochs object with shape `(n_epochs, n_channels, n_samples)`
- `n_epochs`: Number of valid events
- `n_channels`: 95 (EEG channels after removing AUX/Markers)
- `n_samples`: 600 (3 seconds × 200 Hz)

### 5.3 Epoch Extraction Example

```
Events (at 200 Hz):
[[700,  0, 1],    # Event 1: Sample 700, no-task
 [1100, 0, 2],    # Event 2: Sample 1100, task
 [1500, 0, 1]]    # Event 3: Sample 1500, no-task

With tmin=-1.5s, tmax=1.5s (600 samples per epoch):

Epoch 0 (no-task): samples [400, 1000]   (700 - 300 to 700 + 300)
Epoch 1 (task):    samples [800, 1400]   (1100 - 300 to 1100 + 300)
Epoch 2 (no-task): samples [1200, 1800] (1500 - 300 to 1500 + 300)

Result: 3 epochs of shape (3, 95, 600)
Labels: [1, 2, 1]  (no-task, task, no-task)
```

### 5.4 Overlapping Epochs and Edge Cases

**Problem**: If two events are too close, their epoch windows overlap.

```
Event A at sample 1000:  [700 ... 1000 ... 1300]
Event B at sample 1200:  [900 ... 1200 ... 1500]
                         ^^^^^^ Overlap! ^^^^^^
```

**MNE Behavior** (`event_repeated='merge'`):
- **Merge**: Combines overlapping epochs into one
- **Keep first**: Keeps first epoch, drops overlapping
- **Drop**: Removes both overlapping epochs
- **error**: Raises an exception

**Edge case**: Epochs extending beyond recording boundaries are also dropped.

---

## 6. Shape Conventions and Transformations

### 6.1 Data Shape Throughout Pipeline

| Stage | Shape | Format | Reason |
|---|---|---|---|
| **Raw acquisition** | `(samples, channels)` | Time-first | Natural for recording |
| **Into MNE** | `(channels, samples)` | Channel-first | MNE standard |
| **Epochs** | `(n_epochs, n_channels, n_samples)` | 3D tensor | Groups time windows |

### 6.2 Example Transformation

```python
# Step 1: Load from file (time-first)
eeg_data = load_xdf_data()
print(eeg_data.shape)  # (30000, 96) - 60 seconds at 500 Hz, 96 channels

# Step 2: Filter channels (time-first)
eeg_data_filtered = eeg_data[:, keep_indices]
print(eeg_data_filtered.shape)  # (30000, 95) - removed 1 non-EEG channel

# Step 3: Transpose for MNE (channel-first)
data_for_mne = eeg_data_filtered.T
print(data_for_mne.shape)  # (95, 30000)

# Step 4: Create RawArray
raw = mne.io.RawArray(data_for_mne, info, verbose=False)
print(raw.get_data().shape)  # (95, 30000)

# Step 5: Downsample 500 Hz → 200 Hz
raw.resample(200)
print(raw.get_data().shape)  # (95, 12000) - 60 seconds at 200 Hz

# Step 6: Extract epochs
epochs_data = epochs.get_data()
print(epochs_data.shape)  # (45, 95, 600) - 45 epochs, 95 channels, 600 samples
```

---

## 7. Marker Processing and Event Generation

### 7.1 ZL_Dataset Marker Filtering

Raw markers from ZL_Dataset contain noise and non-experimental events:

```
Raw markers:
  "Recording Started"           ← Non-experimental
  "Break"                       ← Non-experimental
  "Task Marker (Def/PD, Onset task)"    ← Filtered (onset/offset)
  "Task Marker (Def/PD)"        ← Valid (task event)
  "No Task Marker (Def/PD)"     ← Valid (no-task event)
  "Rest"                        ← Non-experimental
```

**Processing stages:**

1. **Skip non-experimental**: Remove markers matching skip patterns
   - Input: 200 markers
   - Output: 150 markers (50 skipped)

2. **Filter onset/offset**: Remove boundary markers
   - Input: 150 markers
   - Output: 100 markers (50 filtered)

3. **Deduplicate**: Keep only one marker per consecutive group of same label
   - Input: 100 markers (many consecutive duplicates)
   - Output: 80 markers (20 deduplicated)

4. **Extract binary labels**:
   - "Task" → 1
   - "No Task" → 0

**Result**: 80 clean events with accurate timestamps and binary labels.

### 7.2 Why Deduplication Matters

Duplicate markers represent electrical noise or repeated trigger signals, not separate experimental events:

```
Raw markers at (seconds):
  3.50: "Task"      ├─ Same event
  3.51: "Task"      ├─ Duplicated
  3.52: "Task"      ┘

After deduplication (keep middle):
  3.51: "Task"      ← One representative marker
```

---

## 8. Per-Epoch Normalization

### 8.1 Why Normalize Each Epoch Independently

EEG amplitude varies across time and between channels. Z-score normalization within each epoch:
- Centers data around 0 (mean removal)
- Scales to unit variance (standard deviation normalization)
- **Applied per-channel, per-epoch**

```python
# For each epoch:
for i in range(n_epochs):
    epoch = epochs[i]  # Shape: (95, 600)
    
    # Per-channel statistics
    mean = epoch.mean(axis=1, keepdims=True)      # Shape: (95, 1)
    std = epoch.std(axis=1, keepdims=True)        # Shape: (95, 1)
    
    # Z-score: (x - mean) / std
    normalized = (epoch - mean) / std             # Shape: (95, 600)
```

### 8.2 Effect on Data

```
Original epoch (channel 1):  [100, 110, 105, 115, 108, ...] (μV units)
Mean: 110, Std: 5

Normalized:                  [-2.0, 0.0, -1.0, 1.0, -0.4, ...] (σ units)
```

**Benefits:**
- Removes amplitude bias between channels
- Makes features comparable across epochs
- Improves machine learning model convergence

---

## 9. Timeline: From Recording to Epochs

### Complete Processing Flow

```
1. RAW XDF FILE
   └─ Contains: EEG data (500 Hz), markers (timestamps), metadata

2. LOAD DATA
   └─ eeg_data: (30000, 96)          [60 sec, 500 Hz, 96 channels]
   └─ markers: ['Task Marker', ...]  [raw marker strings]
   └─ timestamps: [1674325600.123, ...]  [absolute wall-clock time]

3. CONVERT TO RELATIVE TIME
   └─ t_start = timestamps[0]
   └─ relative_times = timestamps - t_start
   └─ Result: [0.0, 0.002, 0.004, ...] seconds from recording start

4. EXCLUDE NON-EEG CHANNELS
   └─ eeg_data_filtered: (30000, 95)  [removed AUX and Markers channels]

5. CREATE RAW ARRAY
   └─ Transpose: (95, 30000)  [channels-first for MNE]
   └─ raw = mne.io.RawArray(data.T, info)
   └─ sampling_rate: 500 Hz

6. APPLY BANDPASS FILTER
   └─ raw.filter(0.5, 99.5)
   └─ Removes DC offset and high-frequency noise
   └─ Shape unchanged: (95, 30000)

7. DOWNSAMPLE
   └─ raw.resample(200)
   └─ New shape: (95, 12000)  [same 60 seconds, fewer samples]
   └─ sampling_rate: 200 Hz (updated automatically)

8. PROCESS MARKERS
   └─ filtered_markers = marker_handler.process_markers(markers, timestamps)
   └─ Result: [(name, time, label), ...]  [cleaned, deduplicated]

9. CREATE ANNOTATIONS
   └─ Convert absolute timestamps to relative times
   └─ Create mne.Annotations with onset times (seconds)
   └─ Attach to raw: raw.set_annotations(annotations)

10. EXTRACT EVENTS
    └─ events, event_id = mne.events_from_annotations(raw)
    └─ events: (n_events, 3)  [sample indices at 200 Hz, IDs]
    └─ Example: [[350, 0, 1], [550, 0, 2], [750, 0, 1]]

11. CREATE EPOCHS
    └─ epochs = mne.Epochs(raw, events, event_id, tmin=-1.5, tmax=1.5)
    └─ Extracts 3-second windows around each event
    └─ Result: (45, 95, 600)  [45 epochs, 95 channels, 600 samples]

12. NORMALIZE (OPTIONAL)
    └─ normalized = preprocessor.normalize_epochs()
    └─ Z-score per channel per epoch
    └─ Result: (45, 95, 600)  [same shape, normalized values]

FINAL OUTPUT
└─ epochs: (45, 95, 600)         [EEG signal at 200 Hz]
└─ labels: [1, 0, 1, 2, 0, ...]  [binary labels (0=no-task, 1=task)]
└─ metadata: [...]               [timestamps, sample indices, etc.]
```

---

## 10. Common Pitfalls and Debugging

### 10.1 Timestamps Mismatch

**Problem**: Annotations don't align with data
```python
# ❌ WRONG: Using absolute timestamps
annotations = mne.Annotations(onset=marker_timestamps, ...)  # [1674325603.456, ...]

# ✅ CORRECT: Use relative timestamps
t_start = eeg_timestamps[0]
relative_times = marker_timestamps - t_start
annotations = mne.Annotations(onset=relative_times, ...)  # [3.456, ...]
```

### 10.2 Resampling Before Event Extraction

**Problem**: Events extracted before resampling have different sample indices
```python
# ❌ Extract events BEFORE resampling
events, _ = mne.events_from_annotations(raw)  # Indices at 500 Hz
raw.resample(200)

# ✅ Extract events AFTER resampling
raw.resample(200)
events, _ = mne.events_from_annotations(raw)  # Indices at 200 Hz (correct)
```

### 10.3 Overlapping Epochs

**Problem**: Events too close together cause epoch loss
```python
# Check for overlaps
overlaps = 0
for i in range(len(events) - 1):
    curr_end = events[i, 0] + int(1.5 * sampling_rate)
    next_start = events[i+1, 0] - int(1.5 * sampling_rate)
    if curr_end >= next_start:
        overlaps += 1

if overlaps > 0:
    print(f"Warning: {overlaps} overlapping epoch windows detected")
```

### 10.4 Shape Confusion

Remember the convention:
- **Input data**: `(samples, channels)` - natural time order
- **MNE RawArray**: `(channels, samples)` - requires transpose
- **Epochs**: `(n_epochs, n_channels, n_samples)` - 3D tensor

---

## 11. Reference: Key MNE Classes and Methods

| Class/Function | Purpose | Input | Output |
|---|---|---|---|
| `mne.create_info()` | Create channel metadata | `ch_names, sfreq, ch_types` | `Info` object |
| `mne.io.RawArray()` | Create raw data container | Data, Info | `Raw` object |
| `Raw.filter()` | Apply bandpass filter | `l_freq, h_freq` | Modified `Raw` |
| `Raw.resample()` | Change sampling rate | `sfreq` | Modified `Raw`, auto-reindexes annotations |
| `mne.Annotations()` | Create event markers | `onset, duration, description` | `Annotations` object |
| `Raw.set_annotations()` | Attach annotations | `Annotations` | Modifies `Raw` |
| `mne.events_from_annotations()` | Convert annotations to events | `Raw, event_id` | Events array, event_id dict |
| `mne.Epochs()` | Extract time windows | `Raw, events, event_id, tmin, tmax` | `Epochs` object |
| `Epochs.get_data()` | Convert to numpy | - | NumPy array `(n_epochs, n_channels, n_samples)` |

---

## 12. Summary Checklist

When working with MNE epochs in ZL_Dataset preprocessing:

- [ ] Convert absolute timestamps to relative (subtract t_start)
- [ ] Transpose data from `(samples, channels)` to `(channels, samples)` before RawArray
- [ ] Create Annotations with relative times, not absolute
- [ ] Extract events AFTER resampling (not before)
- [ ] Resampling automatically updates sampling_rate and re-indexes annotations
- [ ] Epoch window duration = `(tmax - tmin) × sampling_rate`
- [ ] Check for overlapping epochs and boundary cases
- [ ] Normalize per-channel, per-epoch (not global)
- [ ] Final epochs shape = `(n_epochs, n_channels, n_samples)`
- [ ] Remember: MNE works with relative time (seconds from start) and sample indices

