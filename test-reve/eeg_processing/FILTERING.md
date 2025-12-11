# EEG Data Filtering & Quality Control

## Channel Filtering

### Channels Excluded (Case-Insensitive)
The preprocessing automatically removes non-EEG channels:
- `AUX1` - Auxiliary channel 1
- `AUX2` - Auxiliary channel 2  
- `Markers` - Marker channel (duplicate of event stream)

### Result
- **Original**: 99 channels
- **After filtering**: 96 EEG channels (kept)
- **Removed**: 3 non-EEG channels

The channel filtering happens automatically when `EEGPreprocessor` is initialized:
```python
preprocessor = EEGPreprocessor(eeg_data, eeg_timestamps, markers, marker_timestamps)
# Automatically excludes AUX1, AUX2, and Markers channels
```

## Non-Experimental Data Filtering

### Markers Excluded (Case-Insensitive)
The preprocessing filters out non-experimental recording periods:
- `Recording/Start` - Session start marker
- `Recording/End` - Session end marker
- `Break` / `break` - Rest periods
- `Pause` / `pause` - Recording pauses
- `Rest` / `rest` - Rest intervals
- `Baseline` / `baseline` - Baseline recordings

These are **excluded from epoch creation**. Only workload-related events are kept.

### Example
If markers include:
```
Recording/Start
workload_low_task
workload_high_task
Break
workload_high_no_task
Recording/End
```

Only these epochs will be created:
```
workload_low_task
workload_high_task
workload_high_no_task
```

### Case Insensitivity
Matching is case-insensitive, so:
- "BREAK" = "Break" = "break" ✓ (all excluded)
- "Baseline" = "baseline" = "BASELINE" ✓ (all excluded)

## Implementation Details

### Channel Filtering Code
Located in `preprocessing.py > _create_raw_array()`:

```python
for idx, ch_name in enumerate(self.channel_labels):
    # Check if channel should be excluded
    should_exclude = any(exclude_str.lower() in ch_name.lower() 
                        for exclude_str in EXCLUDE_CHANNELS)
    
    if should_exclude:
        self.excluded_indices.append(idx)
        print(f"  Excluding channel: {ch_name}")
    else:
        self.kept_channel_labels.append(ch_name)
        self.kept_eeg_data.append(self.eeg_data[:, idx])
```

**Output:**
```
Channel filtering:
  Original channels: 99
  Kept EEG channels: 96
  Excluded channels: 3
  Excluding channel: AUX1
  Excluding channel: AUX2
  Excluding channel: Markers
```

### Marker Filtering Code
Located in `preprocessing.py > _is_skip_marker()` and `_create_events_array()`:

```python
def _is_skip_marker(self, marker):
    """Check if marker should be skipped (non-experimental data)."""
    marker_lower = marker.lower()
    for skip_pattern in SKIP_MARKERS:
        if skip_pattern.lower() in marker_lower:
            return True
    return False
```

**Output:**
```
Skipped non-experimental markers:
  Recording/Start: 1
  Break: 5
  Recording/End: 1
```

## Configuration

To customize excluded channels or skip markers, edit `config.py`:

```python
# Channels to exclude (case-insensitive substring matching)
EXCLUDE_CHANNELS = ['AUX1', 'AUX2', 'Markers']

# Markers to skip (case-insensitive substring matching)
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
```

## Verification

Run the notebook to see filtering in action:
```python
from eeg_processing.main import process_subject

results = process_subject('sub-PD089', 'ses-S001')
# Console output shows:
#   - Channels excluded and kept
#   - Non-experimental markers skipped
#   - Final epoch count and distribution
```

