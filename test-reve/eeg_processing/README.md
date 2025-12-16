# EEG Processing Module with REVE Feature Extraction

Complete pipeline for preprocessing EEG data and extracting features using the REVE model with **actual 3D electrode coordinates** from CapTrak.

## Overview

The `eeg_processing` module provides:

1. **XDF Loading** (`xdf_loader.py`) - Load EEG data from XDF files with CapTrak electrode positions
2. **MNE-based Preprocessing** (`preprocessing.py`) - Bandpass filtering, CAR reference, optional downsampling, epoch extraction, normalization  
3. **REVE Feature Extraction** (`feature_extraction.py`) - Extract features using REVE's 4D positional encoding with actual 3D coordinates
4. **Complete Pipeline** (`main.py`) - End-to-end multi-subject processing

## Key Features

### REVE 4D Positional Encoding with Arbitrary Electrode Configuration Support

**Unlike prior work**, REVE doesn't rely on learned position embeddings. Instead, it uses a **4D Fourier-based positional encoding** that leverages:

1. **Actual 3D coordinates** from your CapTrak system (x, y, z in mm)
2. **Temporal information** (patch timesteps in the sequence)

This enables the model to:
- ✅ Handle **arbitrary electrode configurations** (different montages, densities)
- ✅ Generalize to **unseen electrode layouts** 
- ✅ Use your **actual measured positions** without position embedding lookup tables
- ✅ Process **variable sequence lengths** without retraining

**How coordinates are used:**
```
electrode_positions.json (CapTrak: 96 electrodes × 3D coords)
         ↓
feature_extraction.py::_load_electrode_coordinates()
         ↓
reve_model.py::setup_model(electrode_coordinates=...)
         ↓
feature_extraction.py::extract_features()
    - Reshape: (C, 3) → (B, C, 3) for batch
    - Pass to model: model(eeg_batch, positions)
    - REVE computes 4D encoding: (B, C, p, 4) where p=patches, then Fourier basis
    - Output: Features leveraging spatial geometry + temporal structure
```

### Workload Classification Focus
- Extracts only workload-related markers (filters by `'workload'` string)
- Ignores task/no-task variants, sparkles, count, and other attributes
- Extracts 4 workload levels: original, low, medium, high

### Preprocessing Pipeline
1. **Bandpass filter**: 0.5-99.5 Hz (anti-aliasing included)
2. **Common Average Reference**: CAR across all 96 channels
3. **Downsampling** (optional): Resample to target rate (e.g., 250 Hz)
4. **Epoch extraction**: [-1.5, +1.5] seconds around markers (3 seconds total)
5. **Normalization**: Per-channel z-score normalization per epoch

### Configurable Downsampling
Enable downsampling in `config.py`:
```python
DOWNSAMPLE_RATE = 250  # Downsample from 500 Hz to 250 Hz
# DOWNSAMPLE_RATE = None  # Disable downsampling
```

## File Structure

```
eeg_processing/
├── README.md                              This file
├── REVE_COORDINATES_GUIDE.md             Technical guide on 4D encoding
├── PIPELINE.md                           Pipeline architecture
├── FILTERING.md                          Filter specifications
│
├── config.py                             Configuration parameters (data, training, evaluation)
├── main.py                               Main pipeline orchestrator (preprocessing)
├── test_pipeline.py                      Quick test on single subject
│
├── xdf_loader.py                         Load XDF files + extract electrode positions
├── preprocessing.py                      MNE-based filtering, downsampling, epoching
├── feature_extraction.py                 Load coordinates, extract REVE features
├── reve_model.py                         Setup REVE with 3D coordinates
│
├── train_comprehensive.py                ⭐ Comprehensive evaluation with 5 scenarios
├── analyze_results.py                    ⭐ Detailed results analysis & summary statistics
│
├── output/                               Generated features & training results
│   ├── sub-PD089_ses-S001_epochs.npy     Preprocessed epochs
│   ├── sub-PD089_ses-S001_labels.npy     Epoch labels
│   ├── sub-PD089_ses-S001_features.npy   REVE features (793, 393216)
│   ├── sub-PD094_ses-S001_epochs.npy     
│   ├── sub-PD094_ses-S001_labels.npy     
│   ├── sub-PD094_ses-S001_features.npy   REVE features (797, 393216)
│   └── comprehensive_results.json        ⭐ All evaluation metrics (5 scenarios × 3 seeds)
│
├── electrodes_pos/
│   ├── electrode_positions.json          96 CapTrak electrodes (x, y, z in mm)
│   ├── electrode_mapping_to_standard.json [deprecated] Old mapping system
│   └── reve_all_positions.json           [not used] Old REVE position embeddings
│
└── inspect_electrodes/
    ├── generate_electrode_positions.py   Extract positions from XDF CapTrak metadata
    ├── extract_reve_positions.py         [deprecated]
    ├── map_to_standard_positions.py      [deprecated]
    └── ...other utilities
```

## How Coordinates Are Passed to REVE

### Step-by-Step Data Flow

#### 1. **Load Coordinates** (`feature_extraction.py::_load_electrode_coordinates()`)


#### 2. **Pass to Model Setup** (`feature_extraction.py::load_model()`)
```python
self.model, self.positions = setup_model(
    self.model, 
    self.pos_bank, 
    num_classes=num_classes,
    electrode_coordinates=self.electrode_coordinates  # ← Pass actual 3D coords
)
# Result: self.positions shape = (96, 3) tensor
```

#### 3. **REVE Model Configuration** (`reve_model.py::setup_model()`)
```python
if electrode_coordinates is not None:
    # Use actual 3D coordinates (no learned embedding lookup)
    positions_array = electrode_coordinates[:NUM_CHANNELS]
    positions = torch.from_numpy(positions_array).float()
    
    print(f"Position tensor shape: {positions.shape}")  # (96, 3)
    print(f"X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
    # X: [-86.2, 81.5] mm
    # Y: [-113.0, 110.6] mm
    # Z: [-36.0, 149.1] mm
    
    return model, positions
```

#### 4. **Forward Pass** (`feature_extraction.py::extract_features()`)
```python
with torch.no_grad():
    for batch_eeg in batches:
        # Expand positions for batch
        batch_size = batch_eeg.shape[0]
        pos = self.positions.unsqueeze(0).expand(batch_size, -1, -1)
        # pos shape now: (batch_size, 96, 3)
        
        # REVE forward pass with 4D positional encoding
        output = self.model(batch_eeg, pos)  # ← Both EEG and coordinates
```


## Usage

### Full Pipeline (All Subjects)

```bash
python main.py
```

**Processes:**
- sub-PD089 and sub-PD094
- Uses actual 3D electrode coordinates
- Saves epochs, labels, and REVE features


### Configuration Options

Edit `config.py`:

```python
# Downsampling (optional)
DOWNSAMPLE_RATE = None  # No downsampling
# DOWNSAMPLE_RATE = 250  # Downsample to 250 Hz

# Filtering
BANDPASS_LOW = 0.5      # Hz
BANDPASS_HIGH = 99.5    # Hz

# Epoch extraction
TMIN = -1.5             # Seconds before marker
TMAX = 1.5              # Seconds after marker

# Normalization
Z_SCORE_NORMALIZE = True
```

### Process Multiple Subjects

```python
from eeg_processing.main import process_multiple_subjects

# Process all subjects efficiently
all_results = process_multiple_subjects(
    subject_ids=['sub-PD089', 'sub-PD094'],
    session_id='ses-S001',
    extract_features=True,
    save_data=True
)
```

### Use REVE Feature Extractor Directly

```python
from feature_extraction import REVEFeatureExtractor

# Create extractor (automatically loads mapped positions from electrodes_pos/)
channel_labels = [str(i+1) for i in range(96)]
extractor = REVEFeatureExtractor(channel_labels=channel_labels)

# Load REVE model with your number of classes
extractor.load_model(num_classes=4)  # e.g., 4 workload classes

# Extract features (single efficient pass through REVE)
features = extractor.extract_features(epochs, batch_size=4)
# Output shape: (n_epochs, 393216)

# Organize by label (fast - just array indexing)
label_features = extractor.extract_features_for_labels(features, labels)
# Output: {'medium': (197, 393216), 'high': (191, 393216), ...}
```

### Configuration

Edit [config.py](config.py) to customize:

- `TMIN` / `TMAX` - Epoch time window (default: -1.5 to +1.5 sec)
- `SAMPLING_RATE` - EEG sampling rate
- `BANDPASS_LOW` / `BANDPASS_HIGH` - Filter frequencies
- `TARGET_MARKER_PREFIX` - Which markers to extract (`'workload'` by default)
- `SKIP_MARKERS` - Markers to exclude
- `OUTPUT_DIR` - Where to save results

## Output Files

When `save_data=True`, 4 files are created per subject in `output/`:

| File | Shape | Content | Purpose |
|------|-------|---------|---------|
| `{subject}_{session}_epochs.npy` | (n_epochs, 96, seq_len) | Preprocessed raw EEG | Classical ML, visualization |
| `{subject}_{session}_labels.npy` | (n_epochs,) | Workload labels | Training target |
| `{subject}_{session}_features.npy` | (n_epochs, feature_dim) | REVE features | Deep learning classification |
| `{subject}_{session}_label_features.npy` | dict | Features by label | Analysis by workload class |

**Note:** `seq_len` = 1501 (500 Hz) or 750-751 (250 Hz if downsampled)

### Coordinate System

The CapTrak positions are in a standard head-centered coordinate system:

```
      Front (positive Y)
            ↑
    Z  (--)←●→(+) X
    |       Left  Right
    ↓
   Down (negative Y, back)
```

- **Origin**: Center of head
- **X**: Left (-) to Right (+), ~80mm range each side
- **Y**: Back (-) to Front (+), ~110mm range each direction  
- **Z**: Bottom (-) to Top (+), ~190mm total height


## References

- **REVE Paper**: [Link to paper]
  - 4D Positional Encoding section
  - Spatial Augmentation approach
- **Brain-bzh REVE**: https://huggingface.co/brain-bzh/reve-base
- **MNE-Python**: https://mne.tools/
- **PyTorch**: https://pytorch.org/
| `{subject}_{session}_labels.npy` | (n_epochs,) | Workload labels | Ground truth for training |
| `{subject}_{session}_features.npy` | (n_epochs, 393216) | REVE features | Deep learning, classification |
| `{subject}_{session}_label_features.npy` | dict | Features by label | Per-class analysis |


## Feature Extraction Pipeline Details

### How REVE Features Are Computed

1. **Load mapped positions** → Automatically loads 96 REVE position names from `electrode_mapping_to_standard.json`
2. **Get embeddings** → Retrieves 3D coordinates and position embeddings from REVE's pre-trained model (from `reve_all_positions.json`)
3. **Process epochs** → Feeds EEG data and position embeddings to REVE transformer
4. **Extract features** → Flattens REVE transformer output to 393,216-dimensional feature vectors
5. **Organize by label** → Groups features by workload class for convenient analysis

## Training & Classification

### Comprehensive Evaluation (`train_comprehensive.py`)

Trains a neural network classifier to predict workload levels from REVE features across **5 different scenarios**:

```bash
python train_comprehensive.py
```

#### Architecture

- **Input**: 393,216-dimensional REVE features
- **Hidden layer**: 256 neurons with ReLU activation
- **Output**: 4 workload classes (original, low, medium, high)
- **Regularization**: Dropout (20%)
- **Training**: 20 epochs, Adam optimizer

#### 5 Evaluation Scenarios

| Scenario | Train Data | Test Data | Purpose |
|----------|-----------|-----------|---------|
| **Combined Split** | Both subjects mixed (80%) | Both subjects mixed (20%) | Overall generalization |
| **Subject PD089 Only** | PD089 (80%) | PD089 (20%) | Within-subject performance |
| **Subject PD094 Only** | PD094 (80%) | PD094 (20%) | Within-subject performance |
| **Cross 089→094** | PD089 (100%) | PD094 (100%) | Cross-subject generalization |
| **Cross 094→089** | PD094 (100%) | PD089 (100%) | Cross-subject generalization |

#### Configuration

Set in `config.py`:
```python
EVAL_LEARNING_RATES = [0.001, 0.002, 0.005]        # Learning rates to test
EVAL_SEEDS = [42, 123, 456]          # Random seeds for reproducibility
EVAL_NUM_EPOCHS = 20                 # Epochs per training run
```

**Total runs**: 5 scenarios × 3 LR × 3 seeds = **45 models trained**

#### Results Output

Saves `output/comprehensive_results.json` with:
- Accuracy, balanced accuracy, Cohen's kappa, F1 score
- Confusion matrices and per-class metrics
- All results organized by scenario and hyperparameter config

### Results Analysis (`analyze_results.py`)

Prints detailed summary statistics from training results:

```bash
python analyze_results.py
```

**Output includes:**
- Per-scenario mean ± std accuracy
- Scenario ranking by performance
- Comparison: within-subject vs cross-subject
- Seed stability analysis
- Improvement over random baseline (25%)

#### Key Findings

- **Within-subject (best)**: ~37-39% accuracy (within same participant)
- **Cross-subject (worst)**: ~24-27% accuracy (train on one, test on other)
- **Combined**: ~35% accuracy (mixed subjects)
- **Random baseline**: 25% (4 equal classes)

**Interpretation**: REVE features capture **subject-specific patterns** better than generalizable workload signals. Cross-subject failure suggests individual differences dominate over task effects.

### Next Steps for Improvement

1. **Fine-tune REVE backbone**: Currently frozen; unfreezing layers may unlock subject-generalizable features
2. **Add classical features**: PSD, spectral entropy, band power ratios may complement REVE
3. **Subject-specific adaptation**: Shared backbone + per-subject layers for balanced generalization
4. **Expand dataset**: More subjects (5-10+) crucial for learning portable workload patterns
