"""
Explore THU-EP dataset structure and contents.

This script loads and inspects:
- Subject EEG data (sub_1.mat, etc.)
- Labels (label.mat)
- Ratings (ratings.mat)

Usage:
    python -m src.thu_ep.exploration.explore_data
    uv run python -m src.thu_ep.exploration.explore_data
"""

from pathlib import Path

import numpy as np
import h5py
import scipy.io as sio

from ..config import get_config


# Load configuration
_cfg = get_config()

# Data paths (from config)
DATA_DIR = _cfg.raw_data_dir.parent
EEG_DIR = _cfg.raw_data_dir
RATINGS_FILE = _cfg.ratings_dir / "ratings.mat"
LABELS_FILE = _cfg.others_dir / "label.mat"


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def load_mat_file(filepath: Path) -> dict:
    """
    Load a .mat file.
    
    Returns a dictionary with the file contents.
    """
    try:
        # Try scipy.io first (for older .mat files)
        return sio.loadmat(str(filepath))
    except NotImplementedError:
        # Fall back to h5py for v7.3 files
        data = {}
        with h5py.File(str(filepath), 'r') as f:
            for key in f.keys():
                if key.startswith('#'):  # Skip HDF5 metadata
                    continue
                data[key] = np.array(f[key])
        return data


def explore_h5_contents(filepath: Path):
    """Explore HDF5 file structure in detail."""
    print(f"\n  HDF5 Structure:")
    with h5py.File(str(filepath), 'r') as f:
        def print_h5_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"    {name}/ (group)")
        f.visititems(print_h5_item)


def read_h5_string_array(filepath: Path, dataset_name: str) -> list[str]:
    """
    Read an array of strings from an HDF5 file with object references.
    
    MATLAB v7.3 files store cell arrays of strings as object references.
    This function dereferences them to get the actual strings.
    """
    strings = []
    with h5py.File(str(filepath), 'r') as f:
        dataset = f[dataset_name]
        # Iterate over references
        for ref in dataset[0]:  # dataset is (1, N), take first row
            # Dereference to get the actual data
            dereferenced = f[ref]
            # Convert uint16 array to string
            chars = np.array(dereferenced).flatten()
            string = ''.join(chr(c) for c in chars)
            strings.append(string)
    return strings


def explore_mat_contents(mat_data: dict, prefix: str = ""):
    """Recursively explore and print .mat file contents."""
    for key, value in mat_data.items():
        if key.startswith("__") or key.startswith("#"):  # Skip metadata keys
            continue
        
        full_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, np.ndarray):
            print(f"  {full_key}:")
            print(f"    Type: {type(value).__name__}")
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
            
            # Show sample values
            if value.size > 0:
                if value.dtype.kind in ['i', 'u', 'f']:  # numeric
                    print(f"    Range: [{value.min():.4f}, {value.max():.4f}]")
                    print(f"    Mean: {value.mean():.4f}")
                if value.size <= 20:
                    print(f"    Values: {value.flatten()}")
                elif value.ndim == 1:
                    print(f"    First 5: {value[:5]}")
                    print(f"    Last 5: {value[-5:]}")
            print()
        else:
            print(f"  {full_key}: {type(value).__name__}")
            if hasattr(value, '__len__') and len(value) < 10:
                print(f"    Value: {value}")
            print()


def explore_labels():
    """Load and explore label.mat (contains EEG channel names)."""
    print_separator("EEG CHANNEL LABELS (label.mat)")
    print(f"Path: {LABELS_FILE}")
    
    if not LABELS_FILE.exists():
        print("ERROR: File not found!")
        return None
    
    # Show HDF5 structure
    explore_h5_contents(LABELS_FILE)
    
    # Read channel names from object references
    print("\n  EEG Channel Names (32 channels in order):")
    try:
        channel_names = read_h5_string_array(LABELS_FILE, 'label')
        for i, name in enumerate(channel_names, 1):
            print(f"    {i:2d}. {name}")
        print(f"\n  Total channels: {len(channel_names)}")
    except Exception as e:
        print(f"  ERROR reading channel names: {e}")
        channel_names = None
    
    return channel_names


def explore_ratings():
    """Load and explore ratings.mat."""
    print_separator("RATINGS (ratings.mat)")
    print(f"Path: {RATINGS_FILE}")
    
    if not RATINGS_FILE.exists():
        print("ERROR: File not found!")
        return None
    
    # Show HDF5 structure
    explore_h5_contents(RATINGS_FILE)
    
    mat = load_mat_file(RATINGS_FILE)
    ratings = mat.get('ratings')
    
    if ratings is not None:
        print(f"\n  Ratings array shape: {ratings.shape}")
        print(f"    - Dimension 0: {ratings.shape[0]} rating dimensions (items 1-12)")
        print(f"    - Dimension 1: {ratings.shape[1]} stimuli/videos")
        print(f"    - Dimension 2: {ratings.shape[2]} subjects")
        print(f"\n  Rating scale: {ratings.min():.0f} to {ratings.max():.0f}")
        print(f"  Mean rating: {ratings.mean():.2f}")
        
        # Show detailed ratings per stimulus (all 12 items, averaged across 80 subjects)
        print(f"\n  28 Stimuli - Average ratings for each of 12 items (across 80 subjects):")
        print("  Note: Stimulus labels/names should be in Readme.pdf documentation")
        print()
        
        # Header
        header = "  Stim  |" + "".join([f" Item{i+1:2d}" for i in range(12)]) + " | Overall"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        for stim_idx in range(ratings.shape[1]):
            # Average across subjects (dimension 2) for each item
            # ratings shape: (12, 28, 80) -> for stim_idx: (12, 80) -> mean over subjects -> (12,)
            item_means = ratings[:, stim_idx, :].mean(axis=1)  # Mean across 80 subjects
            overall_mean = item_means.mean()
            
            row = f"  {stim_idx+1:4d}  |" + "".join([f"  {item_means[i]:5.2f}" for i in range(12)]) + f" |  {overall_mean:5.2f}"
            print(row)
    
    return mat


def explore_subject_eeg(subject_id: int = 1):
    """Load and explore a subject's EEG data."""
    print_separator(f"SUBJECT {subject_id} EEG DATA (sub_{subject_id}.mat)")
    
    subject_file = EEG_DIR / f"sub_{subject_id}.mat"
    print(f"Path: {subject_file}")
    
    if not subject_file.exists():
        print("ERROR: File not found!")
        return None
    
    # Show HDF5 structure
    explore_h5_contents(subject_file)
    
    mat = load_mat_file(subject_file)
    print(f"\nKeys in file: {[k for k in mat.keys() if not k.startswith('__') and not k.startswith('#')]}")
    
    explore_mat_contents(mat)
    
    # Additional EEG-specific exploration
    for key in mat.keys():
        if key.startswith("__") or key.startswith("#"):
            continue
        value = mat[key]
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            print(f"\n  {key} detailed info:")
            print(f"    Memory size: {value.nbytes / 1024 / 1024:.2f} MB")
            if value.ndim == 2:
                print(f"    Likely interpretation: {value.shape[0]} channels x {value.shape[1]} samples")
                print(f"    Or: {value.shape[0]} trials x {value.shape[1]} features")
            elif value.ndim == 3:
                print(f"    Likely interpretation: {value.shape[0]} trials x {value.shape[1]} channels x {value.shape[2]} samples")
    
    return mat


def list_all_subjects():
    """List all available subject files."""
    print_separator("AVAILABLE SUBJECTS")
    print(f"Directory: {EEG_DIR}")
    
    if not EEG_DIR.exists():
        print("ERROR: Directory not found!")
        return []
    
    # Sort numerically by extracting subject number
    subjects = list(EEG_DIR.glob("sub_*.mat"))
    subjects.sort(key=lambda f: int(f.stem.split('_')[1]))
    print(f"\nTotal subjects: {len(subjects)}")
    print(f"Files: {[f.name for f in subjects[:5]]} ... {[f.name for f in subjects[-3:]]}")
    
    return subjects


def run_exploration():
    """Run full dataset exploration."""
    print_separator("THU-EP DATASET EXPLORATION")
    print(f"Data directory: {DATA_DIR.absolute()}")
    
    # List subjects
    subjects = list_all_subjects()
    
    # Explore labels
    labels = explore_labels()
    
    # Explore ratings
    ratings = explore_ratings()
    
    # Explore first subject
    subject_data = explore_subject_eeg(subject_id=1)
    
    # Summary
    print_separator("SUMMARY")
    print(f"Total subjects: {len(subjects)}")
    print(f"Labels loaded: {'Yes' if labels else 'No'}")
    print(f"Ratings loaded: {'Yes' if ratings else 'No'}")
    print(f"Sample EEG loaded: {'Yes' if subject_data else 'No'}")
    
    return {
        "subjects": subjects,
        "labels": labels,
        "ratings": ratings,
        "sample_eeg": subject_data,
    }


if __name__ == "__main__":
    run_exploration()
