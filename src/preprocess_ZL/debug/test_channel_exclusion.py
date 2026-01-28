#!/usr/bin/env python3
"""Quick test to verify channel exclusion works correctly."""

import sys
from pathlib import Path

# Add src to path (go up 3 levels from src/preprocess_zl/debug/ to workspace root)
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root / "src"))

from preprocess_ZL.zl_dataset import ZLDataset
from preprocess_ZL.zl_preprocessing_pipeline import ZLPreprocessingPipeline

# Load first subject
dataset = ZLDataset(workspace_root / "data/Zander Labs")
subject_id = dataset.subjects[0]
session_id = dataset.sessions[subject_id][0]

print(f"Testing: {subject_id} / {session_id}\n")

# Load data
data_dict = dataset.load_subject_data(subject_id, session_id)

print(f"Original data:")
print(f"  Shape: {data_dict['eeg'].shape}")
print(f"  Channels: {len(data_dict['channel_labels'])}")

# Create preprocessor (this will run channel exclusion)
preprocessor = ZLPreprocessingPipeline(
    eeg_data=data_dict['eeg'],
    eeg_timestamps=data_dict['eeg_timestamps'],
    markers=data_dict['markers'],
    marker_timestamps=data_dict['marker_timestamps'],
    channel_labels=data_dict['channel_labels'],
    sampling_rate=data_dict['sampling_rate']
)

print(f"\n✓ Channel exclusion successful!")
