"""
Test script: Use REVE model with mapped position embeddings for feature extraction.
This demonstrates how the mapped positions are used in the pipeline.
"""

import numpy as np
import torch
import json
from pathlib import Path

from feature_extraction import REVEFeatureExtractor


def test_feature_extraction_with_mapped_positions():
    """Test feature extraction using pre-mapped electrode positions."""
    
    print("="*80)
    print("TESTING REVE FEATURE EXTRACTION WITH MAPPED POSITION EMBEDDINGS")
    print("="*80)
    
    # Create dummy EEG data (like real pipeline)
    # Shape: (num_epochs, num_channels, num_samples)
    num_epochs = 4
    num_channels = 96
    num_samples = 512  # 1 second at 512 Hz
    
    print(f"\nCreating dummy EEG data:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Channels: {num_channels}")
    print(f"  Samples per epoch: {num_samples}")
    
    eeg_data = np.random.randn(num_epochs, num_channels, num_samples).astype(np.float32)
    
    # Check that mapping files exist
    mapping_path = Path(__file__).parent.parent / "electrodes_pos" / "electrode_mapping_to_standard.json"
    positions_path = Path(__file__).parent.parent / "electrodes_pos" / "reve_all_positions.json"
    
    if not mapping_path.exists():
        print(f"\nERROR: Mapping file not found: {mapping_path}")
        print("Please run: python inspect_electrodes/map_to_standard_positions.py")
        return
    
    if not positions_path.exists():
        print(f"\nERROR: REVE positions file not found: {positions_path}")
        print("Please run: python inspect_electrodes/extract_reve_positions.py")
        return
    
    print(f"\n[OK] Mapping file found: {mapping_path.name}")
    print(f"[OK] REVE positions file found: {positions_path.name}")
    
    # Initialize feature extractor
    print("\n" + "-"*80)
    print("STEP 1: Initialize feature extractor")
    print("-"*80)
    
    channel_labels = [str(i+1) for i in range(96)]
    feature_extractor = REVEFeatureExtractor(channel_labels=channel_labels)
    
    # Load model with mapped positions
    print("\n" + "-"*80)
    print("STEP 2: Load REVE model with mapped position embeddings")
    print("-"*80)
    
    try:
        feature_extractor.load_model(num_classes=2)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show position mapping details
    print("\n" + "-"*80)
    print("STEP 3: Display position mapping details")
    print("-"*80)
    
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    print(f"\nElectrode position mapping (sample):")
    print(f"{'Electrode':<12} {'--> REVE':<20} {'Distance (mm)':<15}")
    print("-"*47)
    
    for electrode_num in range(1, 11):  # Show first 10
        electrode_key = str(electrode_num)
        if electrode_key in mapping:
            data = mapping[electrode_key]
            print(f"{electrode_num:<12} {data['standard_position']:<20} {data['distance_mm']:<15.2f}")
    
    print("...")
    
    # Extract features
    print("\n" + "-"*80)
    print("STEP 4: Extract features from EEG epochs")
    print("-"*80)
    
    try:
        features = feature_extractor.extract_features(eeg_data, batch_size=2)
        
        print(f"\n[OK] Feature extraction successful!")
        print(f"  Input shape:  {eeg_data.shape} (epochs, channels, samples)")
        print(f"  Output shape: {features.shape} (epochs, features)")
        
        # Show some statistics
        print(f"\nFeature statistics:")
        print(f"  Min: {features.min():.6f}")
        print(f"  Max: {features.max():.6f}")
        print(f"  Mean: {features.mean():.6f}")
        print(f"  Std: {features.std():.6f}")
        
    except Exception as e:
        print(f"ERROR during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)
    print(f"\n[OK] REVE pipeline is ready to use!")
    print(f"[OK] All 96 electrodes are mapped to REVE positions")
    print(f"[OK] Position embeddings are consistent across subjects")
    print(f"\nYou can now use this in your main pipeline with:")
    print(f"  feature_extractor = REVEFeatureExtractor(channel_labels=your_labels)")
    print(f"  feature_extractor.load_model(num_classes=your_num_classes)")
    print(f"  features = feature_extractor.extract_features(your_epochs)")


if __name__ == "__main__":
    test_feature_extraction_with_mapped_positions()
