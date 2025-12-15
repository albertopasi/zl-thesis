"""
Main pipeline for EEG processing: load -> preprocess -> extract features.
"""

import os
import numpy as np
import torch
from pathlib import Path

from xdf_loader import XDFLoader
from preprocessing import EEGPreprocessor
from feature_extraction import REVEFeatureExtractor
from config import OUTPUT_DIR, SAVE_PROCESSED_DATA, SAVE_FEATURES


def process_subject(subject_id, session_id, extract_features=True, save_data=True):
    """
    Process a single subject's EEG data through the full pipeline.
    
    Args:
        subject_id: Subject ID (e.g., 'sub-PD089')
        session_id: Session ID (e.g., 'ses-S001')
        extract_features: Extract REVE features
        save_data: Save processed data and features
        
    Returns:
        dict: Results containing epochs, labels, features
    """
    print(f"\n{'='*60}")
    print(f"Processing {subject_id} - {session_id}")
    print(f"{'='*60}")
    
    results = {
        'subject_id': subject_id,
        'session_id': session_id,
        'epochs': None,
        'labels': None,
        'metadata': None,
        'features': None,
        'label_features': None
    }
    
    # Step 1: Load XDF data
    print("\n[Step 1] Loading XDF file...")
    loader = XDFLoader(subject_id, session_id)
    
    try:
        eeg_data, eeg_timestamps, sfreq = loader.get_eeg_data()
        markers, marker_timestamps = loader.get_marker_data()
        channel_labels = loader.get_channel_labels()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return results
    
    # Validate loaded data
    if eeg_data.shape[0] == 0 or eeg_data.shape[1] == 0:
        print(f"ERROR: No valid EEG data loaded!")
        print(f"  EEG shape: {eeg_data.shape}")
        print(f"  Expected: (samples, channels)")
        return results
    
    if len(markers) == 0:
        print(f"ERROR: No markers found!")
        return results
    
    print(f"  EEG shape: {eeg_data.shape} (samples, channels)")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Markers: {len(markers)}")
    
    # Step 2: Preprocess and extract epochs
    print("\n[Step 2] Preprocessing with MNE and extracting epochs...")
    preprocessor = EEGPreprocessor(
        eeg_data, eeg_timestamps, markers, marker_timestamps,
        channel_labels=channel_labels, sampling_rate=sfreq
    )
    
    # Get fully processed epochs (with MNE filtering and normalization)
    epochs, epoch_labels, epoch_metadata = preprocessor.get_processed_epochs(
        preprocess=True,  # Apply MNE filtering and CAR
        normalize=True,    # Apply normalization
        tmin=-1.5,         # 1.5 seconds before marker
        tmax=1.5           # 1.5 seconds after marker
    )
    
    if len(epochs) == 0:
        print("WARNING: No epochs extracted!")
        return results
    
    results['epochs'] = epochs
    results['labels'] = epoch_labels
    results['metadata'] = epoch_metadata
    
    # Step 3: Extract REVE features
    if extract_features:
        print("\n[Step 3] Extracting REVE features with mapped position embeddings...")
        
        try:
            # Determine number of classes from unique labels
            num_classes = len(set(epoch_labels))
            print(f"Number of classes determined from data: {num_classes}")
            
            # Initialize feature extractor (uses pre-mapped positions automatically)
            feature_extractor = REVEFeatureExtractor(channel_labels=channel_labels)
            feature_extractor.load_model(num_classes=num_classes)
            
            # Extract features once
            features = feature_extractor.extract_features(epochs, batch_size=4)
            
            # Organize by label (no re-extraction)
            label_features = feature_extractor.extract_features_for_labels(
                features, epoch_labels
            )
            
            results['label_features'] = label_features
            results['features'] = features
            
        except Exception as e:
            print(f"WARNING: Feature extraction failed: {e}")
            print("Continuing with preprocessed epochs only...")
            import traceback
            traceback.print_exc()
    
    # Step 4: Save results
    if save_data:
        print("\n[Step 4] Saving results...")
        _save_results(results)
    
    return results


def process_multiple_subjects(subject_ids, session_id='ses-S001', **kwargs):
    """
    Process multiple subjects.
    
    Args:
        subject_ids: list of subject IDs
        session_id: Session ID
        **kwargs: Additional arguments for process_subject
        
    Returns:
        list: Results for each subject
    """
    all_results = []
    
    for subject_id in subject_ids:
        results = process_subject(subject_id, session_id, **kwargs)
        all_results.append(results)
    
    return all_results


def _save_results(results):
    """Save processed data to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    subject_id = results['subject_id']
    session_id = results['session_id']
    output_prefix = os.path.join(OUTPUT_DIR, f"{subject_id}_{session_id}")
    
    # Save epochs
    if results['epochs'] is not None:
        epochs_file = f"{output_prefix}_epochs.npy"
        np.save(epochs_file, results['epochs'])
        print(f"  Saved epochs: {epochs_file}")
    
    # Save labels
    if results['labels'] is not None:
        labels_file = f"{output_prefix}_labels.npy"
        np.save(labels_file, np.array(results['labels']), allow_pickle=True)
        print(f"  Saved labels: {labels_file}")
    
    # Save features
    if results['features'] is not None:
        features_file = f"{output_prefix}_features.npy"
        np.save(features_file, results['features'])
        print(f"  Saved features: {features_file}")
    
    # Save label-organized features
    if results['label_features'] is not None:
        label_features_file = f"{output_prefix}_label_features.npy"
        np.save(label_features_file, results['label_features'], allow_pickle=True)
        print(f"  Saved label features: {label_features_file}")


if __name__ == "__main__":
    # Process subjects
    subjects = ['sub-PD089', 'sub-PD094']
    
    all_results = process_multiple_subjects(
        subjects,
        extract_features=True,
        save_data=True
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    for results in all_results:
        if results['epochs'] is not None:
            print(f"{results['subject_id']}: {len(results['labels'])} epochs extracted")
