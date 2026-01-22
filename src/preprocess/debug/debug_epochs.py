"""
Debug script to analyze extracted epochs and their associated events.

This script helps verify that:
1. Epochs are correctly extracted around marker events
2. Epochs are associated with the correct labels (task/no-task)
3. Downsampling is applied correctly before epoch extraction
4. Event timestamps are properly aligned with epoch indices
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys
from datetime import datetime

# Add src to path (go up 3 levels from src/preprocess/debug/ to workspace root)
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root / "src"))

from preprocess.mne_preprocessor import MNEPreprocessor, MNEPreprocessorZLDataset
from preprocess.preprocess_config import (
    EPOCH_TMIN, EPOCH_TMAX, DOWNSAMPLE_RATE,
    MNE_BANDPASS_LOW, MNE_BANDPASS_HIGH
)
from data_loader.zl_dataset import ZLDataset


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*80}")
    print(f"  {title}")
    print(f"{'-'*80}")


def save_epochs_to_csv(results: Dict[str, Any], csv_file: Path):
    """
    Save extracted epochs metadata to CSV for manual comparison with original markers.
    
    Args:
        results: Results dictionary from analyze_epochs_with_events
        csv_file: Path to save CSV file
    """
    if not results or 'metadata' not in results:
        print("No metadata to save")
        return
    
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['EpochIndex', 'Label', 'LabelStr', 'SampleIndex', 'TimeInSeconds', 'NotesForComparison']
            )
            writer.writeheader()
            
            sampling_rate = results.get('sampling_rate_after_preproc', 200)
            
            for epoch_idx, meta in enumerate(results['metadata']):
                sample_idx = meta.get('sample_index', 0)
                label = meta.get('label', -1)
                label_str = meta.get('label_str', 'unknown')
                
                # Convert sample index to time in seconds (relative to recording start after preprocessing)
                time_seconds = sample_idx / sampling_rate
                
                writer.writerow({
                    'EpochIndex': epoch_idx,
                    'Label': label,
                    'LabelStr': label_str,
                    'SampleIndex': sample_idx,
                    'TimeInSeconds': f'{time_seconds:.4f}',
                    'NotesForComparison': f'Epoch {epoch_idx}: {label_str}'
                })
        
        print(f"Extracted epochs CSV saved to: {csv_file}")
    except Exception as e:
        print(f"ERROR saving CSV: {e}")



def analyze_epochs_with_events(dataset: ZLDataset, subject_id: str, session_id: str) -> Dict[str, Any]:
    """
    Load and analyze preprocessed epochs for a subject.
    
    Args:
        dataset: ZLDataset instance
        subject_id: Subject identifier (e.g., 'sub-PD089')
        session_id: Session identifier (e.g., 'ses-S001')
        
    Returns:
        Dictionary with analysis results
    """
    print_section(f"Analyzing Epochs for {subject_id} / {session_id}")
    
    # Load raw data
    data_dict = dataset.load_subject_data(subject_id, session_id)
    eeg_data = data_dict['eeg']
    markers = data_dict['markers']
    eeg_timestamps = data_dict['eeg_timestamps']
    marker_timestamps = data_dict['marker_timestamps']
    sampling_rate = data_dict['sampling_rate']
    channel_labels = data_dict['channel_labels']
    
    file_path = dataset.get_eeg_file_path(subject_id, session_id)
    print(f"\nInput file: {file_path}")
    print(f"File size: {file_path.stat().st_size / (1024**2):.2f} MB")
    
    print_subsection("Raw Data")
    print(f"EEG shape: {eeg_data.shape}")
    print(f"Channels: {len(channel_labels)}")
    print(f"Original sampling rate: {sampling_rate} Hz")
    print(f"Total markers: {len(markers)}")
    
    # Initialize preprocessor with loaded data
    preprocessor = MNEPreprocessorZLDataset(
        eeg_data=eeg_data,
        eeg_timestamps=eeg_timestamps,
        markers=markers,
        marker_timestamps=marker_timestamps,
        channel_labels=channel_labels,
        sampling_rate=sampling_rate
    )
    
    print_subsection("Preprocessing Configuration")
    print(f"Bandpass filter: {MNE_BANDPASS_LOW} - {MNE_BANDPASS_HIGH} Hz")
    print(f"Original sampling rate: {preprocessor.sampling_rate} Hz")
    print(f"Downsample to: {DOWNSAMPLE_RATE if DOWNSAMPLE_RATE else 'No downsampling'} Hz")
    print(f"Epoch window: [{EPOCH_TMIN}, {EPOCH_TMAX}] seconds")
    
    # Preprocess
    print_subsection("Preprocessing Steps")
    preprocessed_data = preprocessor.preprocess()
    print(f"Preprocessed data shape: {preprocessed_data.shape}")
    print(f"  Channels: {preprocessed_data.shape[0]}")
    print(f"  Samples: {preprocessed_data.shape[1]}")
    print(f"  Duration: {preprocessed_data.shape[1] / preprocessor.sampling_rate:.2f} seconds")
    
    # Extract epochs
    print_subsection("Epoch Extraction")
    epochs_data, labels, metadata = preprocessor.extract_epochs(
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX
    )
    
    print(f"\nExtracted epochs:")
    print(f"  Total epochs: {len(labels)}")
    print(f"  Epochs shape: {epochs_data.shape}")
    if epochs_data.size > 0:
        print(f"    Channels per epoch: {epochs_data.shape[1]}")
        print(f"    Samples per epoch: {epochs_data.shape[2]}")
        print(f"    Samples per epoch corresponds to: {epochs_data.shape[2] / preprocessor.sampling_rate:.3f} seconds")
    
    # Summarize label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        label_str = "task" if label == 1 else "no task"
        pct = (count / len(labels)) * 100
        print(f"  {label_str}: {count} ({pct:.1f}%)")
    
    # Print detailed epoch information
    print_subsection("Detailed Epoch Information")
    
    if len(metadata) == 0:
        print("ERROR: No epoch metadata found!")
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "n_epochs": 0,
            "epochs_data": epochs_data,
            "labels": labels,
            "metadata": metadata,
        }
    
    print(f"\n{'Idx':<5} {'Event':<12} {'Label':<12} {'Time (s)':<12} {'Marker TS':<15} {'Samples/Epoch':<15}")
    print("-" * 80)
    
    for epoch_idx, (label, meta) in enumerate(zip(labels, metadata)):
        label_str = "task" if label == 1 else "no task"
        marker_ts = meta.get('marker_timestamp', None)
        sample_idx = meta.get('sample_index', None)
        
        # Format marker timestamp
        if marker_ts is not None:
            marker_ts_str = f"{marker_ts:.4f}"
        else:
            marker_ts_str = "N/A"
        
        # Format sample index
        if sample_idx is not None:
            sample_idx_str = f"{sample_idx}"
        else:
            sample_idx_str = "N/A"
        
        n_samples = epochs_data.shape[2] if epochs_data.size > 0 else 0
        
        print(f"{epoch_idx:<5} {label_str:<12} {label:<12} {marker_ts_str:<12} {marker_ts_str:<15} {n_samples:<15}")
    
    # Print event alignment information
    print_subsection("Event Alignment Check")
    
    print(f"\nOriginal markers in file: {len(preprocessor.marker_timestamps)}")
    print(f"Valid events after filtering: {len(labels)}")
    print(f"\nEvent timestamps (relative to recording start):")
    
    if len(metadata) > 0:
        print(f"\n{'Epoch':<6} {'Marker Time (s)':<20} {'Event ID':<12} {'Label':<12}")
        print("-" * 50)
        
        for epoch_idx, (label, meta) in enumerate(zip(labels, metadata)):
            marker_ts = meta.get('marker_timestamp', None)
            event_id = meta.get('sample_index', None)
            label_str = "task" if label == 1 else "no task"
            
            if marker_ts is not None:
                ts_str = f"{marker_ts:.4f}"
            else:
                ts_str = "N/A"
            
            if event_id is not None:
                id_str = f"{event_id}"
            else:
                id_str = "N/A"
            
            print(f"{epoch_idx:<6} {ts_str:<20} {id_str:<12} {label_str:<12}")
    
    # Check for gaps and overlaps in events
    print_subsection("Event Timeline Analysis")
    
    if len(metadata) > 1:
        # Get sample indices from metadata
        sample_indices = [m.get('sample_index', 0) for m in metadata]
        
        # Convert sample indices to time (in seconds) using current sampling rate
        marker_times = [idx / preprocessor.sampling_rate for idx in sample_indices]
        time_diffs = np.diff(marker_times)
        
        print(f"\nTime differences between consecutive events:")
        print(f"  Min: {np.min(time_diffs):.4f} seconds")
        print(f"  Max: {np.max(time_diffs):.4f} seconds")
        print(f"  Mean: {np.mean(time_diffs):.4f} seconds")
        print(f"  Median: {np.median(time_diffs):.4f} seconds")
        
        # Check for potential overlaps with epoch window
        epoch_window = abs(EPOCH_TMIN) + abs(EPOCH_TMAX)
        potential_overlaps = np.sum(time_diffs < epoch_window)
        if potential_overlaps > 0:
            print(f"\n⚠️  WARNING: {potential_overlaps} consecutive events have time diff < epoch window ({epoch_window}s)")
            print("   These epochs may have overlapping time windows (MNE may merge/drop them)")
    
    # Create results dictionary
    results = {
        "subject_id": subject_id,
        "session_id": session_id,
        "input_file": str(file_path),
        "preprocessing_config": {
            "bandpass_low": MNE_BANDPASS_LOW,
            "bandpass_high": MNE_BANDPASS_HIGH,
            "downsample_rate": DOWNSAMPLE_RATE,
            "epoch_tmin": EPOCH_TMIN,
            "epoch_tmax": EPOCH_TMAX,
        },
        "sampling_rate_before_preproc": preprocessor.sampling_rate,
        "sampling_rate_after_preproc": preprocessor.sampling_rate,
        "n_channels": epochs_data.shape[1] if epochs_data.size > 0 else 0,
        "n_epochs": len(labels),
        "n_samples_per_epoch": epochs_data.shape[2] if epochs_data.size > 0 else 0,
        "label_distribution": dict(zip(
            ["no_task" if l == 0 else "task" for l in unique_labels],
            counts.tolist()
        )),
        "metadata": metadata,
    }
    
    return results


def main():
    """Main debug routine."""
    print("\n" + "="*80)
    print("EPOCH DEBUG ANALYSIS")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Initialize dataset
    workspace_root = Path(__file__).parent.parent.parent.parent
    dataset = ZLDataset(workspace_root / "data/Zander Labs")
    subjects = dataset.subjects
    
    print(f"\nFound {len(subjects)} available subjects: {subjects}")
    
    if len(subjects) == 0:
        print("ERROR: No subjects found!")
        return
    
    # Analyze first subject (or modify to analyze specific subject)
    subject_id = subjects[0]
    sessions = dataset.sessions.get(subject_id, [])
    
    if len(sessions) == 0:
        print(f"ERROR: No sessions found for {subject_id}")
        return
    
    session_id = sessions[0]
    
    print(f"\nAnalyzing: {subject_id} / {session_id}")
    
    # Run analysis
    results = analyze_epochs_with_events(dataset, subject_id, session_id)
    
    # Save results to JSON
    if results:
        output_dir = Path(__file__).parent / "debug_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = output_dir / f"{subject_id}_{session_id}_epoch_analysis.json"
        results_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in results.items()
        }
        with open(json_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save extracted epochs to CSV for manual comparison
        csv_file = output_dir / f"{subject_id}_{session_id}_extracted_epochs.csv"
        save_epochs_to_csv(results, csv_file)
        
        print_section("Results Saved")
        print(f"JSON: {json_file}")
        print(f"CSV:  {csv_file}")
    
    print_section("Analysis Complete")


if __name__ == "__main__":
    main()
