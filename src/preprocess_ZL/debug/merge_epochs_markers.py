#!/usr/bin/env python3
"""
Merge extracted epochs with their corresponding original markers for easy manual comparison.

This script:
1. Reads the extracted epochs CSV
2. Reads the original markers CSV
3. For each extracted epoch, finds all original markers that fall within its time window
4. Creates a detailed comparison CSV showing both together
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

def read_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Read CSV file into list of dicts."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_epochs_with_markers(subject_id: str = 'sub-PD089', session_id: str = 'ses-S001'):
    """
    Merge extracted epochs with their corresponding original markers.
    
    For each extracted epoch, find all markers that fall within the epoch's time window
    and create a CSV that shows them side-by-side.
    """
    debug_output = Path(__file__).parent / "debug_output"
    
    # Read both files
    original_markers_file = debug_output / f"{subject_id}_{session_id}_markers.csv"
    extracted_epochs_file = debug_output / f"{subject_id}_{session_id}_extracted_epochs.csv"
    output_file = debug_output / f"{subject_id}_{session_id}_epochs_with_markers.csv"
    
    if not original_markers_file.exists():
        print(f"ERROR: Original markers file not found: {original_markers_file}")
        return
    
    if not extracted_epochs_file.exists():
        print(f"ERROR: Extracted epochs file not found: {extracted_epochs_file}")
        return
    
    # Read data
    original_markers = read_csv(original_markers_file)
    extracted_epochs = read_csv(extracted_epochs_file)
    
    # Get recording start time (from first marker)
    if not original_markers:
        print("ERROR: No markers found in original markers file")
        return
    
    recording_start_time = float(original_markers[0]['Timestamp'])
    print(f"Recording start time: {recording_start_time}")
    print(f"Total original markers: {len(original_markers)}")
    print(f"Total extracted epochs: {len(extracted_epochs)}\n")
    
    # Epoch window
    EPOCH_TMIN = -1.5
    EPOCH_TMAX = 1.5
    
    # Create mapping of marker timestamps to marker objects
    marker_by_time = {}
    for marker in original_markers:
        ts = float(marker['Timestamp'])
        marker_by_time[ts] = marker
    
    sorted_marker_times = sorted(marker_by_time.keys())
    
    # Create output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as outf:
        fieldnames = [
            'EpochIndex',
            'EpochLabel',
            'EpochTime_Seconds',
            'EpochStartTime',
            'EpochEndTime',
            'MarkersInEpoch',
            'OriginalMarker',
            'MarkerTimestamp',
            'MarkerRelativeTime',
            'MatchType'
        ]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        
        # For each extracted epoch, find its markers
        for epoch in extracted_epochs:
            epoch_idx = int(epoch['EpochIndex'])
            epoch_label = epoch['LabelStr']
            epoch_time = float(epoch['TimeInSeconds'])
            
            # Epoch spans [epoch_time + EPOCH_TMIN, epoch_time + EPOCH_TMAX]
            epoch_start = epoch_time + EPOCH_TMIN
            epoch_end = epoch_time + EPOCH_TMAX
            
            # Convert to absolute recording time
            abs_epoch_start = recording_start_time + epoch_start
            abs_epoch_end = recording_start_time + epoch_end
            
            # Find markers in this epoch's window
            markers_in_window = []
            for marker_time in sorted_marker_times:
                if abs_epoch_start <= marker_time <= abs_epoch_end:
                    markers_in_window.append((marker_time, marker_by_time[marker_time]))
            
            if markers_in_window:
                # Write one row per marker found in this epoch
                for marker_time, marker in markers_in_window:
                    marker_relative_time = marker_time - recording_start_time
                    writer.writerow({
                        'EpochIndex': epoch_idx,
                        'EpochLabel': epoch_label,
                        'EpochTime_Seconds': f'{epoch_time:.4f}',
                        'EpochStartTime': f'{epoch_start:.4f}',
                        'EpochEndTime': f'{epoch_end:.4f}',
                        'MarkersInEpoch': len(markers_in_window),
                        'OriginalMarker': marker['Original_Marker'],
                        'MarkerTimestamp': marker_time,
                        'MarkerRelativeTime': f'{marker_relative_time:.6f}',
                        'MatchType': 'CENTER' if abs(marker_time - (recording_start_time + epoch_time)) < 0.1 else 'IN_WINDOW'
                    })
            else:
                # No markers found in epoch - write a warning row
                writer.writerow({
                    'EpochIndex': epoch_idx,
                    'EpochLabel': epoch_label,
                    'EpochTime_Seconds': f'{epoch_time:.4f}',
                    'EpochStartTime': f'{epoch_start:.4f}',
                    'EpochEndTime': f'{epoch_end:.4f}',
                    'MarkersInEpoch': 0,
                    'OriginalMarker': '*** NO MARKERS IN WINDOW ***',
                    'MarkerTimestamp': '',
                    'MarkerRelativeTime': '',
                    'MatchType': 'WARNING'
                })
    
    print(f"✓ Created detailed comparison CSV: {output_file}\n")
    
    # Print summary
    print("="*120)
    print("SUMMARY: Extracted Epochs with Original Markers")
    print("="*120)
    
    # Read the newly created file to show summary
    comparison_data = read_csv(output_file)
    
    print(f"\n{'Epoch':<6} {'Label':<10} {'Epoch Time':<12} {'Markers Found':<15} {'First Marker':<50}")
    print("-"*120)
    
    current_epoch = None
    for row in comparison_data:
        epoch_idx = row['EpochIndex']
        
        # Only print once per epoch (first marker row)
        if epoch_idx != current_epoch:
            current_epoch = epoch_idx
            label = row['EpochLabel']
            epoch_time = row['EpochTime_Seconds']
            n_markers = row['MarkersInEpoch']
            marker = row['OriginalMarker'][:50] if row['OriginalMarker'] else '(no marker)'
            
            print(f"{epoch_idx:<6} {label:<10} {epoch_time:<12} {n_markers:<15} {marker:<50}")
    
    print(f"\nFull comparison saved to: {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        subject_id = sys.argv[1]
        session_id = sys.argv[2] if len(sys.argv) > 2 else 'ses-S001'
        merge_epochs_with_markers(subject_id, session_id)
    else:
        # Use defaults
        merge_epochs_with_markers()


if __name__ == "__main__":
    main()
