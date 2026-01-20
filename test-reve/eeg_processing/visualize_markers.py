"""
Visualize marker filtering and binary label assignment.
Shows which markers from the stream are kept and their assigned binary labels.
"""

import os
import sys
import pyxdf
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import SKIP_MARKERS, DATA_ROOT


def is_skip_marker(marker):
    """Check if marker should be skipped (non-experimental data)."""
    marker_lower = marker.lower()
    for skip_pattern in SKIP_MARKERS:
        if skip_pattern.lower() in marker_lower:
            return True
    return False


def is_valid_marker(marker):
    """Check if marker is a valid task/no-task marker (skip onset/offset)."""
    marker_lower = marker.lower()
    # Skip 'onset' and 'offset' markers
    if marker_lower.startswith('onset ') or marker_lower.startswith('offset '):
        return False
    # Only keep markers that start with 'task' or 'no task'
    if marker_lower.startswith('task') or marker_lower.startswith('no task'):
        return True
    return False


def extract_binary_label(marker):
    """
    Extract binary label from marker: task=1, no task=0
    
    Args:
        marker: Full marker string
        
    Returns:
        int: 0 for 'no task', 1 for 'task'
    """
    marker_lower = marker.lower()
    if marker_lower.startswith('no task'):
        return 0
    elif marker_lower.startswith('task'):
        return 1
    else:
        return None


def visualize_marker_filtering(subject_id):
    """
    Visualize marker filtering and label assignment for a subject.
    
    Args:
        subject_id: Subject ID (e.g., 'sub-PD089')
    """
    # Construct path to XDF file
    xdf_path = os.path.join(
        DATA_ROOT,
        subject_id,
        'ses-S001',
        'eeg',
        f'{subject_id}_ses-S001_task-demo_workload_run-001_eeg.xdf'
    )
    
    if not os.path.exists(xdf_path):
        print(f"Error: File not found: {xdf_path}")
        return
    
    print(f"\n{'='*100}")
    print(f"MARKER FILTERING VISUALIZATION: {subject_id}")
    print(f"{'='*100}\n")
    
    # Load XDF file
    print(f"Loading {xdf_path}...")
    streams, header = pyxdf.load_xdf(xdf_path)
    
    # Find marker stream
    marker_stream = None
    for stream in streams:
        if stream['info']['type'][0] == 'Markers':
            stream_name = stream['info']['name'][0]
            if 'ZLT' in stream_name:
                marker_stream = stream
                break
    
    if marker_stream is None:
        print("Error: Could not find ZLT-markers stream")
        return
    
    marker_data = marker_stream['time_series']
    marker_timestamps = marker_stream['time_stamps']
    
    # Flatten marker data
    try:
        flat_markers = [item[0] for item in marker_data]
    except:
        flat_markers = marker_data
    
    print(f"Total markers in stream: {len(flat_markers)}\n")
    
    # Categorize markers
    skip_list = []
    onset_offset_list = []
    invalid_list = []
    valid_markers_with_ts = []
    
    for i, (marker, marker_ts) in enumerate(zip(flat_markers, marker_timestamps)):
        if is_skip_marker(marker):
            skip_list.append((i, marker))
        elif marker.lower().startswith('onset ') or marker.lower().startswith('offset '):
            onset_offset_list.append((i, marker))
        elif not is_valid_marker(marker):
            invalid_list.append((i, marker))
        else:
            label = extract_binary_label(marker)
            valid_markers_with_ts.append((i, marker, marker_ts, label))
    
    # Print filtering results
    print("FILTERING SUMMARY:")
    print(f"  Skip markers (Recording/Start/End, Break, etc.): {len(skip_list)}")
    print(f"  Onset/Offset markers: {len(onset_offset_list)}")
    print(f"  Invalid markers (not task/no-task): {len(invalid_list)}")
    print(f"  Valid task/no-task markers: {len(valid_markers_with_ts)}\n")
    
    if skip_list:
        print("Skipped markers (Recording/Start/End, Break, etc.):")
        for idx, marker in skip_list[:10]:
            print(f"  [{idx:4d}] {marker}")
        if len(skip_list) > 10:
            print(f"  ... and {len(skip_list) - 10} more")
        print()
    
    if onset_offset_list:
        print("Onset/Offset markers (filtered out):")
        for idx, marker in onset_offset_list[:10]:
            print(f"  [{idx:4d}] {marker}")
        if len(onset_offset_list) > 10:
            print(f"  ... and {len(onset_offset_list) - 10} more")
        print()
    
    # Filter consecutive duplicates (keep middle)
    filtered_markers = []
    i = 0
    while i < len(valid_markers_with_ts):
        idx, marker, marker_ts, label = valid_markers_with_ts[i]
        
        # Find consecutive markers with same label
        j = i
        while j < len(valid_markers_with_ts) and valid_markers_with_ts[j][3] == label:
            j += 1
        
        # Count consecutive markers with same label
        count = j - i
        if count == 1:
            # Single marker, keep it
            filtered_markers.append((idx, marker, marker_ts, label, 1))
        else:
            # Multiple consecutive markers with same label, keep middle
            middle_idx_in_group = count // 2
            middle_full_idx = i + middle_idx_in_group
            middle_data = valid_markers_with_ts[middle_full_idx]
            filtered_markers.append((middle_data[0], middle_data[1], middle_data[2], middle_data[3], count))
        
        i = j
    
    # Print the kept markers
    print(f"FINAL MARKERS (after deduplication):\n")
    print(f"{'Stream#':<8} {'Label':<8} {'Marker String':<60} {'Info':<20}")
    print(f"{'-'*8} {'-'*8} {'-'*60} {'-'*20}")
    
    label_counts = defaultdict(int)
    for stream_idx, marker, marker_ts, label, group_count in filtered_markers:
        label_str = 'no task' if label == 0 else 'task'
        label_counts[label_str] += 1
        
        if group_count > 1:
            info = f"[group of {group_count}, kept middle]"
        else:
            info = "[single]"
        
        print(f"{stream_idx:<8} {label_str:<8} {marker:<60} {info:<20}")
    
    print(f"\n{'='*100}")
    print("LABEL DISTRIBUTION:")
    print(f"  no task (0): {label_counts['no task']}")
    print(f"  task (1):    {label_counts['task']}")
    print(f"  Total kept:  {sum(label_counts.values())}")
    print(f"{'='*100}\n")


def main():
    """Main function."""
    print("\nMarker Filtering Visualization Tool")
    print("====================================\n")
    
    # Process both subjects
    subjects = ['sub-PD089', 'sub-PD094']
    
    for subject_id in subjects:
        visualize_marker_filtering(subject_id)


if __name__ == '__main__':
    main()
