#!/usr/bin/env python3
"""
Compare original markers with extracted epochs.

This script reads:
1. Original markers CSV (from print_all_events.py)
2. Extracted epochs CSV (from debug_epochs.py)

And shows them side-by-side for manual verification.
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


def print_comparison(subject_id: str = 'sub-PD089', session_id: str = 'ses-S001'):
    """Print side-by-side comparison of original markers vs extracted epochs."""
    debug_output = Path(__file__).parent / "debug_output"
    
    # Read both files
    original_markers_file = debug_output / f"{subject_id}_{session_id}_markers.csv"
    extracted_epochs_file = debug_output / f"{subject_id}_{session_id}_extracted_epochs.csv"
    
    if not original_markers_file.exists():
        print(f"ERROR: Original markers file not found: {original_markers_file}")
        return
    
    if not extracted_epochs_file.exists():
        print(f"ERROR: Extracted epochs file not found: {extracted_epochs_file}")
        return
    
    print("\n" + "="*120)
    print(f"MARKERS COMPARISON: {subject_id} / {session_id}")
    print("="*120)
    print(f"\nOriginal markers file: {original_markers_file}")
    print(f"Extracted epochs file: {extracted_epochs_file}\n")
    
    # Read data
    original_markers = read_csv(original_markers_file)
    extracted_epochs = read_csv(extracted_epochs_file)
    
    print(f"Total original markers: {len(original_markers)}")
    print(f"Total extracted epochs: {len(extracted_epochs)}\n")
    
    # Print extracted epochs
    print("\n" + "-"*120)
    print("EXTRACTED EPOCHS (what you're keeping):")
    print("-"*120)
    print(f"{'Idx':<5} {'Label':<12} {'Time (s)':<15} {'Sample Index':<15} {'Description':<50}")
    print("-"*120)
    
    for epoch in extracted_epochs:
        idx = epoch['EpochIndex']
        label = epoch['LabelStr']
        time_sec = epoch['TimeInSeconds']
        sample_idx = epoch['SampleIndex']
        desc = epoch['NotesForComparison']
        
        print(f"{idx:<5} {label:<12} {time_sec:<15} {sample_idx:<15} {desc:<50}")
    
    # Print guidance for manual comparison
    print("\n" + "="*120)
    print("HOW TO VERIFY MANUALLY:")
    print("="*120)
    print("""
1. Open the extracted epochs CSV in a spreadsheet
2. For each epoch time in TimeInSeconds, search the original markers CSV for markers near that time
3. Verify that:
   - The extracted epoch timestamps correspond to the markers you WANT to keep
   - The labels (task/no-task) are correct
   - No important markers are being skipped

The original CSV columns are:
  - Timestamp: Marker timestamp in recording time
  - Original_Marker: Full marker text
  - Status: Processing status
  - Label: Label applied
  
The extracted epochs are centered at TimeInSeconds (after downsampling to 200 Hz).
The epoch window is [-1.5s, +1.5s], so the full epoch spans 3 seconds.

Example verification:
  - If extracted epoch has TimeInSeconds = 223.0050 and label = 'no task'
  - Search original markers around timestamp 223.0050
  - Verify that a marker like "no task - ..." exists at that time
""")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        subject_id = sys.argv[1]
        session_id = sys.argv[2] if len(sys.argv) > 2 else 'ses-S001'
        print_comparison(subject_id, session_id)
    else:
        # Use defaults
        print_comparison()


if __name__ == "__main__":
    main()
