"""
Example usage of the modular data loader.
Run this to verify the loader works with your dataset.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from data_loader import get_data_loader, DatasetRegistry


def example_basic_usage():
    """Example: Basic usage with auto-detection."""
    print("=" * 60)
    print("Example 1: Auto-detect and load ZL_Dataset")
    print("=" * 60)
    
    # Auto-detect dataset root
    loader = get_data_loader('zl')
    print(f"\n{loader}\n")
    
    # Get all subjects
    subjects = loader.get_all_subjects()
    print(f"Available subjects: {subjects}\n")
    
    # Get all (subject, session) pairs
    pairs = loader.get_all_subject_sessions()
    print(f"Total (subject, session) pairs: {len(pairs)}")
    for subj, sess in pairs[:5]:
        print(f"  {subj} / {sess}")
    print()


def example_load_subject():
    """Example: Load data for a specific subject."""
    print("=" * 60)
    print("Example 2: Load subject data from ZL_Dataset")
    print("=" * 60)
    
    loader = get_data_loader('zl')
    
    # Get first subject and session
    pairs = loader.get_all_subject_sessions()
    if pairs:
        subject_id, session_id = pairs[0]
        print(f"\nLoading {subject_id} / {session_id}...")
        
        data = loader.load_subject_data(subject_id, session_id)
        
        print(f"\nLoaded data keys: {data.keys()}")
        print(f"  EEG shape: {data['eeg'].shape} (samples, channels)")
        print(f"  EEG sampling rate: {data['sampling_rate']} Hz")
        print(f"  Number of markers: {len(data['markers'])}")
        print(f"  Unique markers: {set(data['markers'])}")
        print(f"  Channel labels: {data['channel_labels'][:5]}...")


def example_iterate_subjects():
    """Example: Iterate over all subjects and sessions."""
    print("=" * 60)
    print("Example 3: Iterate over all subjects in ZL_Dataset")
    print("=" * 60)
    
    loader = get_data_loader('zl')
    
    pairs = loader.get_all_subject_sessions()
    print(f"\nTotal datasets to process: {len(pairs)}\n")
    
    for i, (subject_id, session_id) in enumerate(pairs, 1):
        try:
            data = loader.load_subject_data(subject_id, session_id)
            print(f"[{i}/{len(pairs)}] {subject_id}/{session_id}: "
                  f"EEG {data['eeg'].shape}, {len(data['markers'])} markers")
        except Exception as e:
            print(f"[{i}/{len(pairs)}] {subject_id}/{session_id}: ERROR - {e}")


def example_register_custom():
    """Example: Register and use a custom dataset loader."""
    print("=" * 60)
    print("Example 4: Register custom dataset (template)")
    print("=" * 60)
    
    print("\nAvailable loaders:", DatasetRegistry.list_available())
    print("\n ZL_Dataset is registered as 'zl'")
    print("\nTo add a new dataset:")
    print("  1. Create a class that extends DataLoader")
    print("  2. Implement _discover_subjects(), get_eeg_file_path(), load_subject_data()")
    print("  3. Register it: DatasetRegistry.register('my_dataset', MyCustomLoader)")
    print("  4. Use it: get_data_loader('my_dataset', dataset_root)")


if __name__ == '__main__':
    try:
        example_basic_usage()
        print()
        
        example_load_subject()
        print()
        
        example_iterate_subjects()
        print()
        
        example_register_custom()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
