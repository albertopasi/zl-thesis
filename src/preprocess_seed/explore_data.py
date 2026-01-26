"""Explore SEED EEG dataset - basic information and statistics."""

import sys
from pathlib import Path
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocess_seed.seed_loader import SEEDEEGLoader
from src.preprocess_seed.config import SEEDConfig


def load_time_markers(seed_raw_dir):
    """Load actual recording time markers from time.txt file."""
    time_file = Path(seed_raw_dir) / "time.txt"
    
    try:
        with open(time_file, 'r') as f:
            content = f.read()
        
        # Extract start and end point lists
        start_match = re.search(r'start_point_list\s*=\s*\[(.*?)\]', content)
        end_match = re.search(r'end_point_list\s*=\s*\[(.*?)\]', content)
        
        if start_match and end_match:
            start_points = [int(x.strip()) for x in start_match.group(1).split(',')]
            end_points = [int(x.strip()) for x in end_match.group(1).split(',')]
            return start_points, end_points
    except Exception as e:
        print(f"Warning: Could not load time markers: {e}")
    
    return None, None


def main():
    """Load and explore SEED data."""
    config = SEEDConfig()
    loader = SEEDEEGLoader(config.seed_raw_dir, config.montage_file)
    
    # Get available subjects and sessions
    subjects = loader.get_subject_sessions()
    
    print("\n" + "="*70)
    print("SEED DATASET EXPLORATION")
    print("="*70)
    
    print(f"\nTotal Subjects: {len(subjects)}")
    print(f"Sessions per Subject: {len(list(subjects.values())[0]) if subjects else 0}")
    print(f"Total Subject-Session Combinations: {sum(len(s) for s in subjects.values())}")
    
    print("\nAvailable Subjects and Sessions:")
    for subject_id in sorted(subjects.keys()):
        sessions = sorted(subjects[subject_id])
        print(f"  Subject {subject_id:2d}: Sessions {sessions}")
    
    print("\n" + "="*70)
    print("MONTAGE INFORMATION")
    print("="*70)
    
    try:
        raw_sample = loader.load_raw(1, 1)
        montage = raw_sample.get_montage()
        pos = montage.get_positions()
        ch_pos = pos['ch_pos']
        
        # Get channels that have valid positions
        channels_with_pos = [ch for ch in ch_pos.keys() if ch_pos[ch] is not None]
        
        print(f"\nTotal Channels with Positions: {len(channels_with_pos)}")
        print(f"Channel Names:")
        for i, ch in enumerate(sorted(channels_with_pos), 1):
            if i % 8 == 0:
                print(f"  {ch}")
            else:
                print(f"  {ch:6s}", end="")
        print()
    except Exception as e:
        print(f"\nCould not load montage info from sample file")
        print(f"Error: {type(e).__name__}")

    # Load first file to get data info
    print("\n" + "="*70)
    print("SAMPLE RAW DATA (Subject 1, Session 1)")
    print("="*70)
    
    # Load time markers
    start_points, end_points = load_time_markers(config.seed_raw_dir)
    
    try:
        raw = loader.load_raw(1, 1)
        
        print(f"\nFile: 1_1.cnt")
        print(f"Number of Channels: {len(raw.ch_names)}")
        print(f"Sampling Rate: {raw.info['sfreq']} Hz")
        
        # Calculate actual recording duration from time markers
        if start_points and end_points:
            sfreq = raw.info['sfreq']
            print(f"\nAll {len(start_points)} Movie/Trial Windows (from time.txt):")
            print(f"{'Trial':<8} {'Start Sample':<15} {'End Sample':<15} {'Duration (sec)':<16} {'Duration (min)':<12}")
            print("-" * 70)
            
            for trial_idx, (start_sample, end_sample) in enumerate(zip(start_points, end_points), 1):
                duration_sec = (end_sample - start_sample) / sfreq
                duration_min = duration_sec / 60
                print(f"{trial_idx:<8} {start_sample:<15} {end_sample:<15} {duration_sec:<16.2f} {duration_min:<12.2f}")
        else:
            # Fall back to file header (which may be corrupted)
            print(f"Recording Duration (from header): {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
            print(f"Total Samples (from header): {raw.n_times}")
        
        print(f"\nChannel Names in Raw File:")
        for i, ch in enumerate(raw.ch_names, 1):
            if i % 8 == 0:
                print(f"  {ch}")
            else:
                print(f"  {ch:6s}", end="")
        print()
        
        
        # Check montage coverage
        montage = raw.get_montage()
        pos = montage.get_positions()
        ch_pos = pos['ch_pos']
        
        valid_positions = sum(1 for p in ch_pos.values() if p is not None)
        print(f"\nChannels with Valid Electrode Positions: {valid_positions}/{len(ch_pos)}")
        
        # Info about channel types
        print(f"\nChannel Information:")
        for ch_name in raw.ch_names:
            ch_type = raw.get_channel_types([ch_name])[0]
            print(f"  {ch_name:6s}: {ch_type}")
        
        # Derive data quality notes from actual data
        print("\n" + "="*70)
        print("DATA QUALITY NOTES")
        print("="*70)
        
        # Count extra channels (those not in the 62-channel montage)
        # Use case-insensitive comparison
        montage_channels_upper = set(ch.upper() for ch in loader.channels)
        raw_channels_upper = set(ch.upper() for ch in raw.ch_names)
        extra_channels = raw_channels_upper - montage_channels_upper
        
        eeg_channel_count = len(montage_channels_upper)
        extra_channel_count = len(extra_channels)
        total_channels = len(raw.ch_names)
        
        # Check if all EEG channels have positions
        eeg_channels_with_pos = sum(1 for ch in montage_channels_upper if ch in [k.upper() for k, v in ch_pos.items() if v is not None])
        
        print(f"- Total channels in raw file: {total_channels}")
        print(f"- EEG channels (from montage): {eeg_channel_count}")
        print(f"- Extra channels (not in montage): {extra_channel_count} ({', '.join(sorted(extra_channels))})")
        print(f"- EEG channels with valid electrode positions: {eeg_channels_with_pos}/{eeg_channel_count}")
        print(f"- Sampling rate: {int(raw.info['sfreq'])} Hz")
            
    except Exception as e:
        print(f"\nNote: Could not fully load raw data due to header issues")
        print(f"Error: {type(e).__name__}: {str(e)[:100]}")
        print("\nThis is expected - some .cnt files have corrupted headers")

if __name__ == "__main__":
    main()
