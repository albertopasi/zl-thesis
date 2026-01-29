"""Visualize original preprocessed EEG signals using MNE."""

import argparse
import mne
import numpy as np
from pathlib import Path
import sys
import scipy.io
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from preprocess_seed.seed_preprocessing_config import SEEDPreprocessingConfig


def visualize_mat_eeg(subject_id: int = None, session_id: int = 1, trial_id: int = 1, 
                      n_channels_display: int = None, duration: float = 30):
    """
    Visualize EEG signals from original preprocessed .mat file using MNE.
    
    Args:
        subject_id: Subject ID (1-15), shows first file if not specified
        session_id: Session ID (1-3, corresponds to date order), default 1
        trial_id: Movie trial index (1-15, default 1)
        n_channels_display: Number of channels to display at once
        duration: Time window to display in seconds
    """
    config = SEEDPreprocessingConfig()
    
    # Find .mat file
    preprocessed_dir = Path("data/SEED/Preprocessed_EEG_ORIGINAL")
    mat_files = sorted([f for f in preprocessed_dir.glob("*.mat") if f.name != "label.mat"])
    
    if not mat_files:
        print(f"✗ No .mat files found")
        return
    
    if subject_id is not None:
        subject_files = sorted([f for f in mat_files if f.name.startswith(f"{subject_id}_")])
        if not subject_files:
            print(f"✗ No files found for subject {subject_id}")
            return
        
        if session_id < 1 or session_id > len(subject_files):
            print(f"✗ Session {session_id} not found for subject {subject_id} (available: 1-{len(subject_files)})")
            return
        
        mat_file = subject_files[session_id - 1]
    else:
        mat_file = mat_files[0]
    
    print(f"\nLoading: {mat_file.name}")
    
    # Load .mat file
    mat_data = scipy.io.loadmat(str(mat_file))
    
    # Get the correct EEG variable for this trial
    # Variables are like 'djc_eeg1', 'djc_eeg2', etc. or 'ww_eeg1', etc.
    eeg_var = None
    for key in mat_data.keys():
        if key.endswith(f"eeg{trial_id}"):
            eeg_var = key
            break
    
    if eeg_var is None:
        print(f"✗ EEG variable for trial {trial_id} not found")
        print(f"Available: {[k for k in mat_data.keys() if 'eeg' in k]}")
        return
    
    eeg_data = mat_data[eeg_var]  # Shape: (62, samples)
    
    print(f"✓ Loaded {eeg_var}: shape {eeg_data.shape}")
    
    # Load montage to get channel names and positions
    montage_path = Path(config.montage_file)
    if not montage_path.exists():
        print(f"✗ Montage file not found: {montage_path}")
        return
    
    montage = mne.channels.read_custom_montage(str(montage_path))
    ch_names = montage.ch_names
    
    if len(ch_names) != eeg_data.shape[0]:
        print(f"✗ Channel count mismatch: montage has {len(ch_names)}, data has {eeg_data.shape[0]}")
        return
    
    # Sampling frequency - original preprocessed data is at 200 Hz (downsampled)
    sfreq = 200  # Hz
    
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    
    # Create RawArray
    raw = mne.io.RawArray(eeg_data, info)
    
    # Print info
    print(f"\nEEG Info:")
    print(f"  Channels: {len(ch_names)}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Duration: {eeg_data.shape[1] / sfreq:.1f} seconds")
    print(f"  Digitized positions: {len(raw.info['dig'])}")
    
    # Print data statistics
    print(f"\nData Statistics:")
    print(f"  Mean: {np.mean(eeg_data):.4f} µV")
    print(f"  Std:  {np.std(eeg_data):.4f} µV")
    print(f"  Min:  {np.min(eeg_data):.4f} µV")
    print(f"  Max:  {np.max(eeg_data):.4f} µV")
    
    # Default to all channels if not specified
    if n_channels_display is None:
        n_channels_display = len(ch_names)
    
    # Plot
    print(f"\nOpening interactive plot...")
    print(f"  Displaying {n_channels_display} channels")
    print(f"  Use arrow keys to navigate, scroll to zoom")
    print(f"  Close the window to exit\n")
    
    try:
        # Calculate start time to show the LAST 'duration' seconds
        total_duration = eeg_data.shape[1] / sfreq
        start_time = max(0, total_duration - duration)
        
        fig = raw.plot(
            start=start_time,
            duration=min(duration, total_duration),
            n_channels=n_channels_display,
            scalings='auto',
            title=f"Original Preprocessed EEG: Subject {subject_id or mat_file.name.split('_')[0]}, Trial {trial_id}",
            show=True
        )
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original preprocessed EEG signals using MNE"
    )
    parser.add_argument('-s', '--subject', type=int, help='Subject ID (1-15), optional')
    parser.add_argument('-se', '--session', type=int, default=1, help='Session ID (1-3, corresponds to date order, default 1)')
    parser.add_argument('-t', '--trial', type=int, default=1, help='Movie trial index (1-15, default 1)')
    parser.add_argument('-n', '--n-channels', type=int, help='Number of channels to display')
    parser.add_argument('-d', '--duration', type=float, default=30, help='Duration to display in seconds (default 30)')
    
    # EXAMPLE USAGE
    # View subject 1, session 1, trial 1, all channels
    # uv run tests/seed_preprocess/visualize_original_eeg.py -s 1 -se 1 -t 1

    args = parser.parse_args()
    
    visualize_mat_eeg(args.subject, args.session, args.trial, args.n_channels, args.duration)


if __name__ == '__main__':
    main()
