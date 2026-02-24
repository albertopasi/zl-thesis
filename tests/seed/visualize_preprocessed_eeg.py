"""Visualize preprocessed EEG signals from seed/preprocessed_seed using MNE."""

import argparse
import mne
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from preprocess_seed.seed_preprocessing_config import SEEDPreprocessingConfig


def visualize_preprocessed_eeg(subject_id: int = 1, session_id: int = 1, 
                               trial_id: int = 1, n_channels_display: int = None, 
                               duration: float = 15):
    """
    Visualize preprocessed EEG signals from .npy files using MNE.
    
    Args:
        subject_id: Subject ID (1-15, default 1)
        session_id: Session ID (1-3, default 1)
        trial_id: Trial ID (1-15, default 1)
        n_channels_display: Number of channels to display at once
        duration: Time window to display in seconds
    """
    config = SEEDPreprocessingConfig()
    
    # Find available sessions for this subject
    subject_dir = Path(config.preprocessed_output_dir) / f"sub-{subject_id:02d}"
    
    if not subject_dir.exists():
        print(f"✗ Subject directory not found: {subject_dir}")
        return
    
    # Find all session folders (they're named ses-1, ses-2, etc. or session1, session2, etc.)
    session_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
    
    if not session_dirs:
        print(f"✗ No session folders found in {subject_dir}")
        return
    
    if session_id < 1 or session_id > len(session_dirs):
        print(f"✗ Session {session_id} not found (available: 1-{len(session_dirs)})")
        return
    
    # Get the session folder by index
    preprocessed_dir = session_dirs[session_id - 1]
    
    # Find trial file
    npy_files = sorted(preprocessed_dir.glob("trial-*.npy"))
    
    if not npy_files:
        print(f"✗ No trial files found in {preprocessed_dir}")
        return
    
    if trial_id < 1 or trial_id > len(npy_files):
        print(f"✗ Trial {trial_id} not found (available: 1-{len(npy_files)})")
        return
    
    trial_file = npy_files[trial_id - 1]
    print(f"\nLoading: {trial_file.name}")
    
    # Load .npy file
    eeg_data = np.load(trial_file)  # Shape: (62, samples)
    
    print(f"✓ Loaded shape {eeg_data.shape}")
    
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
    
    # Sampling frequency - preprocessed data is downsampled to 200 Hz
    sfreq = config.downsample_freq_hz  # 200 Hz
    
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
    
    # Per-channel statistics
    print(f"\nPer-channel statistics (first 10 channels):")
    print(f"{'Channel':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*56}")
    
    for ch_idx in range(min(10, eeg_data.shape[0])):
        ch_data = eeg_data[ch_idx]
        print(f"{ch_names[ch_idx]:<8} {np.mean(ch_data):>10.4f}  {np.std(ch_data):>10.4f}  {np.min(ch_data):>10.4f}  {np.max(ch_data):>10.4f}")
    
    # Default to all channels if not specified
    if n_channels_display is None:
        n_channels_display = len(ch_names)
    
    # Plot
    print(f"\nOpening interactive plot...")
    print(f"  Displaying {n_channels_display} channels")
    print(f"  Use arrow keys to navigate, scroll to zoom")
    print(f"  Close the window to exit\n")
    
    try:
        fig = raw.plot(
            duration=min(duration, eeg_data.shape[1] / sfreq),
            n_channels=n_channels_display,
            scalings='auto',
            title=f"Preprocessed EEG: Subject {subject_id:02d}, Session {session_id}, Trial {trial_id}",
            show=True
        )
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed EEG signals from seed/preprocessed_seed using MNE"
    )
    parser.add_argument('-s', '--subject', type=int, default=1, help='Subject ID (1-15, default 1)')
    parser.add_argument('-se', '--session', type=int, default=1, help='Session ID (1-3, default 1)')
    parser.add_argument('-t', '--trial', type=int, default=1, help='Trial ID (1-15, default 1)')
    parser.add_argument('-n', '--n-channels', type=int, help='Number of channels to display')
    parser.add_argument('-d', '--duration', type=float, default=60, help='Duration to display in seconds')
    
    # EXAMPLE USAGE
    # View subject 1, session 1, trial 1, all channels
    # uv run tests/seed_preprocess/visualize_preprocessed_eeg.py -s 1 -se 1 -t 1

    args = parser.parse_args()
    
    visualize_preprocessed_eeg(args.subject, args.session, args.trial, args.n_channels, args.duration)


if __name__ == '__main__':
    main()
