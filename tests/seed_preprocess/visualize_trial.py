"""Visualize preprocessed SEED EEG trials."""

import argparse
import numpy as np
import mne
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from preprocess_seed.seed_preprocessing_config import SEEDPreprocessingConfig


def visualize_trial(trial_file: Path, montage_file: Path = None, 
                   n_channels_display: int = None, duration: float = 10):
    """
    Visualize a preprocessed trial.
    
    Args:
        trial_file: Path to .npy trial file
        montage_file: Path to electrode montage file (.locs)
        n_channels_display: Number of channels to display at once (None = all)
        duration: Time window to display in seconds
    """
    # Load config for filter values
    config = SEEDPreprocessingConfig()
    
    # Load trial data
    trial_data = np.load(trial_file)
    n_channels, n_samples = trial_data.shape
    
    print(f"\nLoading trial: {trial_file.name}")
    print(f"  Shape: {n_channels} channels × {n_samples} samples")
    
    # Determine sampling rate from file size (assuming standard preprocessing)
    # After downsampling to 200 Hz
    sfreq = 200  # Hz
    duration_sec = n_samples / sfreq
    print(f"  Duration: {duration_sec:.1f} seconds at {sfreq} Hz")
    
    # Default to all channels if not specified
    if n_channels_display is None:
        n_channels_display = n_channels
    
    # Create channel names (62 channels in SEED)
    if n_channels == 62:
        # Standard SEED EEG channels
        ch_names = [f"Ch{i:02d}" for i in range(1, 63)]
    else:
        # Generic channel names
        ch_names = [f"Ch{i:02d}" for i in range(n_channels)]
    
    # Create MNE Info object
    info = mne.create_info(
        ch_names=ch_names, 
        sfreq=sfreq, 
        ch_types='eeg'
    )
    
    # Create Raw object (data is already channels × samples)
    raw = mne.io.RawArray(trial_data, info, verbose=False)
    
    # Apply filter from config
    raw.filter(config.filter_lowcut_hz, config.filter_highcut_hz, verbose=False)
    print(f"  Applied filter: {config.filter_lowcut_hz}-{config.filter_highcut_hz} Hz")
    
    # Load and apply montage if provided
    if montage_file and Path(montage_file).exists():
        try:
            montage = mne.channels.read_custom_montage(
                montage_file,
                head_size=1.0,
                coord_frame='head'
            )
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            print(f"  Montage loaded: {Path(montage_file).name}")
        except Exception as e:
            print(f"  Warning: Could not load montage: {e}")
    
    # Display info
    print(f"\n{'='*70}")
    print(f"Raw object info:")
    print(f"{'='*70}")
    print(raw.info)
    
    # Plot the signal
    print(f"\nOpening interactive plot...")
    print(f"  Displaying {n_channels_display} channels")
    print(f"  Use arrow keys to navigate, scroll to zoom")
    print(f"  Close the window to exit\n")
    
    try:
        fig = raw.plot(
            duration=min(duration, duration_sec),
            n_channels=n_channels_display,
            scalings='auto',
            title=f"Trial: {trial_file.parent.parent.name}/{trial_file.parent.name}/{trial_file.name}",
            show=True
        )
        plt.show()  # Keep window open
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


def plot_power_spectrum(trial_file: Path, montage_file: Path = None):
    """Plot power spectral density."""
    trial_data = np.load(trial_file)
    n_channels, n_samples = trial_data.shape
    sfreq = 200
    
    ch_names = [f"Ch{i:02d}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(trial_data, info, verbose=False)
    
    if montage_file and Path(montage_file).exists():
        try:
            montage = mne.channels.read_custom_montage(montage_file, head_size=1.0, coord_frame='head')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        except:
            pass
    
    print(f"Plotting power spectrum for: {trial_file.name}")
    raw.plot_psd(fmin=0, fmax=50, title=f"Power Spectrum: {trial_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed SEED EEG trials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a specific trial file
  python visualize_trial.py data/SEED/preprocessed_seed/sub-01/ses-1/trial-01.npy
  
  # Visualize by subject/session/trial
  python visualize_trial.py -s 1 -se 1 -t 1
  
  # Plot power spectrum
  python visualize_trial.py -s 1 -se 1 -t 1 --psd
  
  # Display 30 channels at once
  python visualize_trial.py -s 1 -se 1 -t 1 -n 30
        """
    )
    
    parser.add_argument('trial_file', nargs='?', default=None,
                       help='Path to trial .npy file')
    parser.add_argument('-s', '--subject', type=int, default=None,
                       help='Subject ID (1-15)')
    parser.add_argument('-se', '--session', type=int, default=None,
                       help='Session ID (1-3)')
    parser.add_argument('-t', '--trial', type=int, default=None,
                       help='Trial ID')
    parser.add_argument('-n', '--n-channels', type=int, default=None,
                       help='Number of channels to display (default: all)')
    parser.add_argument('-d', '--duration', type=float, default=10,
                       help='Duration window in seconds (default: 10)')
    parser.add_argument('--psd', action='store_true',
                       help='Plot power spectral density instead')
    parser.add_argument('-m', '--montage', type=str, default=None,
                       help='Path to montage file (.locs)')
    
    args = parser.parse_args()
    
    # Determine trial file path
    if args.trial_file:
        trial_file = Path(args.trial_file)
    elif args.subject and args.session and args.trial:
        # Construct path from subject/session/trial
        preprocessed_dir = Path(__file__).parent.parent.parent / "data" / "SEED" / "preprocessed_seed"
        trial_file = (
            preprocessed_dir / 
            f"sub-{args.subject:02d}" / 
            f"ses-{args.session}" / 
            f"trial-{args.trial:02d}.npy"
        )
    else:
        parser.print_help()
        print("\nError: Either provide trial_file or use -s -se -t options")
        sys.exit(1)
    
    # Check file exists
    if not trial_file.exists():
        print(f"Error: Trial file not found: {trial_file}")
        sys.exit(1)
    
    # Default montage path
    if args.montage is None:
        montage_candidate = Path(__file__).parent.parent.parent / "data" / "SEED" / "channel_62_pos.locs"
        if montage_candidate.exists():
            args.montage = str(montage_candidate)
    
    # Visualize
    if args.psd:
        plot_power_spectrum(trial_file, args.montage)
    else:
        visualize_trial(trial_file, args.montage, args.n_channels, args.duration)


if __name__ == "__main__":
    main()
