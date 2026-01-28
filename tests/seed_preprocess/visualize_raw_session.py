"""Visualize raw SEED EEG sessions (before preprocessing)."""

import argparse
import mne
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from preprocess_seed.seed_loader import SEEDEEGLoader
from preprocess_seed.seed_preprocessing_config import SEEDPreprocessingConfig


def visualize_raw_session(subject_id: int, session_id: int, 
                         n_channels_display: int = None, 
                         duration: float = 60):
    """
    Visualize raw EEG session from CNT file.
    
    Args:
        subject_id: Subject ID (1-15)
        session_id: Session ID (1-3)
        n_channels_display: Number of channels to display at once (None = all)
        duration: Time window to display in seconds
    """
    config = SEEDPreprocessingConfig()
    
    print(f"\nLoading raw session: Subject {subject_id:02d}, Session {session_id}...")
    
    # Initialize loader
    loader = SEEDEEGLoader(
        seed_raw_dir=config.seed_raw_dir,
        montage_path=config.montage_file
    )
    
    # Load raw data
    try:
        raw = loader.load_raw(subject_id, session_id)
    except Exception as e:
        print(f"Error loading raw data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get info
    n_channels = len(raw.ch_names)
    sfreq = raw.info['sfreq']
    duration_sec = len(raw) / sfreq
    
    print(f"  Shape: {n_channels} channels × {len(raw):,} samples")
    print(f"  Duration: {duration_sec:.1f} seconds at {sfreq} Hz")
    print(f"  Montage: {len(raw.info['dig'])} points")
    
    # Default to all channels if not specified
    if n_channels_display is None:
        n_channels_display = n_channels
    
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
            title=f"Raw EEG: Subject {subject_id:02d}, Session {session_id}",
            show=True
        )
        plt.show()  # Keep window open
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


def plot_raw_psd(subject_id: int, session_id: int):
    """Plot power spectral density of raw session."""
    config = SEEDPreprocessingConfig()
    
    print(f"\nLoading raw session: Subject {subject_id:02d}, Session {session_id}...")
    
    loader = SEEDEEGLoader(
        seed_raw_dir=config.seed_raw_dir,
        montage_path=config.montage_file
    )
    
    try:
        raw = loader.load_raw(subject_id, session_id)
        print(f"Plotting power spectrum...")
        raw.plot_psd(fmin=0, fmax=50, title=f"Power Spectrum: Subject {subject_id:02d}, Session {session_id}")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def plot_raw_topomap(subject_id: int, session_id: int):
    """Plot topographic map of raw session mean."""
    config = SEEDPreprocessingConfig()
    
    print(f"\nLoading raw session: Subject {subject_id:02d}, Session {session_id}...")
    
    loader = SEEDEEGLoader(
        seed_raw_dir=config.seed_raw_dir,
        montage_path=config.montage_file
    )
    
    try:
        raw = loader.load_raw(subject_id, session_id)
        
        # Compute mean amplitude across time
        data = raw.get_data()
        mean_amplitude = data.mean(axis=1)
        
        print(f"Plotting topographic map...")
        mne.viz.plot_topomap(mean_amplitude, raw.info, show=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw SEED EEG sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize raw session
  python visualize_raw_session.py -s 1 -se 1
  
  # Display only 30 channels
  python visualize_raw_session.py -s 1 -se 1 -n 30
  
  # Show first 5 minutes
  python visualize_raw_session.py -s 1 -se 1 -d 300
  
  # Plot power spectrum
  python visualize_raw_session.py -s 1 -se 1 --psd
  
  # Plot topographic map
  python visualize_raw_session.py -s 1 -se 1 --topomap
        """
    )
    
    parser.add_argument('-s', '--subject', type=int, required=True,
                       help='Subject ID (1-15)')
    parser.add_argument('-se', '--session', type=int, required=True,
                       help='Session ID (1-3)')
    parser.add_argument('-n', '--n-channels', type=int, default=None,
                       help='Number of channels to display (default: all)')
    parser.add_argument('-d', '--duration', type=float, default=60,
                       help='Duration window in seconds (default: 60)')
    parser.add_argument('--psd', action='store_true',
                       help='Plot power spectral density instead')
    parser.add_argument('--topomap', action='store_true',
                       help='Plot topographic map instead')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.subject < 1 or args.subject > 15:
        print("Error: Subject ID must be 1-15")
        sys.exit(1)
    
    if args.session < 1 or args.session > 3:
        print("Error: Session ID must be 1-3")
        sys.exit(1)
    
    # Visualize
    if args.psd:
        plot_raw_psd(args.subject, args.session)
    elif args.topomap:
        plot_raw_topomap(args.subject, args.session)
    else:
        visualize_raw_session(args.subject, args.session, args.n_channels, args.duration)


if __name__ == "__main__":
    main()
