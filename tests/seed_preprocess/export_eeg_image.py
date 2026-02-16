"""Export EEG signals as image without grid, excluding specific channels."""

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


def export_eeg_as_image(subject_id: int = None, session_id: int = 1, trial_id: int = 1,
                        start_time: float = 20, duration: float = 30,
                        exclude_channels: list = None, output_path: str = None, format: str = 'pdf',
                        vertical_scale: float = 1.7, first_n_channels: int = None):
    """
    Export EEG signals as image without grid.
    
    Args:
        subject_id: Subject ID (1-15), shows first file if not specified
        session_id: Session ID (1-3, corresponds to date order), default 1
        trial_id: Movie trial index (1-15, default 1)
        start_time: Start time in seconds from beginning of recording
        duration: Duration to display in seconds
        exclude_channels: List of channel names to exclude
        output_path: Path to save the image. If None, uses default naming
        format: Output format ('pdf', 'svg', or 'png'), default 'pdf' (vector format)
        vertical_scale: Vertical scaling factor for channel separation (default 1.0, increase for more spacing)
        first_n_channels: If specified, keep only the first N channels (after exclusions)
    """
    if exclude_channels is None:
        # Default: remove non-EEG channels + CPz, Cz
        exclude_channels = ['M1', 'M2', 'VEO', 'HEO', 'CPz', 'Cz']
    
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
    
    # Sampling frequency - original preprocessed data is at 200 Hz
    sfreq = 200  # Hz
    
    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    
    # Create RawArray
    raw = mne.io.RawArray(eeg_data, info)
    
    # Keep only first N channels if specified (before exclusions)
    if first_n_channels is not None:
        if first_n_channels > len(ch_names):
            print(f"✗ Requested {first_n_channels} channels but only {len(ch_names)} available")
            return
        channels_to_keep = ch_names[:first_n_channels]
        channels_to_drop = ch_names[first_n_channels:]
        print(f"\n✓ Keeping only first {first_n_channels} channels")
        raw.drop_channels(channels_to_drop)
        ch_names = raw.ch_names
    
    # Drop excluded channels (from remaining channels)
    channels_to_drop = [ch for ch in exclude_channels if ch in ch_names]
    if channels_to_drop:
        print(f"✓ Excluding channels: {channels_to_drop}")
        raw.drop_channels(channels_to_drop)
    
    ch_names = raw.ch_names
    
    # Print info
    print(f"\nEEG Info:")
    print(f"  Total channels: {len(ch_names)}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Total duration: {eeg_data.shape[1] / sfreq:.1f} seconds")
    
    # Validate start time and duration
    total_duration = eeg_data.shape[1] / sfreq
    if start_time < 0:
        print(f"✗ Start time cannot be negative")
        return
    if start_time >= total_duration:
        print(f"✗ Start time ({start_time}s) exceeds total duration ({total_duration:.1f}s)")
        return
    
    actual_duration = min(duration, total_duration - start_time)
    
    print(f"  Plot start time: {start_time:.1f}s")
    print(f"  Plot duration: {actual_duration:.1f}s")
    
    # Print data statistics
    data = raw.get_data()
    print(f"\nData Statistics:")
    print(f"  Mean: {np.mean(data):.4f} µV")
    print(f"  Std:  {np.std(data):.4f} µV")
    print(f"  Min:  {np.min(data):.4f} µV")
    print(f"  Max:  {np.max(data):.4f} µV")
    
    # Create figure with no grid
    print(f"\nGenerating image...")
    
    try:
        # Get time vector
        times = raw.times
        
        # Find indices for start_time and duration
        start_idx = int(start_time * sfreq)
        end_idx = int((start_time + actual_duration) * sfreq)
        
        time_window = times[start_idx:end_idx]
        
        # Create single plot with all channels stacked - larger horizontal size
        fig, ax = plt.subplots(figsize=(22, 10))
        
        # Calculate offset for each channel (reduced for better visibility)
        max_amplitude = np.percentile(np.abs(data[:, start_idx:end_idx]), 95)
        offset = max_amplitude * 1.2 * vertical_scale
        
        # Plot all channels stacked
        for idx, ch_name in enumerate(ch_names):
            ch_data = data[idx, start_idx:end_idx]
            ax.plot(time_window, ch_data + (idx * offset), linewidth=0.7, color='black')
        
        # Remove all borders and padding
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(False)
        ax.margins(0)  # Remove margins
        
        plt.tight_layout(pad=0)
        
        # Determine output path and format
        if output_path is None:
            output_dir = Path("tests/seed_preprocess/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"eeg_s{subject_id or 1}_se{session_id}_t{trial_id}_t{start_time:.1f}_{actual_duration:.1f}s.{format}"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in vector format (PDF/SVG are scalable) or raster (PNG)
        if format.lower() == 'pdf':
            plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
        elif format.lower() == 'svg':
            plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
        else:  # png or other raster formats
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"✓ Image saved to: {output_path} ({format.upper()})")
        
        plt.close(fig)
        
    except Exception as e:
        print(f"✗ Error during image generation: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Export EEG signals as image without grid"
    )
    parser.add_argument('-s', '--subject', type=int, help='Subject ID (1-15), optional')
    parser.add_argument('-se', '--session', type=int, default=1, help='Session ID (1-3, default 1)')
    parser.add_argument('-t', '--trial', type=int, default=1, help='Movie trial index (1-15, default 1)')
    parser.add_argument('-st', '--start-time', type=float, default=0, help='Start time in seconds (default 0)')
    parser.add_argument('-d', '--duration', type=float, default=10, help='Duration in seconds (default 10)')
    parser.add_argument('-e', '--exclude', nargs='+', help='Channel names to exclude (in addition to defaults)')
    parser.add_argument('-o', '--output', type=str, help='Output file path (default: auto-named in outputs/)')
    parser.add_argument('-f', '--format', type=str, default='pdf', choices=['svg', 'pdf', 'png'], help='Output format: pdf (vector, default), svg (vector), or png (raster)')
    parser.add_argument('-v', '--vertical-scale', type=float, default=1.0, help='Vertical scaling factor for channel separation (default 1.0)')
    parser.add_argument('-n', '--first-n-channels', type=int, help='Keep only first N channels (e.g., 20 for first 20 channels)')
    
    # EXAMPLE USAGE
    # Export subject 2, session 1, trial 1, from 0s for 30s, excluding default channels
    # uv run tests/seed_preprocess/export_eeg_image.py -s 2 -se 1 -t 1 -st 0 -d 30
    #
    # Custom start time and duration
    # uv run tests/seed_preprocess/export_eeg_image.py -s 2 -se 1 -t 1 -st 10 -d 15
    #
    # Custom output path
    # uv run tests/seed_preprocess/export_eeg_image.py -s 2 -se 1 -t 1 -o ./my_eeg_image.png

    args = parser.parse_args()
    
    exclude_channels = ['M1', 'M2', 'VEO', 'HEO', 'CPz', 'Cz']
    if args.exclude:
        exclude_channels.extend(args.exclude)
    
    export_eeg_as_image(
        args.subject,
        args.session,
        args.trial,
        args.start_time,
        args.duration,
        exclude_channels=exclude_channels,
        output_path=args.output,
        format=args.format,
        vertical_scale=args.vertical_scale,
        first_n_channels=args.first_n_channels
    )


if __name__ == '__main__':
    main()
