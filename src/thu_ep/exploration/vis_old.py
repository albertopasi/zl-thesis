"""
Visualize raw vs preprocessed THU-EP EEG segments as waveforms.

This script compares raw and preprocessed EEG data for a specific subject and stimulus.
It displays two EEG waveform plots stacked vertically:
- Top: Raw EEG segment (broad-band, band 5)
- Bottom: Preprocessed EEG segment

The visualization shows EEG traces as traditional waveforms (not heatmaps),
allowing detailed inspection of the preprocessing effects.

Usage:
    # Interactive viewer with sliders (RECOMMENDED)
    uv run python -m src.thu_ep.exploration.vis_old --viewer
    
    # Static plot for specific subject/stimulus
    uv run python -m src.thu_ep.exploration.vis_old --subject 1 --stimulus 0
    
    # Export comparison images
    uv run python -m src.thu_ep.exploration.vis_old --subject 1 --stimulus 0 --export outputs/comparison.png

Controls in interactive viewer:
    - Sliders: Change subject (1-80) and stimulus (0-27)
    - Keyboard: ←/→ = change stimulus, ↑/↓ = change subject, q = quit

"""

from pathlib import Path
from typing import Tuple, Optional
import argparse
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

try:
    import mne
except ImportError:
    print("MNE not installed.")
    sys.exit(1)


# Configuration
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "thu ep"
RAW_DATA_DIR = DATA_DIR / "EEG data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"

# Channel configuration
ALL_CHANNELS = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2"
]

CHANNELS_TO_REMOVE = ["A1", "A2"]
FINAL_CHANNELS = [ch for ch in ALL_CHANNELS if ch not in CHANNELS_TO_REMOVE]

# Sampling rates
ORIGINAL_SFREQ = 250.0  # Hz
TARGET_SFREQ = 200.0    # Hz

# Band configuration
BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broad-band"]
BROAD_BAND_INDEX = 5


def load_raw_mat_file(filepath: Path) -> np.ndarray:
    """
    Load raw EEG data from MATLAB v7.3 (HDF5) file.
    
    Args:
        filepath: Path to .mat file
        
    Returns:
        EEG data array with shape (7500, 32, 28, 6)
        Meaning: (samples, channels, stimuli, bands)
    """
    with h5py.File(filepath, 'r') as f:
        if 'data' in f:
            data = np.array(f['data'])
        else:
            keys = [k for k in f.keys() if not k.startswith('#')]
            if len(keys) == 1:
                data = np.array(f[keys[0]])
            else:
                raise ValueError(f"Cannot determine data key. Available: {list(f.keys())}")
    
    return data


def extract_raw_band(
    subject_id: int,
    stimulus_idx: int
) -> Tuple[np.ndarray, float, list]:
    """
    Load raw EEG data - only extract the broad-band, no other preprocessing.
    
    No downsampling, no channel removal, no normalization.
    Just the raw broad-band (0.5-47 Hz) signal at original 250 Hz.
    
    Args:
        subject_id: Subject number (1-80)
        stimulus_idx: Stimulus index (0-27)
        
    Returns:
        Tuple of:
        - EEG data (32, 7500) - all 32 channels, 7500 samples at 250 Hz
        - Sampling frequency (250 Hz)
        - Channel names (all 32)
    """
    filepath = RAW_DATA_DIR / f"sub_{subject_id}.mat"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    # Load raw data: (7500, 32, 28, 6)
    data = load_raw_mat_file(filepath)
    
    # Extract broad-band (band 5) only - no other preprocessing
    eeg = data[:, :, stimulus_idx, BROAD_BAND_INDEX]  # (7500, 32)
    
    # Transpose to (channels, samples) for MNE
    eeg = eeg.T  # (32, 7500)
    
    return eeg, ORIGINAL_SFREQ, ALL_CHANNELS.copy()


def load_preprocessed(subject_id: int, stimulus_idx: int) -> Tuple[np.ndarray, float, list]:
    """
    Load preprocessed EEG data.
    
    Args:
        subject_id: Subject number (1-80)
        stimulus_idx: Stimulus index (0-27)
        
    Returns:
        Tuple of:
        - EEG data (30, 6000) - 30 channels, 6000 samples
        - Sampling frequency (200 Hz)
        - Channel names
    """
    # Preprocessed files are named sub_XX.npy (zero-padded to 2 digits)
    filepath = PREPROCESSED_DATA_DIR / f"sub_{subject_id:02d}.npy"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {filepath}")
    
    # Load preprocessed data: (28, 30, 6000)
    data = np.load(filepath)
    
    # Extract stimulus: (30, 6000)
    eeg = data[stimulus_idx, :, :]
    
    sfreq = TARGET_SFREQ
    channels = FINAL_CHANNELS.copy()
    
    return eeg, sfreq, channels


def create_mne_info(channel_names: list, sfreq: float) -> mne.Info:
    """Create MNE Info object for EEG data."""
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types=['eeg'] * len(channel_names)
    )
    return info


def plot_comparison_simple(
    subject_id: int,
    stimulus_idx: int,
    save_path: Optional[Path] = None,
    duration: float = 30.0,
    start_time: float = 0.0,
    show: bool = True
):
    """
    Create a stacked comparison of raw vs preprocessed EEG as waveforms.
    
    Shows EEG traces as traditional waveforms (not heatmaps) for detailed inspection.
    Each channel is displayed as a separate trace, stacked vertically.
    
    Raw EEG (top): 32 channels at 250 Hz, 7500 samples (30 seconds)
    Preprocessed EEG (bottom): 30 channels at 200 Hz, 6000 samples (30 seconds)
    
    Args:
        subject_id: Subject number (1-80)
        stimulus_idx: Stimulus index (0-27)
        save_path: Optional path to save the figure
        duration: Duration of data to display (seconds), max 30s
        start_time: Start time in seconds
        show: Whether to show the plot
    """
    # Load data
    print(f"Loading data for subject {subject_id}, stimulus {stimulus_idx}...")
    raw_eeg, raw_sfreq, raw_channels = extract_raw_band(subject_id, stimulus_idx)
    prep_eeg, prep_sfreq, prep_channels = load_preprocessed(subject_id, stimulus_idx)
    
    print(f"  Raw: {len(raw_channels)} channels at {raw_sfreq} Hz ({raw_eeg.shape[1]} samples)")
    print(f"  Preprocessed: {len(prep_channels)} channels at {prep_sfreq} Hz ({prep_eeg.shape[1]} samples)")
    
    n_raw_channels = len(raw_channels)
    n_prep_channels = len(prep_channels)
    
    # Limit duration to available data
    max_duration_raw = raw_eeg.shape[1] / raw_sfreq
    max_duration_prep = prep_eeg.shape[1] / prep_sfreq
    duration = min(duration, max_duration_raw, max_duration_prep)
    
    # Create figure with two subplots (make raw subplot taller due to more channels)
    height_ratio_raw = n_raw_channels / n_prep_channels
    fig, axes = plt.subplots(2, 1, figsize=(20, 16), 
                              gridspec_kw={'height_ratios': [height_ratio_raw, 1]})
    
    # Calculate sample ranges
    raw_start_sample = int(start_time * raw_sfreq)
    raw_end_sample = int((start_time + duration) * raw_sfreq)
    prep_start_sample = int(start_time * prep_sfreq)
    prep_end_sample = int((start_time + duration) * prep_sfreq)
    
    # Extract data slices
    raw_data = raw_eeg[:, raw_start_sample:raw_end_sample]
    prep_data = prep_eeg[:, prep_start_sample:prep_end_sample]
    
    # Time axes
    raw_time = np.arange(raw_data.shape[1]) / raw_sfreq + start_time
    prep_time = np.arange(prep_data.shape[1]) / prep_sfreq + start_time
    
    # Calculate offsets for stacking channels
    # Scale each channel and add offset for visualization
    raw_std = np.std(raw_data)
    prep_std = np.std(prep_data)
    
    # Use consistent spacing between channels
    raw_spacing = raw_std * 4 if raw_std > 0 else 1
    prep_spacing = prep_std * 4 if prep_std > 0 else 1
    
    # Plot Raw EEG (top) - 32 channels at 250 Hz
    ax = axes[0]
    for i, ch_name in enumerate(raw_channels):
        offset = (n_raw_channels - 1 - i) * raw_spacing
        trace = raw_data[i, :] + offset
        ax.plot(raw_time, trace, 'b-', linewidth=0.5, alpha=0.8)
    
    # Set y-axis labels for raw
    ax.set_yticks([(n_raw_channels - 1 - i) * raw_spacing for i in range(n_raw_channels)])
    ax.set_yticklabels(raw_channels, fontsize=7)
    ax.set_xlim(raw_time[0], raw_time[-1])
    ax.set_ylim(-raw_spacing, n_raw_channels * raw_spacing)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Channels', fontsize=10)
    ax.set_title(
        f"Raw EEG - Subject {subject_id}, Stimulus {stimulus_idx}\n"
        f"Band: {BANDS[BROAD_BAND_INDEX]} (0.5-47 Hz) | {n_raw_channels} channels | {raw_sfreq} Hz | "
        f"{raw_data.shape[1]} samples | Duration: {duration:.1f}s",
        fontsize=12,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot Preprocessed EEG (bottom) - 30 channels at 200 Hz
    ax = axes[1]
    for i, ch_name in enumerate(prep_channels):
        offset = (n_prep_channels - 1 - i) * prep_spacing
        trace = prep_data[i, :] + offset
        ax.plot(prep_time, trace, 'r-', linewidth=0.5, alpha=0.8)
    
    # Set y-axis labels for preprocessed
    ax.set_yticks([(n_prep_channels - 1 - i) * prep_spacing for i in range(n_prep_channels)])
    ax.set_yticklabels(prep_channels, fontsize=7)
    ax.set_xlim(prep_time[0], prep_time[-1])
    ax.set_ylim(-prep_spacing, n_prep_channels * prep_spacing)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Channels', fontsize=10)
    ax.set_title(
        f"Preprocessed EEG - Subject {subject_id}, Stimulus {stimulus_idx}\n"
        f"{n_prep_channels} channels (A1, A2 removed) | {prep_sfreq} Hz | "
        f"{prep_data.shape[1]} samples | Z-normalized | Clipped at ±15 SD",
        fontsize=12,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Main title
    fig.suptitle(
        f"THU-EP EEG Comparison: Raw vs Preprocessed\n"
        f"Subject {subject_id} | Stimulus {stimulus_idx}",
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


class InteractiveEEGViewer:
    """
    Interactive EEG visualization with sliders to change subject and stimulus.
    
    Usage:
        viewer = InteractiveEEGViewer()
        viewer.show()
    """
    
    def __init__(self, initial_subject: int = 1, initial_stimulus: int = 0):
        self.subject_id = initial_subject
        self.stimulus_idx = initial_stimulus
        self.duration = 30.0
        
        # Create figure with extra space for sliders
        self.fig = plt.figure(figsize=(20, 18))
        
        # Create grid for plots and controls
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.1, 1, 0.15], hspace=0.25)
        
        # Create axes for EEG plots
        self.ax_raw = self.fig.add_subplot(gs[0])
        self.ax_prep = self.fig.add_subplot(gs[1])
        
        # Create axes for sliders
        gs_controls = gs[2].subgridspec(2, 3, hspace=0.5, wspace=0.3)
        
        ax_subject_slider = self.fig.add_subplot(gs_controls[0, :2])
        ax_stimulus_slider = self.fig.add_subplot(gs_controls[1, :2])
        ax_prev_btn = self.fig.add_subplot(gs_controls[0, 2])
        ax_next_btn = self.fig.add_subplot(gs_controls[1, 2])
        
        # Create sliders
        self.slider_subject = Slider(
            ax_subject_slider, 'Subject', 1, 80, 
            valinit=self.subject_id, valstep=1,
            color='steelblue'
        )
        self.slider_stimulus = Slider(
            ax_stimulus_slider, 'Stimulus', 0, 27, 
            valinit=self.stimulus_idx, valstep=1,
            color='coral'
        )
        
        # Create navigation buttons
        self.btn_prev = Button(ax_prev_btn, '← Prev Subject')
        self.btn_next = Button(ax_next_btn, 'Next Subject →')
        
        # Connect callbacks
        self.slider_subject.on_changed(self._on_subject_change)
        self.slider_stimulus.on_changed(self._on_stimulus_change)
        self.btn_prev.on_clicked(self._on_prev_subject)
        self.btn_next.on_clicked(self._on_next_subject)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial plot
        self._update_plot()
        
        # Instructions
        self.fig.text(
            0.5, 0.01, 
            'Keyboard: ←/→ = change stimulus | ↑/↓ = change subject | q = quit',
            ha='center', fontsize=10, style='italic', color='gray'
        )
    
    def _load_data(self):
        """Load raw and preprocessed data for current subject/stimulus."""
        try:
            raw_eeg, raw_sfreq, raw_channels = extract_raw_band(
                self.subject_id, self.stimulus_idx
            )
            prep_eeg, prep_sfreq, prep_channels = load_preprocessed(
                self.subject_id, self.stimulus_idx
            )
            return raw_eeg, raw_sfreq, raw_channels, prep_eeg, prep_sfreq, prep_channels
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None, None, None, None, None, None
    
    def _update_plot(self):
        """Redraw the EEG plots with current subject/stimulus."""
        # Clear axes
        self.ax_raw.clear()
        self.ax_prep.clear()
        
        # Load data
        result = self._load_data()
        if result[0] is None:
            self.ax_raw.text(0.5, 0.5, f'Error loading subject {self.subject_id}',
                           ha='center', va='center', fontsize=14, color='red')
            self.ax_prep.text(0.5, 0.5, 'Data not available',
                            ha='center', va='center', fontsize=14, color='red')
            self.fig.canvas.draw_idle()
            return
        
        raw_eeg, raw_sfreq, raw_channels, prep_eeg, prep_sfreq, prep_channels = result
        
        n_raw_channels = len(raw_channels)
        n_prep_channels = len(prep_channels)
        
        # Time axes
        raw_time = np.arange(raw_eeg.shape[1]) / raw_sfreq
        prep_time = np.arange(prep_eeg.shape[1]) / prep_sfreq
        
        # Calculate offsets for stacking
        raw_std = np.std(raw_eeg)
        prep_std = np.std(prep_eeg)
        raw_spacing = raw_std * 4 if raw_std > 0 else 1
        prep_spacing = prep_std * 4 if prep_std > 0 else 1
        
        # Plot Raw EEG (top)
        for i, ch_name in enumerate(raw_channels):
            offset = (n_raw_channels - 1 - i) * raw_spacing
            trace = raw_eeg[i, :] + offset
            self.ax_raw.plot(raw_time, trace, 'b-', linewidth=0.5, alpha=0.8)
        
        self.ax_raw.set_yticks([(n_raw_channels - 1 - i) * raw_spacing 
                                for i in range(n_raw_channels)])
        self.ax_raw.set_yticklabels(raw_channels, fontsize=7)
        self.ax_raw.set_xlim(0, raw_time[-1])
        self.ax_raw.set_ylim(-raw_spacing, n_raw_channels * raw_spacing)
        self.ax_raw.set_xlabel('Time (s)', fontsize=10)
        self.ax_raw.set_ylabel('Channels', fontsize=10)
        self.ax_raw.set_title(
            f"Raw EEG - Subject {self.subject_id}, Stimulus {self.stimulus_idx}\n"
            f"Band: broad-band (0.5-47 Hz) | {n_raw_channels} ch | {raw_sfreq} Hz | "
            f"{raw_eeg.shape[1]} samples",
            fontsize=11, fontweight='bold'
        )
        self.ax_raw.grid(True, alpha=0.3, axis='x')
        
        # Plot Preprocessed EEG (bottom)
        for i, ch_name in enumerate(prep_channels):
            offset = (n_prep_channels - 1 - i) * prep_spacing
            trace = prep_eeg[i, :] + offset
            self.ax_prep.plot(prep_time, trace, 'r-', linewidth=0.5, alpha=0.8)
        
        self.ax_prep.set_yticks([(n_prep_channels - 1 - i) * prep_spacing 
                                 for i in range(n_prep_channels)])
        self.ax_prep.set_yticklabels(prep_channels, fontsize=7)
        self.ax_prep.set_xlim(0, prep_time[-1])
        self.ax_prep.set_ylim(-prep_spacing, n_prep_channels * prep_spacing)
        self.ax_prep.set_xlabel('Time (s)', fontsize=10)
        self.ax_prep.set_ylabel('Channels', fontsize=10)
        self.ax_prep.set_title(
            f"Preprocessed EEG - Subject {self.subject_id}, Stimulus {self.stimulus_idx}\n"
            f"{n_prep_channels} ch (A1, A2 removed) | {prep_sfreq} Hz | "
            f"{prep_eeg.shape[1]} samples | Z-norm | ±15 SD clip",
            fontsize=11, fontweight='bold'
        )
        self.ax_prep.grid(True, alpha=0.3, axis='x')
        
        # Update main title
        self.fig.suptitle(
            f"THU-EP EEG Comparison: Raw vs Preprocessed",
            fontsize=14, fontweight='bold'
        )
        
        # Redraw
        self.fig.canvas.draw_idle()
        print(f"Displaying: Subject {self.subject_id}, Stimulus {self.stimulus_idx}")
    
    def _on_subject_change(self, val):
        self.subject_id = int(val)
        self._update_plot()
    
    def _on_stimulus_change(self, val):
        self.stimulus_idx = int(val)
        self._update_plot()
    
    def _on_prev_subject(self, event):
        if self.subject_id > 1:
            self.subject_id -= 1
            self.slider_subject.set_val(self.subject_id)
    
    def _on_next_subject(self, event):
        if self.subject_id < 80:
            self.subject_id += 1
            self.slider_subject.set_val(self.subject_id)
    
    def _on_key_press(self, event):
        if event.key == 'right':
            if self.stimulus_idx < 27:
                self.stimulus_idx += 1
                self.slider_stimulus.set_val(self.stimulus_idx)
        elif event.key == 'left':
            if self.stimulus_idx > 0:
                self.stimulus_idx -= 1
                self.slider_stimulus.set_val(self.stimulus_idx)
        elif event.key == 'up':
            if self.subject_id < 80:
                self.subject_id += 1
                self.slider_subject.set_val(self.subject_id)
        elif event.key == 'down':
            if self.subject_id > 1:
                self.subject_id -= 1
                self.slider_subject.set_val(self.subject_id)
        elif event.key == 'q':
            plt.close(self.fig)
    
    def show(self):
        """Display the interactive viewer."""
        plt.show()


def launch_interactive_viewer(subject: int = 1, stimulus: int = 0):
    """
    Launch the interactive EEG viewer with sliders.
    
    Args:
        subject: Initial subject ID (1-80)
        stimulus: Initial stimulus index (0-27)
    """
    print("Launching interactive EEG viewer...")
    print("Controls:")
    print("  - Sliders: Change subject (1-80) and stimulus (0-27)")
    print("  - Keyboard: ←/→ = stimulus, ↑/↓ = subject, q = quit")
    print()
    
    viewer = InteractiveEEGViewer(initial_subject=subject, initial_stimulus=stimulus)
    viewer.show()


def plot_mne_interactive(
    subject_id: int,
    stimulus_idx: int,
    duration: float = 10.0,
    scalings: str = 'auto'
):
    """
    Open MNE's interactive browser for raw and preprocessed EEG.
    
    This opens two separate MNE interactive windows for detailed inspection.
    
    Args:
        subject_id: Subject number (1-80)
        stimulus_idx: Stimulus index (0-27)
        duration: Duration visible in browser (seconds)
        scalings: Scaling mode ('auto' or dict)
    """
    print(f"Loading data for subject {subject_id}, stimulus {stimulus_idx}...")
    raw_eeg, raw_sfreq, raw_channels = extract_raw_band(subject_id, stimulus_idx)
    prep_eeg, prep_sfreq, prep_channels = load_preprocessed(subject_id, stimulus_idx)
    
    print(f"  Raw: {len(raw_channels)} channels at {raw_sfreq} Hz")
    print(f"  Preprocessed: {len(prep_channels)} channels at {prep_sfreq} Hz")
    
    # Create MNE Raw objects
    raw_info = create_mne_info(raw_channels, raw_sfreq)
    prep_info = create_mne_info(prep_channels, prep_sfreq)
    
    raw_mne = mne.io.RawArray(raw_eeg, raw_info, verbose=False)
    prep_mne = mne.io.RawArray(prep_eeg, prep_info, verbose=False)
    
    print("\nOpening MNE interactive browser...")
    print("  - Window 1: Raw EEG (32 channels, 250 Hz)")
    print("  - Window 2: Preprocessed EEG (30 channels, 200 Hz)")
    print("Use arrow keys to navigate, +/- to scale, ? for help\n")
    
    # Plot both
    raw_mne.plot(
        duration=duration,
        scalings=scalings,
        title=f"RAW - Subject {subject_id}, Stimulus {stimulus_idx} | Band: {BANDS[BROAD_BAND_INDEX]} | 32ch @ 250Hz",
        show=False,
        block=False
    )
    
    prep_mne.plot(
        duration=duration,
        scalings=scalings,
        title=f"PREPROCESSED - Subject {subject_id}, Stimulus {stimulus_idx} | 30ch @ 200Hz",
        show=True,
        block=True
    )


def interactive_mode():
    """
    Interactive mode for selecting subject and stimulus to visualize.
    """
    print("\n" + "=" * 70)
    print("THU-EP EEG Visualization: Interactive Mode")
    print("=" * 70)
    print("Subjects: 1-80")
    print("Stimuli: 0-27 (28 different film clips)")
    print()
    
    while True:
        try:
            subject_id = int(input("Enter subject ID (1-80) or 0 to exit: "))
            if subject_id == 0:
                break
            if subject_id < 1 or subject_id > 80:
                print("Invalid subject ID. Please enter a number between 1 and 80.")
                continue
            
            stimulus_idx = int(input(
                f"Enter stimulus index (0-27) for subject {subject_id}: "
            ))
            if stimulus_idx < 0 or stimulus_idx > 27:
                print("Invalid stimulus index. Please enter a number between 0 and 27.")
                continue
            
            print("\nGenerating visualization...")
            plot_comparison_simple(
                subject_id=subject_id,
                stimulus_idx=stimulus_idx,
                show=True
            )
            
        except ValueError:
            print("Invalid input. Please enter a number.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize raw vs preprocessed THU-EP EEG segments as waveforms"
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Subject ID (1-80), default: 1"
    )
    parser.add_argument(
        "--stimulus",
        type=int,
        default=0,
        help="Stimulus index (0-27), default: 0"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch interactive viewer with sliders (recommended)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for selecting subject and stimulus via terminal"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export figure to specified path (e.g., outputs/comparison.png)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of data to display in seconds (default: 30, max: 30)"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0)"
    )
    parser.add_argument(
        "--mne-browser",
        action="store_true",
        help="Open MNE's interactive browser for detailed inspection"
    )
    
    args = parser.parse_args()
    
    # Interactive viewer with sliders (recommended)
    if args.viewer:
        launch_interactive_viewer(subject=args.subject, stimulus=args.stimulus)
        return
    
    # Terminal-based interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Validate ranges
    if args.subject < 1 or args.subject > 80:
        print(f"Error: Subject must be between 1 and 80, got {args.subject}")
        sys.exit(1)
    
    if args.stimulus < 0 or args.stimulus > 27:
        print(f"Error: Stimulus must be between 0 and 27, got {args.stimulus}")
        sys.exit(1)
    
    # Generate visualization
    save_path = Path(args.export) if args.export else None
    
    try:
        if args.mne_browser:
            # Use MNE's interactive browser
            plot_mne_interactive(
                subject_id=args.subject,
                stimulus_idx=args.stimulus,
                duration=args.duration
            )
        else:
            # Use static matplotlib plot with waveforms
            plot_comparison_simple(
                subject_id=args.subject,
                stimulus_idx=args.stimulus,
                save_path=save_path,
                duration=args.duration,
                start_time=args.start,
                show=True
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
