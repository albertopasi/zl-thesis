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
    uv run python -m src.thu_ep.exploration.visualize_raw_vs_preprocessed --viewer
    
    # Static plot for specific subject/stimulus
    uv run python -m src.thu_ep.exploration.visualize_raw_vs_preprocessed --subject 1 --stimulus 0
    
    # Export comparison images
    uv run python -m src.thu_ep.exploration.visualize_raw_vs_preprocessed --subject 1 --stimulus 0 --export outputs/comparison.png

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
from matplotlib.widgets import Slider, Button

try:
    import mne
except ImportError:
    print("MNE not installed.")
    sys.exit(1)

from ..config import get_config

# Load configuration
_cfg = get_config()

# Configuration (from config)
DATA_DIR = _cfg.raw_data_dir.parent
RAW_DATA_DIR = _cfg.raw_data_dir
PREPROCESSED_DATA_DIR = _cfg.preprocessed_dir

# Channel configuration (from config)
ALL_CHANNELS = _cfg.all_channels
CHANNELS_TO_REMOVE = _cfg.channels_to_remove
FINAL_CHANNELS = _cfg.final_channels

# Sampling rates (from config)
ORIGINAL_SFREQ = _cfg.original_sfreq
TARGET_SFREQ = _cfg.target_sfreq

# Band configuration (from config)
BANDS = _cfg.band_names
BROAD_BAND_INDEX = _cfg.broad_band_index


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
    Open MNE's interactive browser for raw vs preprocessed EEG comparison.

    Opens two separate MNE `.plot()` windows with proper amplitude display,
    so that raw (microvolt-scale) and preprocessed (z-normalized) data are
    shown at their true scales.

    Args:
        subject_id: Subject number (1-80)
        stimulus_idx: Stimulus index (0-27)
        save_path: Optional path to save the figures (creates *_raw and *_preprocessed variants)
        duration: Duration of data to display (seconds), max 30s
        start_time: Start time in seconds
        show: Whether to show the plots
    """
    # Load data
    print(f"Loading data for subject {subject_id}, stimulus {stimulus_idx}...")
    raw_eeg, raw_sfreq, raw_channels = extract_raw_band(subject_id, stimulus_idx)
    prep_eeg, prep_sfreq, prep_channels = load_preprocessed(subject_id, stimulus_idx)

    # Print amplitude statistics
    raw_std = np.std(raw_eeg)
    prep_std = np.std(prep_eeg)
    print(f"  Raw:  std={raw_std:.2f} | range=[{raw_eeg.min():.1f}, {raw_eeg.max():.1f}]")
    print(f"  Prep: std={prep_std:.4f} | range=[{prep_eeg.min():.3f}, {prep_eeg.max():.3f}]")

    # Limit duration to available data
    max_duration_raw = raw_eeg.shape[1] / raw_sfreq
    max_duration_prep = prep_eeg.shape[1] / prep_sfreq
    duration = min(duration, max_duration_raw, max_duration_prep)

    # Create MNE RawArray objects
    raw_info = create_mne_info(raw_channels, raw_sfreq)
    prep_info = create_mne_info(prep_channels, prep_sfreq)
    raw_mne = mne.io.RawArray(raw_eeg, raw_info, verbose=False)
    prep_mne = mne.io.RawArray(prep_eeg, prep_info, verbose=False)

    # Open MNE interactive browsers
    fig_raw = raw_mne.plot(
        duration=duration,
        start=start_time,
        n_channels=len(raw_channels),
        scalings='auto',
        title=f"RAW - Subject {subject_id}, Stimulus {stimulus_idx} | "
              f"{BANDS[BROAD_BAND_INDEX]} | {len(raw_channels)}ch @ {raw_sfreq}Hz",
        show=False,
        block=False,
    )
    fig_prep = prep_mne.plot(
        duration=duration,
        start=start_time,
        n_channels=len(prep_channels),
        scalings='auto',
        title=f"PREPROCESSED - Subject {subject_id}, Stimulus {stimulus_idx} | "
              f"{len(prep_channels)}ch @ {prep_sfreq}Hz | Z-norm | ±15 SD clip",
        show=False,
        block=False,
    )

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        raw_save = save_path.with_name(f"{save_path.stem}_raw{save_path.suffix}")
        prep_save = save_path.with_name(f"{save_path.stem}_preprocessed{save_path.suffix}")
        fig_raw.savefig(raw_save, dpi=150, bbox_inches='tight')
        fig_prep.savefig(prep_save, dpi=150, bbox_inches='tight')
        print(f"Saved figures to: {raw_save} and {prep_save}")

    if show:
        plt.show()

    return fig_raw, fig_prep


class InteractiveEEGViewer:
    """
    Interactive EEG viewer using MNE's browser for proper amplitude display.

    A small control panel with sliders and buttons opens alongside two MNE
    `.plot()` windows (raw and preprocessed).  Sliders do NOT auto-update;
    press "Update Plot" or use keyboard shortcuts to load new data.

    Usage:
        viewer = InteractiveEEGViewer()
        viewer.show()
    """

    def __init__(self, initial_subject: int = 1, initial_stimulus: int = 0):
        self.subject_id = initial_subject
        self.stimulus_idx = initial_stimulus
        self.duration = 30.0

        # MNE browser figures (managed separately)
        self.raw_fig = None
        self.prep_fig = None

        # --- Control panel (small separate figure) ---
        self.ctrl_fig = plt.figure('EEG Viewer Controls', figsize=(10, 2.5))
        gs = self.ctrl_fig.add_gridspec(3, 4, hspace=0.8, wspace=0.3,
                                        left=0.08, right=0.95,
                                        top=0.88, bottom=0.18)

        # Sliders
        ax_subject = self.ctrl_fig.add_subplot(gs[0, :3])
        ax_stimulus = self.ctrl_fig.add_subplot(gs[1, :3])

        self.slider_subject = Slider(
            ax_subject, 'Subject', 1, 80,
            valinit=self.subject_id, valstep=1, color='steelblue'
        )
        self.slider_stimulus = Slider(
            ax_stimulus, 'Stimulus', 0, 27,
            valinit=self.stimulus_idx, valstep=1, color='coral'
        )

        # Buttons
        ax_prev = self.ctrl_fig.add_subplot(gs[0, 3])
        ax_next = self.ctrl_fig.add_subplot(gs[1, 3])
        ax_update = self.ctrl_fig.add_subplot(gs[2, 1:3])

        self.btn_prev = Button(ax_prev, '\u2190 Prev Subj')
        self.btn_next = Button(ax_next, 'Next Subj \u2192')
        self.btn_update = Button(ax_update, 'Update Plot')

        # Callbacks (sliders have NO on_changed -- only button / keyboard)
        self.btn_prev.on_clicked(self._on_prev_subject)
        self.btn_next.on_clicked(self._on_next_subject)
        self.btn_update.on_clicked(lambda _: self._on_update())

        # Keyboard
        self.ctrl_fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        # Footer instructions
        self.ctrl_fig.text(
            0.5, 0.02,
            'Focus this window | Keys: \u2190/\u2192 stimulus | '
            '\u2191/\u2193 subject | Enter update | q quit',
            ha='center', fontsize=9, style='italic', color='gray'
        )

        # Open initial MNE plots
        self._open_mne_plots()

    # ------------------------------------------------------------------ #
    #  MNE browser management
    # ------------------------------------------------------------------ #
    def _close_mne_plots(self):
        """Close existing MNE browser figures if they exist."""
        if self.raw_fig is not None:
            plt.close(self.raw_fig)
            self.raw_fig = None
        if self.prep_fig is not None:
            plt.close(self.prep_fig)
            self.prep_fig = None

    def _open_mne_plots(self):
        """Close old MNE figures, load data, and open new MNE browsers."""
        self._close_mne_plots()

        try:
            raw_eeg, raw_sfreq, raw_channels = extract_raw_band(
                self.subject_id, self.stimulus_idx
            )
            prep_eeg, prep_sfreq, prep_channels = load_preprocessed(
                self.subject_id, self.stimulus_idx
            )
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return

        # Amplitude statistics
        raw_std = np.std(raw_eeg)
        prep_std = np.std(prep_eeg)
        print(
            f"Subject {self.subject_id}, Stimulus {self.stimulus_idx}:\n"
            f"  Raw:  std={raw_std:.2f} | range=[{raw_eeg.min():.1f}, {raw_eeg.max():.1f}]\n"
            f"  Prep: std={prep_std:.4f} | range=[{prep_eeg.min():.3f}, {prep_eeg.max():.3f}]"
        )

        # Create MNE objects
        raw_mne = mne.io.RawArray(
            raw_eeg, create_mne_info(raw_channels, raw_sfreq), verbose=False
        )
        prep_mne = mne.io.RawArray(
            prep_eeg, create_mne_info(prep_channels, prep_sfreq), verbose=False
        )

        # Open MNE browsers
        self.raw_fig = raw_mne.plot(
            duration=self.duration,
            n_channels=len(raw_channels),
            scalings='auto',
            title=f"RAW - Subject {self.subject_id}, Stimulus {self.stimulus_idx} | "
                  f"{BANDS[BROAD_BAND_INDEX]} | {len(raw_channels)}ch @ {raw_sfreq}Hz",
            show=False,
            block=False,
        )
        self.prep_fig = prep_mne.plot(
            duration=self.duration,
            n_channels=len(prep_channels),
            scalings='auto',
            title=f"PREPROCESSED - Subject {self.subject_id}, Stimulus {self.stimulus_idx} | "
                  f"{len(prep_channels)}ch @ {prep_sfreq}Hz | Z-norm | \u00b115 SD clip",
            show=False,
            block=False,
        )

        # Show the new figures without blocking
        self.raw_fig.show()
        self.prep_fig.show()

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #
    def _on_update(self):
        """Read slider values and refresh MNE plots."""
        self.subject_id = int(self.slider_subject.val)
        self.stimulus_idx = int(self.slider_stimulus.val)
        self._open_mne_plots()

    def _on_prev_subject(self, event):
        if self.subject_id > 1:
            self.subject_id -= 1
            self.slider_subject.set_val(self.subject_id)
            self._open_mne_plots()

    def _on_next_subject(self, event):
        if self.subject_id < 80:
            self.subject_id += 1
            self.slider_subject.set_val(self.subject_id)
            self._open_mne_plots()

    def _on_key_press(self, event):
        if event.key == 'right':
            if self.stimulus_idx < 27:
                self.stimulus_idx += 1
                self.slider_stimulus.set_val(self.stimulus_idx)
                self._open_mne_plots()
        elif event.key == 'left':
            if self.stimulus_idx > 0:
                self.stimulus_idx -= 1
                self.slider_stimulus.set_val(self.stimulus_idx)
                self._open_mne_plots()
        elif event.key == 'up':
            if self.subject_id < 80:
                self.subject_id += 1
                self.slider_subject.set_val(self.subject_id)
                self._open_mne_plots()
        elif event.key == 'down':
            if self.subject_id > 1:
                self.subject_id -= 1
                self.slider_subject.set_val(self.subject_id)
                self._open_mne_plots()
        elif event.key == 'enter':
            self._on_update()
        elif event.key == 'q':
            self._close_mne_plots()
            plt.close(self.ctrl_fig)

    def show(self):
        """Display the interactive viewer (blocks until all windows closed)."""
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
