"""Individual preprocessing steps for SEED EEG pipeline."""

import numpy as np
import mne
from pathlib import Path
from typing import Optional, Tuple


def downsample_buffer(buffer: np.ndarray, original_sfreq: float, target_sfreq: float, 
                     verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Downsample the entire buffer.
    
    Args:
        buffer: (n_channels, n_samples) data
        original_sfreq: Original sampling rate in Hz
        target_sfreq: Target sampling rate in Hz
        verbose: Whether to print progress
        
    Returns:
        Tuple of (downsampled_buffer, new_sfreq)
    """
    down_factor = int(original_sfreq / target_sfreq)
    if verbose:
        print(f"  Downsampling: {original_sfreq} Hz -> {target_sfreq} Hz (factor 1:{down_factor})...")
    
    buffer_resampled = mne.filter.resample(
        buffer,
        up=1,
        down=down_factor,
        verbose=False
    )
    
    if verbose:
        print(f"  Downsampled to {buffer_resampled.shape[1]:,} samples")
    
    return buffer_resampled, target_sfreq


def apply_bandpass_filter(buffer: np.ndarray, sfreq: float, lowcut_hz: float, 
                         highcut_hz: float, verbose: bool = False) -> np.ndarray:
    """
    Apply bandpass filter to the entire buffer.
    
    Args:
        buffer: (n_channels, n_samples) data
        sfreq: Sampling rate in Hz
        lowcut_hz: Lower cutoff frequency in Hz
        highcut_hz: Upper cutoff frequency in Hz
        verbose: Whether to print progress
        
    Returns:
        Filtered buffer
    """
    if verbose:
        print(f"  Applying bandpass filter: {lowcut_hz}-{highcut_hz} Hz...")
    
    buffer_filtered = mne.filter.filter_data(
        buffer,
        sfreq=sfreq,
        l_freq=lowcut_hz,
        h_freq=highcut_hz,
        verbose=False
    )
    
    if verbose:
        print(f"  Filtered")
    
    return buffer_filtered


def compute_session_statistics(buffer: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute session-level statistics for Z-normalization.
    
    Args:
        buffer: (n_channels, n_samples) data
        verbose: Whether to print progress
        
    Returns:
        Tuple of (session_mean, session_std)
    """
    session_mean = np.mean(buffer, axis=1, keepdims=True)
    session_std = np.std(buffer, axis=1, keepdims=True)
    
    if verbose:
        mean_range = (session_mean.min(), session_mean.max())
        std_range = (session_std.min(), session_std.max())
        print(f"  Computed session-level statistics: μ∈[{mean_range[0]:.4f}, {mean_range[1]:.4f}], σ∈[{std_range[0]:.4f}, {std_range[1]:.4f}]")
    
    return session_mean, session_std


def extract_last_30s(window_data: np.ndarray, sfreq: float, window_duration_sec: float,
                    verbose: bool = False) -> Optional[np.ndarray]:
    """
    Extract last N seconds from window.
    
    Args:
        window_data: (n_channels, n_samples) data
        sfreq: Sampling rate in Hz
        window_duration_sec: Duration to extract in seconds
        verbose: Whether to print warnings
        
    Returns:
        Extracted window or None if too short
    """
    n_samples_needed = int(window_duration_sec * sfreq)
    n_samples_available = window_data.shape[1]
    
    if n_samples_available < n_samples_needed:
        if verbose:
            print(f"    Window too short: {n_samples_available/sfreq:.1f}s < {window_duration_sec}s")
        return None
    
    # Extract last N seconds
    start_idx = n_samples_available - n_samples_needed
    window_extracted = window_data[:, start_idx:]
    
    return window_extracted


def z_normalize_with_stats(window_data: np.ndarray, session_mean: np.ndarray, 
                          session_std: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization using session-level statistics.
    
    Formula: X_norm = (X - μ_session) / σ_session
    
    Args:
        window_data: (n_channels, n_samples) - single trial data
        session_mean: (n_channels, 1) - session-wide mean per channel
        session_std: (n_channels, 1) - session-wide std per channel
    
    Returns:
        Normalized window with shape (n_channels, n_samples)
    """
    # Avoid division by zero
    session_std_safe = np.copy(session_std)
    session_std_safe[session_std_safe == 0] = 1.0
    
    # Z-normalize using session statistics
    window_normalized = (window_data - session_mean) / session_std_safe
    
    return window_normalized


def artifact_clipping(window_data: np.ndarray, artifact_threshold_std: float,
                     verbose: bool = False) -> np.ndarray:
    """
    Clip artifacts exceeding threshold.
    
    Args:
        window_data: (n_channels, n_samples) data
        artifact_threshold_std: Clipping threshold in standard deviations
        verbose: Whether to print clipping statistics
        
    Returns:
        Clipped window
    """
    # Find and clip values exceeding threshold
    clipped_count = np.sum(np.abs(window_data) > artifact_threshold_std)
    
    window_clipped = np.clip(window_data, -artifact_threshold_std, artifact_threshold_std)
    
    if verbose and clipped_count > 0:
        pct = 100 * clipped_count / window_data.size
        print(f"    Clipped artifacts: {clipped_count} samples ({pct:.2f}%)")
    
    return window_clipped


def export_npy(subject_id: int, session_id: int, trial_idx: int, window_data: np.ndarray,
              preprocessed_output_dir: str, verbose: bool = False) -> str:
    """
    Export preprocessed data to NPY file in organized folder structure.
    
    Args:
        subject_id: Subject ID
        session_id: Session ID
        trial_idx: Trial index
        window_data: (n_channels, n_samples) data to export
        preprocessed_output_dir: Output directory path
        verbose: Whether to print export info
        
    Returns:
        Path to exported file
    """
    # Create nested directory structure: sub-XX/ses-Y/
    subject_dir = Path(preprocessed_output_dir) / f"sub-{subject_id:02d}"
    session_dir = subject_dir / f"ses-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"trial-{trial_idx:02d}.npy"
    output_path = session_dir / output_filename
    
    np.save(output_path, window_data.astype(np.float32))
    
    if verbose:
        shape_str = f"({window_data.shape[0]} channels, {window_data.shape[1]} samples)"
        print(f"    Exported: {output_filename} {shape_str}")
    
    return str(output_path)
