"""Individual preprocessing steps for THU-EP EEG pipeline.

These functions operate on THU-EP data with shape transformations:
- Original: (7500, 32, 28, 6) = (samples, channels, stimuli, bands)
- After band extraction: (28, 32, 7500) = (stimuli, channels, samples)
- After channel removal: (28, 30, 7500) = (stimuli, channels, samples)
- After resampling: (28, 30, 6000) = (stimuli, channels, samples)
- Final output: (28, 30, 6000) = (stimuli, channels, samples)
"""

import numpy as np
import mne
from typing import Tuple, List


def extract_frequency_band(data: np.ndarray, band_index: int, band_name: str = "",
                          verbose: bool = False) -> np.ndarray:
    """
    Extract a single frequency band from multi-band data and reshape to output format.
    
    Args:
        data: (n_samples, n_channels, n_stimuli, n_bands)
        band_index: Index of band to extract (0-5)
        band_name: Name of band for logging
        verbose: Whether to print progress
        
    Returns:
        Data with shape (n_stimuli, n_channels, n_samples)
    """
    extracted = data[:, :, :, band_index]  # (n_samples, n_channels, n_stimuli)
    # Reshape to (n_stimuli, n_channels, n_samples) for easier processing
    result = extracted.transpose(2, 1, 0)  # (n_stimuli, n_channels, n_samples)
    
    if verbose:
        print(f"  Extracted band {band_index} ({band_name}): {data.shape} -> {result.shape}")
    
    return result


def remove_reference_channels(data: np.ndarray, channels_to_remove_indices: List[int],
                              verbose: bool = False) -> np.ndarray:
    """
    Remove reference channels (A1, A2) from EEG data.
    
    Args:
        data: (n_stimuli, n_channels, n_samples)
        channels_to_remove_indices: List of channel indices to remove
        verbose: Whether to print progress
        
    Returns:
        Data with reference channels removed, shape (n_stimuli, n_channels_final, n_samples)
    """
    mask = np.ones(data.shape[1], dtype=bool)
    mask[channels_to_remove_indices] = False
    
    # Remove from channel axis (1)
    result = data[:, mask, :]
    
    if verbose:
        removed_count = len(channels_to_remove_indices)
        print(f"  Removed {removed_count} reference channels: {data.shape[1]} -> {result.shape[1]} channels")
    
    return result


def downsample_stimuli(data: np.ndarray, original_sfreq: float, target_sfreq: float,
                      verbose: bool = False) -> np.ndarray:
    """
    Downsample all stimuli from original to target sampling rate using MNE.
    
    Uses mne.filter.resample with polyphase FIR filtering on the last axis (samples).
    
    Args:
        data: (n_stimuli, n_channels, n_samples)
        original_sfreq: Original sampling frequency in Hz
        target_sfreq: Target sampling frequency in Hz
        verbose: Whether to print progress
        
    Returns:
        Resampled data with shape (n_stimuli, n_channels, n_samples_new)
    """
    
    n_samples_orig = data.shape[-1]
    
    # Calculate down factor for downsampling
    # Example: 250 Hz / 200 Hz = 1.25 -> down=1.25
    down = original_sfreq / target_sfreq
    
    # Resample along last axis (samples) using polyphase FIR method
    # Polyphase method provides better frequency characteristics than FFT for EEG
    result = mne.filter.resample(data, down=down, method='polyphase')
    
    if verbose:
        print(f"  Downsampled: {original_sfreq} Hz -> {target_sfreq} Hz "
              f"({n_samples_orig} -> {result.shape[-1]} samples) "
              f"using MNE polyphase resampling (down={down:.2f})")
    
    return result


def compute_global_statistics(data: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and standard deviation per channel across all stimuli and samples.
    
    Args:
        data: (n_stimuli, n_channels, n_samples)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (global_mean, global_std) with shapes (n_channels,)
    """
    n_stimuli, n_channels, n_samples = data.shape
    
    # Reshape to (n_channels, total_samples) for global statistics
    # Transpose to (n_channels, n_stimuli, n_samples) then flatten last two dims
    data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)
    
    global_mean = np.mean(data_reshaped, axis=1)  # (n_channels,)
    global_std = np.std(data_reshaped, axis=1)    # (n_channels,)
    
    if verbose:
        print(f"  Computed global statistics across {n_stimuli} stimuli "
              f"({n_samples * n_stimuli:,} samples/channel)")
        print(f"    μ range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
        print(f"    σ range: [{global_std.min():.4f}, {global_std.max():.4f}]")
    
    return global_mean, global_std


def z_normalize_global(data: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray,
                      verbose: bool = False) -> np.ndarray:
    """
    Apply global Z-score normalization per channel.
    
    Formula: X_norm = (X - μ_global) / σ_global
    
    Args:
        data: (n_stimuli, n_channels, n_samples)
        global_mean: (n_channels,) global mean per channel
        global_std: (n_channels,) global std per channel
        
    Returns:
        Normalized data with same shape
    """
    # Avoid division by zero
    std_safe = global_std.copy()
    std_safe[std_safe == 0] = 1.0
    
    # Reshape for broadcasting: (1, n_channels, 1)
    mean_broadcast = global_mean.reshape(1, -1, 1)
    std_broadcast = std_safe.reshape(1, -1, 1)
    
    result = (data - mean_broadcast) / std_broadcast
    
    if verbose:
        print(f"  Applied global Z-normalization")
    
    return result


def artifact_clipping(data: np.ndarray, threshold_std: float,
                     verbose: bool = False) -> np.ndarray:
    """
    Clip artifacts exceeding threshold standard deviations.
    
    Args:
        data: Any shape array (assumed already Z-normalized)
        threshold_std: Clipping threshold in standard deviations
        verbose: Whether to print clipping statistics
        
    Returns:
        Clipped data
    """
    # Count values exceeding threshold before clipping
    clipped_count = np.sum(np.abs(data) > threshold_std)
    
    result = np.clip(data, -threshold_std, threshold_std)
    
    if verbose and clipped_count > 0:
        pct = 100 * clipped_count / data.size
        print(f"  Clipped artifacts: {clipped_count:,} samples ({pct:.4f}%)")
    elif verbose:
        print(f"  No artifacts exceeded ±{threshold_std} stddev threshold")
    
    return result


def transpose_to_output_format(data: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Pass-through function - data is already in output format (n_stimuli, n_channels, n_samples).
    
    Args:
        data: (n_stimuli, n_channels, n_samples)
        verbose: Whether to print progress
        
    Returns:
        Same data, shape (n_stimuli, n_channels, n_samples)
    """
    if verbose:
        print(f"  Output format verified: {data.shape}")
    
    return data


def export_subject_npy(subject_id: int, data: np.ndarray, output_dir: str,
                      verbose: bool = False) -> str:
    """
    Export preprocessed subject data to NPY file.
    
    Args:
        subject_id: Subject ID (1-80)
        data: (n_stimuli, n_channels, n_samples) preprocessed data
        output_dir: Output directory path
        verbose: Whether to print export info
        
    Returns:
        Path to exported file
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"sub_{subject_id:02d}.npy"
    output_file = output_path / output_filename
    
    np.save(output_file, data.astype(np.float32))
    
    if verbose:
        shape_str = f"({data.shape[0]} stimuli, {data.shape[1]} channels, {data.shape[2]} samples)"
        print(f"  Exported: {output_filename} {shape_str}")
    
    return str(output_file)
