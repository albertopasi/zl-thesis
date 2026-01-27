"""SEED EEG preprocessing pipeline."""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import re
import warnings

warnings.filterwarnings('ignore')

from .preprocessing_config import SEEDPreprocessingConfig
from .seed_loader import SEEDEEGLoader


class SEEDPreprocessingPipeline:
    """SEED EEG preprocessing pipeline."""
    
    def __init__(self, config: SEEDPreprocessingConfig):
        """
        Initialize SEED preprocessing pipeline.
        
        Args:
            config: SEEDPreprocessingConfig instance
        """
        self.config = config
        self.loader = SEEDEEGLoader(
            seed_raw_dir=config.seed_raw_dir,
            montage_path=config.montage_file
        )
        
        # Load time markers for window extraction
        self.start_points, self.end_points = self._load_time_markers()
        
        # Create output directory
        Path(self.config.preprocessed_output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.config.verbose:
            print("Preprocessing pipeline initialized")
            self._print_config()
    
    def _print_config(self):
        """Print current configuration."""
        print("\nActive preprocessing steps:")
        for step_name, enabled in self.config.steps_enabled.items():
            status = "✓" if enabled else "✗"
            print(f"  {status} {step_name}")
        
        print(f"\nConfiguration parameters:")
        print(f"  Filter: {self.config.filter_lowcut_hz}-{self.config.filter_highcut_hz} Hz")
        print(f"  Downsample: {self.config.downsample_freq_hz} Hz")
        print(f"  Window duration: {self.config.window_duration_sec} seconds")
        print(f"  Artifact threshold: {self.config.artifact_threshold_std} std")
        print()
    
    def _load_time_markers(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Load time markers from time.txt file."""
        time_file = Path(self.config.time_markers_file)
        
        if not time_file.exists():
            if self.config.verbose:
                print(f"Time markers file not found: {time_file}")
            return None, None
        
        try:
            with open(time_file, 'r') as f:
                content = f.read()
            
            start_match = re.search(r'start_point_list\s*=\s*\[(.*?)\]', content)
            end_match = re.search(r'end_point_list\s*=\s*\[(.*?)\]', content)
            
            if start_match and end_match:
                start_points = [int(x.strip()) for x in start_match.group(1).split(',')]
                end_points = [int(x.strip()) for x in end_match.group(1).split(',')]
                
                if self.config.verbose:
                    print(f"Loaded time markers: {len(start_points)} windows")
                
                return start_points, end_points
        except Exception as e:
            if self.config.verbose:
                print(f"Error loading time markers: {e}")
        
        return None, None
    
    def process_subject_session(self, subject_id: int, session_id: int) -> Dict:
        """
        Process a single subject-session through the pipeline.
        
        Args:
            subject_id: Subject ID
            session_id: Session ID
            
        Returns:
            Dictionary with processing results and metadata
        """
        results = {
            'subject_id': subject_id,
            'session_id': session_id,
            'success': False,
            'error': None,
            'num_trials': 0,
            'output_files': []
        }
        
        try:
            # Load raw data (header-independent)
            if not self.config.is_step_enabled('load_raw'):
                if self.config.verbose:
                    print(f"Load raw: skipped")
                return results
            
            if self.config.verbose:
                print(f"\nProcessing Subject {subject_id:02d}, Session {session_id}...")
            
            # Header-Independent Lazy Loading
            raw = self.loader.load_raw(subject_id, session_id)
            original_sfreq = raw.info['sfreq']
            
            # Drop non-EEG channels immediately (reduces data width)
            if self.config.is_step_enabled('drop_non_eeg'):
                raw.drop_channels(self.config.non_eeg_channels, on_missing='ignore')
                if self.config.verbose:
                    print(f"  ✓ Dropped non-EEG channels: {self.config.non_eeg_channels}")
            
            # Construct Manual Buffer
            # Extract individual trials using time.txt indices
            if self.start_points is None or self.end_points is None:
                if self.config.verbose:
                    print(" Cannot construct manual buffer: time markers not available")
                return results
            
            trial_list = []
            trial_boundaries = []  # Track where each trial starts in the concatenated buffer
            buffer_position = 0
            
            for trial_idx, (start_sample, end_sample) in enumerate(
                zip(self.start_points, self.end_points), 1
            ):
                try:
                    # Extract window using raw.get_data() with lazy loading
                    trial_data = raw.get_data(start=start_sample, stop=end_sample)
                    trial_list.append(trial_data)
                    trial_boundaries.append((buffer_position, buffer_position + trial_data.shape[1]))
                    buffer_position += trial_data.shape[1]
                except Exception as e:
                    if self.config.verbose:
                        print(f"    Could not extract trial {trial_idx}: {e}")
                    return results
            
            # Concatenate trials into a single contiguous buffer
            buffer = np.concatenate(trial_list, axis=1)
            if self.config.verbose:
                print(f"  Constructed manual buffer: {buffer.shape[0]} channels × {buffer.shape[1]} samples "
                      f"({buffer.shape[1]/original_sfreq/60:.1f} minutes at {original_sfreq} Hz)")
            
            # Signal Conditioning (Continuous Logic)
            
            # Resample the entire buffer (continuous)
            if self.config.is_step_enabled('downsample'):
                # Downsample: up=1, down=original_sfreq/target_sfreq
                down_factor = int(original_sfreq / self.config.downsample_freq_hz)
                print(f"  Downsampling: {original_sfreq} Hz -> {self.config.downsample_freq_hz} Hz (factor 1:{down_factor})...")
                buffer = mne.filter.resample(
                    buffer,
                    up=1,
                    down=down_factor,
                    verbose=False
                )
                current_sfreq = self.config.downsample_freq_hz
                print(f"  Downsampled to {buffer.shape[1]:,} samples")
            else:
                current_sfreq = original_sfreq

            # Apply bandpass filter to the entire buffer (continuous)
            if self.config.is_step_enabled('bandpass_filter'):
                print(f"  Applying bandpass filter: {self.config.filter_lowcut_hz}-{self.config.filter_highcut_hz} Hz...")
                buffer = mne.filter.filter_data(
                    buffer,
                    sfreq=current_sfreq,
                    l_freq=self.config.filter_lowcut_hz,
                    h_freq=self.config.filter_highcut_hz,
                    verbose=False
                )
                print(f"  Filtered")
            
            # Calculate Session-Level Statistics
            # Compute global mean and std from entire resampled buffer for Z-normalization
            session_mean = np.mean(buffer, axis=1, keepdims=True)
            session_std = np.std(buffer, axis=1, keepdims=True)
            
            if self.config.verbose:
                # Show min/max of per-channel statistics for visibility
                mean_range = (session_mean.min(), session_mean.max())
                std_range = (session_std.min(), session_std.max())
                print(f"  Computed session-level statistics: μ∈[{mean_range[0]:.4f}, {mean_range[1]:.4f}], σ∈[{std_range[0]:.4f}, {std_range[1]:.4f}]")
            
            # Re-slice Processed Buffer into Trials
            # Adjust trial boundaries for the downsampled buffer
            downsample_ratio = current_sfreq / original_sfreq
            windows_data = []
            
            for trial_idx, (start_idx, end_idx) in enumerate(trial_boundaries, 1):
                # Scale indices to match resampled buffer
                start_new = int(start_idx * downsample_ratio)
                end_new = int(end_idx * downsample_ratio)
                
                try:
                    window = buffer[:, start_new:end_new]
                    windows_data.append((f'trial_{trial_idx:02d}', window, session_mean, session_std))
                except Exception as e:
                    if self.config.verbose:
                        print(f"    Could not re-slice trial {trial_idx}: {e}")
                    continue
            
            if self.config.verbose:
                print(f"  Re-sliced buffer into {len(windows_data)} trials at {current_sfreq} Hz")
            
            # Final Windowing & Normalization
            for trial_idx, (trial_name, window_data, sess_mean, sess_std) in enumerate(windows_data, 1):
                
                if window_data is None:
                    continue
                
                # Extract last 30 seconds (capture stabilized emotional state)
                if self.config.is_step_enabled('extract_last_30s'):
                    window_data = self._extract_last_30s(window_data, current_sfreq)
                    if window_data is None:
                        continue
                
                # Z-normalize using session-level statistics
                if self.config.is_step_enabled('z_normalize'):
                    window_data = self._z_normalize_with_stats(window_data, sess_mean, sess_std)
                
                # Artifact clipping (±15σ threshold)
                if self.config.is_step_enabled('artifact_clipping'):
                    window_data = self._artifact_clipping(window_data)
                
                # Export to NPY
                if self.config.is_step_enabled('export_npy'):
                    output_file = self._export_npy(
                        subject_id, session_id, trial_idx, window_data
                    )
                    results['output_files'].append(output_file)
                    results['num_trials'] += 1
            
            results['success'] = True
            
            if self.config.verbose:
                print(f"Processed {results['num_trials']} trials")
        
        except Exception as e:
            results['error'] = str(e)
            if self.config.verbose:
                print(f"Error: {e}")
        
        return results
    
    def _extract_last_30s(self, window_data: np.ndarray, sfreq: float) -> Optional[np.ndarray]:
        """Extract last 30s from window."""
        n_samples_needed = int(self.config.window_duration_sec * sfreq)
        n_samples_available = window_data.shape[1]
        
        if n_samples_available < n_samples_needed:
            if self.config.verbose:
                print(f"    Window too short: {n_samples_available/sfreq:.1f}s < {self.config.window_duration_sec}s")
            return None
        
        # Extract last 30s
        start_idx = n_samples_available - n_samples_needed
        window_data = window_data[:, start_idx:]
        
        return window_data
    
    def _z_normalize(self, window_data: np.ndarray) -> np.ndarray:
        """[DEPRECATED - Use _z_normalize_with_stats instead] Apply Z-score normalization per channel."""
        mean = np.mean(window_data, axis=1, keepdims=True)
        std = np.std(window_data, axis=1, keepdims=True)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        window_data = (window_data - mean) / std
        
        return window_data
    
    def _z_normalize_with_stats(self, window_data: np.ndarray, session_mean: np.ndarray, 
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
    
    def _artifact_clipping(self, window_data: np.ndarray) -> np.ndarray:
        """Clip artifacts exceeding ±15σ threshold."""
        threshold = self.config.artifact_threshold_std
        
        # Find and clip values exceeding threshold
        clipped_count = np.sum(np.abs(window_data) > threshold)
        
        window_data = np.clip(window_data, -threshold, threshold)
        
        if self.config.verbose and clipped_count > 0:
            pct = 100 * clipped_count / window_data.size
            print(f"    Clipped artifacts: {clipped_count} samples ({pct:.2f}%)")
        
        return window_data
    
    def _export_npy(self, subject_id: int, session_id: int, trial_idx: int, 
                    window_data: np.ndarray) -> str:
        """Export preprocessed data to NPY file in organized folder structure."""
        # Create nested directory structure: sub-XX/ses-Y/
        subject_dir = Path(self.config.preprocessed_output_dir) / f"sub-{subject_id:02d}"
        session_dir = subject_dir / f"ses-{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"trial-{trial_idx:02d}.npy"
        output_path = session_dir / output_filename
        
        np.save(output_path, window_data.astype(np.float32))
        
        if self.config.verbose:
            shape_str = f"({window_data.shape[0]} channels, {window_data.shape[1]} samples)"
            print(f"    Exported: {output_filename} {shape_str}")
        
        return str(output_path)
    
    def process_all_subjects(self) -> Dict:
        """Process all available subjects and sessions."""
        subjects = self.loader.get_subject_sessions()
        
        all_results = {
            'total_subjects': len(subjects),
            'total_sessions': sum(len(s) for s in subjects.values()),
            'subject_results': {}
        }
        
        for subject_id in sorted(subjects.keys()):
            for session_id in sorted(subjects[subject_id]):
                result = self.process_subject_session(subject_id, session_id)
                
                key = f"sub-{subject_id:02d}_ses-{session_id}"
                all_results['subject_results'][key] = result
        
        return all_results
    
    def get_preprocessed_files(self, subject_id: int, session_id: int) -> List[str]:
        """Get list of preprocessed files for a subject-session."""
        # Files are organized in sub-XX/ses-Y/ directories
        session_dir = Path(self.config.preprocessed_output_dir) / f"sub-{subject_id:02d}" / f"ses-{session_id}"
        
        if not session_dir.exists():
            return []
        
        files = sorted(session_dir.glob("trial-*.npy"))
        return [str(f) for f in files]
    
    def load_preprocessed_data(self, subject_id: int, session_id: int) -> Dict[str, np.ndarray]:
        """Load all preprocessed files for a subject-session."""
        files = self.get_preprocessed_files(subject_id, session_id)
        
        data = {}
        for file_path in files:
            trial_name = Path(file_path).stem
            data[trial_name] = np.load(file_path)
        
        return data
