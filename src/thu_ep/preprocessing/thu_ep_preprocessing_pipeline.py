"""THU-EP EEG preprocessing pipeline.

Implements Phase 1 preprocessing from docs/plan.md:
1. Remove A1, A2 reference channels (32 -> 30 channels)
2. Extract broad-band (band index 5 = 0.5-47 Hz)
3. Downsample 250 Hz -> 200 Hz (7500 -> 6000 samples)
4. Global Z-score normalization per channel
5. Artifact clipping at ±15 standard deviations
6. Export as .npy files

Input:  data/thu ep/EEG data/sub_X.mat
Output: data/thu ep/preprocessed/sub_XX.npy (28, 30, 6000)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re

from .thu_ep_preprocessing_config import THUEPPreprocessingConfig
from .preprocessing_steps import (
    remove_reference_channels,
    extract_frequency_band,
    downsample_stimuli,
    compute_global_statistics,
    z_normalize_global,
    artifact_clipping,
    transpose_to_output_format,
    export_subject_npy,
)


class THUEPPreprocessingPipeline:
    """Pipeline for preprocessing THU-EP EEG data."""
    
    def __init__(self, config: Optional[THUEPPreprocessingConfig] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or THUEPPreprocessingConfig()
    
    def _load_mat_file(self, filepath: Path) -> np.ndarray:
        """
        Load EEG data from MATLAB v7.3 (HDF5) file.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            EEG data array with shape (7500, 32, 28, 6)
        """
        import h5py
        
        with h5py.File(filepath, 'r') as f:
            # THU-EP files have 'data' key containing the EEG
            if 'data' in f:
                data = np.array(f['data'])
            else:
                # Find the main data variable (usually the only non-# variable)
                keys = [k for k in f.keys() if not k.startswith('#')]
                if len(keys) == 1:
                    data = np.array(f[keys[0]])
                else:
                    raise ValueError(f"Cannot determine data key. Available: {list(f.keys())}")
        
        return data
    
    def get_subject_files(self) -> List[Path]:
        """
        Get list of all subject files sorted numerically.
        
        Returns:
            List of paths to subject .mat files
        """
        raw_path = self.config.raw_data_path
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_path}")
        
        # Find all sub_X.mat files
        files = list(raw_path.glob("sub_*.mat"))
        
        # Sort numerically by subject ID
        def get_subject_id(f: Path) -> int:
            match = re.search(r'sub_(\d+)', f.stem)
            return int(match.group(1)) if match else 0
        
        files.sort(key=get_subject_id)
        
        return files
    
    def get_subject_id(self, filepath: Path) -> int:
        """Extract subject ID from filename."""
        match = re.search(r'sub_(\d+)', filepath.stem)
        return int(match.group(1)) if match else 0
    
    def process_subject(self, filepath: Path) -> Dict:
        """
        Process a single subject's EEG data.
        
        Args:
            filepath: Path to subject's .mat file
            
        Returns:
            Dictionary with processing results
        """
        subject_id = self.get_subject_id(filepath)
        
        results = {
            'subject_id': subject_id,
            'input_file': str(filepath),
            'output_file': None,
            'success': False,
            'error': None,
            'statistics': {}
        }
        
        try:
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Processing subject {subject_id}: {filepath.name}")
                print('='*60)
            
            # Load data
            data = self._load_mat_file(filepath)
            if self.config.verbose:
                print(f"  Loaded data: {data.shape}")
                print(f"    Expected: (7500, 32, 28, 6) = (samples, channels, stimuli, bands)")
            
            # Validate shape
            expected_shape = (
                self.config.original_n_samples,
                self.config.n_channels_original,
                self.config.n_stimuli,
                len(self.config.band_names)
            )
            if data.shape != expected_shape:
                raise ValueError(f"Unexpected shape {data.shape}, expected {expected_shape}")
            
            # Step 1: Extract frequency band
            if self.config.is_step_enabled('extract_band'):
                band_name = self.config.band_names[self.config.extract_band_index]
                data = extract_frequency_band(
                    data,
                    self.config.extract_band_index,
                    band_name,
                    verbose=self.config.verbose
                )
                if self.config.verbose:
                    print(f"    Shape after extract_band: {data.shape}")
            
            # Step 2: Remove reference channels
            if self.config.is_step_enabled('remove_reference_channels'):
                data = remove_reference_channels(
                    data,
                    self.config.channels_to_remove_indices,
                    verbose=self.config.verbose
                )
                if self.config.verbose:
                    print(f"    Shape after remove_reference_channels: {data.shape}")
            
            # Step 3: Downsample
            if self.config.is_step_enabled('downsample'):
                data = downsample_stimuli(
                    data,
                    self.config.original_sfreq_hz,
                    self.config.target_sfreq_hz,
                    verbose=self.config.verbose
                )
                if self.config.verbose:
                    print(f"    Shape after downsample: {data.shape}")
            
            # Step 4: Compute global statistics and Z-normalize
            if self.config.is_step_enabled('z_normalize'):
                global_mean, global_std = compute_global_statistics(
                    data,
                    verbose=self.config.verbose
                )
                
                results['statistics']['global_mean'] = global_mean.tolist()
                results['statistics']['global_std'] = global_std.tolist()
                
                data = z_normalize_global(
                    data,
                    global_mean,
                    global_std,
                    verbose=self.config.verbose
                )
                if self.config.verbose:
                    print(f"    Shape after z_normalize: {data.shape}")
            
            # Step 5: Artifact clipping
            if self.config.is_step_enabled('artifact_clipping'):
                data = artifact_clipping(
                    data,
                    self.config.artifact_threshold_std,
                    verbose=self.config.verbose
                )
                if self.config.verbose:
                    print(f"    Shape after artifact_clipping: {data.shape}")
            
            # Transpose to output format: (n_samples, n_channels, n_stimuli) -> (n_stimuli, n_channels, n_samples)
            data = transpose_to_output_format(data, verbose=self.config.verbose)
            
            # Validate final shape
            expected_final = (
                self.config.n_stimuli,
                self.config.n_channels_final,
                self.config.target_n_samples
            )
            if data.shape != expected_final:
                raise ValueError(f"Final shape {data.shape} != expected {expected_final}")
            
            # Step 6: Export
            if self.config.is_step_enabled('export_npy'):
                output_file = export_subject_npy(
                    subject_id,
                    data,
                    str(self.config.output_path),
                    verbose=self.config.verbose
                )
                results['output_file'] = output_file
            
            results['success'] = True
            
            if self.config.verbose:
                print(f"  Subject {subject_id} completed successfully")
        
        except Exception as e:
            results['error'] = str(e)
            if self.config.verbose:
                print(f"  Error: {e}")
        
        return results
    
    def process_all_subjects(self, subject_ids: Optional[List[int]] = None) -> Dict:
        """
        Process all subjects or a subset.
        
        Args:
            subject_ids: Optional list of subject IDs to process. 
                        Processes all if None.
        
        Returns:
            Dictionary with overall processing results
        """
        files = self.get_subject_files()
        
        if subject_ids is not None:
            files = [f for f in files if self.get_subject_id(f) in subject_ids]
        
        all_results = {
            'total_subjects': len(files),
            'successful': 0,
            'failed': 0,
            'subject_results': {}
        }
        
        if self.config.verbose:
            print(f"\nProcessing {len(files)} subjects...")
            print(f"Output directory: {self.config.output_path}")
        
        for filepath in files:
            result = self.process_subject(filepath)
            subject_id = result['subject_id']
            
            all_results['subject_results'][f'sub_{subject_id:02d}'] = result
            
            if result['success']:
                all_results['successful'] += 1
            else:
                all_results['failed'] += 1
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Total subjects: {all_results['total_subjects']}")
            print(f"Successful: {all_results['successful']}")
            print(f"Failed: {all_results['failed']}")
        
        return all_results
    
    def get_preprocessed_files(self) -> List[Path]:
        """Get list of all preprocessed .npy files."""
        output_path = self.config.output_path
        
        if not output_path.exists():
            return []
        
        files = list(output_path.glob("sub_*.npy"))
        
        # Sort numerically
        def get_subject_id(f: Path) -> int:
            match = re.search(r'sub_(\d+)', f.stem)
            return int(match.group(1)) if match else 0
        
        files.sort(key=get_subject_id)
        
        return files
    
    def load_preprocessed_subject(self, subject_id: int) -> np.ndarray:
        """
        Load preprocessed data for a single subject.
        
        Args:
            subject_id: Subject ID (1-80)
            
        Returns:
            Preprocessed data with shape (28, 30, 6000)
        """
        filepath = self.config.output_path / f"sub_{subject_id:02d}.npy"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessed file not found: {filepath}")
        
        return np.load(filepath)
    
    def validate_preprocessed_data(self, subject_id: int) -> Dict:
        """
        Validate that preprocessed data has expected properties.
        
        Args:
            subject_id: Subject ID to validate
            
        Returns:
            Validation results dictionary
        """
        data = self.load_preprocessed_subject(subject_id)
        
        expected_shape = (
            self.config.n_stimuli,
            self.config.n_channels_final,
            self.config.target_n_samples
        )
        
        validation = {
            'subject_id': subject_id,
            'shape': data.shape,
            'shape_valid': data.shape == expected_shape,
            'dtype': str(data.dtype),
            'dtype_valid': data.dtype == np.float32,
            'has_nan': bool(np.isnan(data).any()),
            'has_inf': bool(np.isinf(data).any()),
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'within_clip_range': bool(data.min() >= -self.config.artifact_threshold_std and 
                                      data.max() <= self.config.artifact_threshold_std),
        }
        
        validation['valid'] = (
            validation['shape_valid'] and
            validation['dtype_valid'] and
            not validation['has_nan'] and
            not validation['has_inf'] and
            validation['within_clip_range']
        )
        
        return validation
