"""Test that preprocessed NPY files have artifact clipping applied correctly."""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from preprocess_seed.seed_preprocessing_pipeline import SEEDPreprocessingPipeline
from preprocess_seed.seed_preprocessing_config import SEEDPreprocessingConfig


class TestArtifactClipping:
    """Test artifact clipping validation on preprocessed data."""
    
    @pytest.fixture
    def pipeline(self):
        """Create preprocessing pipeline."""
        config = SEEDPreprocessingConfig(verbose=False)
        return SEEDPreprocessingPipeline(config)
    
    @pytest.fixture
    def preprocessed_files(self, pipeline):
        """Process subject 1, session 1 and return file paths."""
        # Process subject 1, session 1
        result = pipeline.process_subject_session(subject_id=1, session_id=1)
        
        assert result['success'], "Preprocessing failed"
        assert result['num_trials'] > 0, "No trials were processed"
        
        return pipeline.get_preprocessed_files(subject_id=1, session_id=1)
    
    def test_all_values_within_artifact_threshold(self, preprocessed_files):
        """Test that all values in NPY files are within ±15σ threshold."""
        threshold = 15.0
        
        print(f"\n{'='*80}")
        print("ARTIFACT CLIPPING VERIFICATION")
        print(f"{'='*80}")
        print(f"Testing {len(preprocessed_files)} trials for values within ±{threshold}σ\n")
        
        all_within_threshold = True
        
        for file_path in preprocessed_files:
            data = np.load(file_path)
            
            # Find any values outside threshold
            violations = np.abs(data) > threshold
            violation_count = np.sum(violations)
            
            trial_name = Path(file_path).stem
            
            if violation_count > 0:
                all_within_threshold = False
                max_val = np.max(np.abs(data))
                pct = 100 * violation_count / data.size
                print(f"❌ {trial_name}: {violation_count} violations ({pct:.4f}%), max={max_val:.2f}σ")
            else:
                max_val = np.max(np.abs(data))
                print(f"✓ {trial_name}: All values within ±{threshold}σ (max={max_val:.4f}σ)")
        
        print(f"\n{'='*80}")
        
        assert all_within_threshold, f"Found values exceeding ±{threshold}σ threshold"
        print(f"✓ All trials passed artifact clipping verification")
    
    def test_data_shape_is_62x6000(self, preprocessed_files):
        """Test that all trials have correct shape (62 channels, 6000 samples)."""
        expected_channels = 62
        expected_samples = 6000
        
        print(f"\n{'='*80}")
        print("DATA SHAPE VERIFICATION")
        print(f"{'='*80}")
        print(f"Expected shape: ({expected_channels}, {expected_samples})\n")
        
        for file_path in preprocessed_files:
            data = np.load(file_path)
            trial_name = Path(file_path).stem
            
            assert data.shape[0] == expected_channels, \
                f"{trial_name}: Expected {expected_channels} channels, got {data.shape[0]}"
            assert data.shape[1] == expected_samples, \
                f"{trial_name}: Expected {expected_samples} samples, got {data.shape[1]}"
            
            print(f"✓ {trial_name}: shape {data.shape}")
        
        print(f"\n{'='*80}")
        print(f"✓ All {len(preprocessed_files)} trials have correct shape")
    
    def test_data_statistics(self, preprocessed_files):
        """Print statistics for each trial to verify normalization."""
        print(f"\n{'='*80}")
        print("DATA STATISTICS")
        print(f"{'='*80}")
        print(f"{'Trial':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        all_means = []
        all_stds = []
        
        for file_path in preprocessed_files:
            data = np.load(file_path)
            trial_name = Path(file_path).stem
            
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            all_means.append(mean)
            all_stds.append(std)
            
            print(f"{trial_name:<12} {mean:<12.6f} {std:<12.6f} {min_val:<12.4f} {max_val:<12.4f}")
        
        print(f"\n{'='*80}")
        print(f"Overall statistics across all trials:")
        print(f"  Mean of means: {np.mean(all_means):.6f}")
        print(f"  Mean of stds: {np.mean(all_stds):.6f}")
        print(f"  Min mean: {np.min(all_means):.6f}")
        print(f"  Max mean: {np.max(all_means):.6f}")


def test_load_and_verify_single_trial():
    """Load and verify a single trial file directly."""
    config = SEEDPreprocessingConfig(verbose=False)
    pipeline = SEEDPreprocessingPipeline(config)
    
    files = pipeline.get_preprocessed_files(subject_id=1, session_id=1)
    
    if not files:
        pytest.skip("No preprocessed files found for subject 1, session 1")
    
    # Load first trial
    data = np.load(files[0])
    
    print(f"\n{'='*80}")
    print("SINGLE TRIAL VERIFICATION")
    print(f"{'='*80}")
    print(f"File: {Path(files[0]).name}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.min(data):.4f}")
    print(f"Max value: {np.max(data):.4f}")
    print(f"Mean: {np.mean(data):.6f}")
    print(f"Std: {np.std(data):.6f}")
    
    # Check threshold
    threshold = 15.0
    violations = np.sum(np.abs(data) > threshold)
    
    if violations > 0:
        print(f"\n⚠ Found {violations} values exceeding ±{threshold}σ")
    else:
        print(f"\n✓ All values within ±{threshold}σ")
    
    assert violations == 0, f"Found {violations} values exceeding threshold"
