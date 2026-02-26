"""
Comprehensive data quality inspection for THU-EP EEG data.

Checks for:
- Flat lines (constant or near-constant values)
- NaN/Inf values
- Extreme outliers
- Zero variance channels
- Shape validation
- Statistics anomalies

Usage:
    # Inspect specific subject/stimulus
    uv run python -m src.thu_ep.exploration.inspect_data_quality --subject 1 --stimulus 0
    
    # Scan all subjects for issues
    uv run python -m src.thu_ep.exploration.inspect_data_quality --scan-all
    
    # Detailed report for one subject (all stimuli)
    uv run python -m src.thu_ep.exploration.inspect_data_quality --subject 33 --all-stimuli
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys

import numpy as np
import h5py

from ..config import get_config

# Load configuration
_cfg = get_config()

# Configuration (from config)
DATA_DIR = _cfg.raw_data_dir.parent
RAW_DATA_DIR = _cfg.raw_data_dir
PREPROCESSED_DATA_DIR = _cfg.preprocessed_dir

ALL_CHANNELS = _cfg.all_channels

EXPECTED_SHAPE = _cfg.expected_raw_shape
BROAD_BAND_INDEX = _cfg.broad_band_index
ORIGINAL_SFREQ = _cfg.original_sfreq


def load_raw_mat_file(filepath: Path) -> np.ndarray:
    """Load raw EEG data from MATLAB v7.3 (HDF5) file."""
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


def check_shape(data: np.ndarray) -> Dict:
    """Check if data shape matches expected."""
    return {
        'actual_shape': data.shape,
        'expected_shape': EXPECTED_SHAPE,
        'shape_ok': data.shape == EXPECTED_SHAPE,
    }


def check_nan_inf(data: np.ndarray) -> Dict:
    """Check for NaN and Inf values."""
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    total = data.size
    return {
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'nan_percent': 100 * nan_count / total,
        'inf_percent': 100 * inf_count / total,
        'has_nan': nan_count > 0,
        'has_inf': inf_count > 0,
    }


def check_zeros(eeg: np.ndarray) -> Dict:
    """Check for all-zero or mostly-zero data."""
    zero_count = np.sum(eeg == 0)
    total = eeg.size
    return {
        'zero_count': int(zero_count),
        'zero_percent': 100 * zero_count / total,
        'all_zeros': zero_count == total,
    }


def check_flat_channels(eeg: np.ndarray, threshold: float = 1e-6) -> Dict:
    """
    Check for flat (constant) channels.
    
    Args:
        eeg: Shape (channels, samples)
        threshold: Std threshold below which channel is considered flat
    """
    channel_stds = np.std(eeg, axis=1)
    flat_mask = channel_stds < threshold
    flat_indices = np.where(flat_mask)[0]
    return {
        'channel_stds': channel_stds,
        'flat_channels': [ALL_CHANNELS[i] for i in flat_indices if i < len(ALL_CHANNELS)],
        'flat_channel_indices': flat_indices.tolist(),
        'n_flat': int(np.sum(flat_mask)),
        'has_flat': np.any(flat_mask),
    }


def check_low_variance(eeg: np.ndarray, percentile_threshold: float = 1.0) -> Dict:
    """
    Check for abnormally low variance channels (relative to dataset).
    
    Args:
        eeg: Shape (channels, samples)
        percentile_threshold: Flag channels below this percentile of variance
    """
    channel_vars = np.var(eeg, axis=1)
    var_threshold = np.percentile(channel_vars, percentile_threshold)
    low_var_mask = channel_vars < var_threshold
    return {
        'channel_variances': channel_vars,
        'variance_threshold': var_threshold,
        'low_variance_channels': [ALL_CHANNELS[i] for i in np.where(low_var_mask)[0] if i < len(ALL_CHANNELS)],
        'n_low_variance': int(np.sum(low_var_mask)),
    }


def check_statistics(eeg: np.ndarray) -> Dict:
    """Compute comprehensive statistics."""
    return {
        'global_mean': float(np.mean(eeg)),
        'global_std': float(np.std(eeg)),
        'global_min': float(np.min(eeg)),
        'global_max': float(np.max(eeg)),
        'global_range': float(np.max(eeg) - np.min(eeg)),
        'channel_means': np.mean(eeg, axis=1),
        'channel_stds': np.std(eeg, axis=1),
        'channel_mins': np.min(eeg, axis=1),
        'channel_maxs': np.max(eeg, axis=1),
    }


def check_outliers(eeg: np.ndarray, n_std: float = 10.0) -> Dict:
    """Check for extreme outliers."""
    mean = np.mean(eeg)
    std = np.std(eeg)
    threshold_low = mean - n_std * std
    threshold_high = mean + n_std * std
    outliers_low = np.sum(eeg < threshold_low)
    outliers_high = np.sum(eeg > threshold_high)
    return {
        'threshold_low': threshold_low,
        'threshold_high': threshold_high,
        'n_outliers_low': int(outliers_low),
        'n_outliers_high': int(outliers_high),
        'outlier_percent': 100 * (outliers_low + outliers_high) / eeg.size,
    }


def check_constant_segments(eeg: np.ndarray, min_length: int = 100) -> Dict:
    """
    Check for long constant segments (potential recording issues).
    
    Args:
        eeg: Shape (channels, samples)
        min_length: Minimum consecutive samples to flag as constant
    """
    constant_segments = []
    
    for ch_idx in range(eeg.shape[0]):
        channel = eeg[ch_idx, :]
        diff = np.diff(channel)
        
        # Find runs of zeros in diff (constant regions)
        is_constant = np.abs(diff) < 1e-10
        
        # Find start/end of constant runs
        changes = np.diff(np.concatenate([[0], is_constant.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for start, end in zip(starts, ends):
            length = end - start
            if length >= min_length:
                constant_segments.append({
                    'channel': ALL_CHANNELS[ch_idx] if ch_idx < len(ALL_CHANNELS) else f"ch{ch_idx}",
                    'channel_idx': ch_idx,
                    'start_sample': int(start),
                    'end_sample': int(end),
                    'length_samples': int(length),
                    'length_seconds': length / ORIGINAL_SFREQ,
                })
    
    return {
        'constant_segments': constant_segments,
        'n_segments': len(constant_segments),
        'has_constant_segments': len(constant_segments) > 0,
    }


def inspect_stimulus(
    subject_id: int,
    stimulus_idx: int,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive inspection of one stimulus for one subject.
    
    Returns dict with all checks and an 'issues' summary.
    """
    filepath = RAW_DATA_DIR / f"sub_{subject_id}.mat"
    
    if not filepath.exists():
        return {'error': f"File not found: {filepath}", 'has_issues': True}
    
    try:
        data = load_raw_mat_file(filepath)
    except Exception as e:
        return {'error': f"Failed to load file: {e}", 'has_issues': True}
    
    # Shape check
    shape_result = check_shape(data)
    if not shape_result['shape_ok']:
        return {
            'error': f"Shape mismatch: {shape_result['actual_shape']} != {shape_result['expected_shape']}",
            'has_issues': True,
            'shape': shape_result,
        }
    
    # Extract broad-band for this stimulus
    eeg = data[:, :, stimulus_idx, BROAD_BAND_INDEX]  # (7500, 32)
    eeg = eeg.T  # (32, 7500)
    
    # Run all checks
    results = {
        'subject_id': subject_id,
        'stimulus_idx': stimulus_idx,
        'shape': shape_result,
        'nan_inf': check_nan_inf(eeg),
        'zeros': check_zeros(eeg),
        'flat_channels': check_flat_channels(eeg),
        'statistics': check_statistics(eeg),
        'outliers': check_outliers(eeg),
        'constant_segments': check_constant_segments(eeg),
    }
    
    # Summarize issues
    issues = []
    if results['nan_inf']['has_nan']:
        issues.append(f"NaN values: {results['nan_inf']['nan_count']}")
    if results['nan_inf']['has_inf']:
        issues.append(f"Inf values: {results['nan_inf']['inf_count']}")
    if results['zeros']['all_zeros']:
        issues.append("All zeros")
    if results['zeros']['zero_percent'] > 50:
        issues.append(f"High zero content: {results['zeros']['zero_percent']:.1f}%")
    if results['flat_channels']['has_flat']:
        issues.append(f"Flat channels: {results['flat_channels']['flat_channels']}")
    if results['constant_segments']['has_constant_segments']:
        n = results['constant_segments']['n_segments']
        issues.append(f"Constant segments: {n}")
    if results['statistics']['global_std'] < 0.1:
        issues.append(f"Very low std: {results['statistics']['global_std']:.4f}")
    if results['statistics']['global_range'] < 1.0:
        issues.append(f"Very small range: {results['statistics']['global_range']:.4f}")
    if results['outliers']['outlier_percent'] > 1.0:
        issues.append(f"Many outliers: {results['outliers']['outlier_percent']:.2f}%")
    
    results['issues'] = issues
    results['has_issues'] = len(issues) > 0
    
    if verbose:
        print_inspection_report(results)
    
    return results


def print_inspection_report(results: Dict):
    """Print formatted inspection report."""
    print("\n" + "=" * 70)
    print(f"DATA QUALITY REPORT: Subject {results['subject_id']}, Stimulus {results['stimulus_idx']}")
    print("=" * 70)
    
    if 'error' in results:
        print(f"\n✗ ERROR: {results['error']}")
        return
    
    # Statistics
    stats = results['statistics']
    print(f"\n📊 STATISTICS:")
    print(f"   Mean:  {stats['global_mean']:>10.4f}")
    print(f"   Std:   {stats['global_std']:>10.4f}")
    print(f"   Min:   {stats['global_min']:>10.4f}")
    print(f"   Max:   {stats['global_max']:>10.4f}")
    print(f"   Range: {stats['global_range']:>10.4f}")
    
    # Per-channel stats (first 5 and last 5)
    print(f"\n📈 PER-CHANNEL (first 5):")
    print(f"   {'Channel':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for i in range(min(5, len(ALL_CHANNELS))):
        print(f"   {ALL_CHANNELS[i]:<8} {stats['channel_means'][i]:>10.2f} "
              f"{stats['channel_stds'][i]:>10.2f} {stats['channel_mins'][i]:>10.2f} "
              f"{stats['channel_maxs'][i]:>10.2f}")
    
    # NaN/Inf
    ni = results['nan_inf']
    status = "✓" if not (ni['has_nan'] or ni['has_inf']) else "✗"
    print(f"\n{status} NaN/Inf: {ni['nan_count']} NaN, {ni['inf_count']} Inf")
    
    # Zeros
    z = results['zeros']
    status = "✓" if not z['all_zeros'] and z['zero_percent'] < 10 else "⚠" if z['zero_percent'] < 50 else "✗"
    print(f"{status} Zeros: {z['zero_percent']:.2f}% ({z['zero_count']} samples)")
    
    # Flat channels
    fc = results['flat_channels']
    status = "✓" if not fc['has_flat'] else "✗"
    if fc['has_flat']:
        print(f"{status} Flat channels: {fc['flat_channels']}")
    else:
        print(f"{status} No flat channels")
    
    # Constant segments
    cs = results['constant_segments']
    status = "✓" if not cs['has_constant_segments'] else "⚠"
    if cs['has_constant_segments']:
        print(f"{status} Constant segments: {cs['n_segments']} found")
        for seg in cs['constant_segments'][:3]:  # Show first 3
            print(f"      {seg['channel']}: {seg['length_seconds']:.2f}s "
                  f"(samples {seg['start_sample']}-{seg['end_sample']})")
        if len(cs['constant_segments']) > 3:
            print(f"      ... and {len(cs['constant_segments']) - 3} more")
    else:
        print(f"{status} No constant segments (>100 samples)")
    
    # Outliers
    ol = results['outliers']
    status = "✓" if ol['outlier_percent'] < 0.1 else "⚠" if ol['outlier_percent'] < 1 else "✗"
    print(f"{status} Outliers (±10σ): {ol['outlier_percent']:.3f}% "
          f"({ol['n_outliers_low']} low, {ol['n_outliers_high']} high)")
    
    # Summary
    print("\n" + "-" * 70)
    if results['has_issues']:
        print("⚠ ISSUES FOUND:")
        for issue in results['issues']:
            print(f"   • {issue}")
    else:
        print("✓ No significant issues detected")
    print()


def scan_all_subjects(
    verbose: bool = False,
    check_all_stimuli: bool = False
) -> List[Dict]:
    """
    Scan all subjects for data quality issues.
    
    Args:
        verbose: Print details for each subject
        check_all_stimuli: Check all 28 stimuli per subject (slow) vs just stimulus 0
    """
    print("\n" + "=" * 70)
    print("SCANNING ALL SUBJECTS FOR DATA QUALITY ISSUES")
    print("=" * 70)
    
    issues_found = []
    
    for subject_id in range(1, 81):
        filepath = RAW_DATA_DIR / f"sub_{subject_id}.mat"
        if not filepath.exists():
            print(f"Subject {subject_id}: ✗ File not found")
            continue
        
        stimuli_to_check = range(28) if check_all_stimuli else [0]
        subject_issues = []
        
        for stimulus_idx in stimuli_to_check:
            result = inspect_stimulus(subject_id, stimulus_idx, verbose=verbose)
            if result['has_issues']:
                subject_issues.append({
                    'stimulus': stimulus_idx,
                    'issues': result.get('issues', [result.get('error', 'Unknown')]),
                })
        
        if subject_issues:
            issues_found.append({
                'subject_id': subject_id,
                'stimuli_with_issues': subject_issues,
            })
            n_stim = len(subject_issues)
            print(f"Subject {subject_id:2d}: ⚠ Issues in {n_stim} stimulus/stimuli")
            if not check_all_stimuli:
                for si in subject_issues:
                    for issue in si['issues']:
                        print(f"           • {issue}")
        else:
            print(f"Subject {subject_id:2d}: ✓ OK")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total subjects: 80")
    print(f"Subjects with issues: {len(issues_found)}")
    
    if issues_found:
        print("\nSubjects requiring attention:")
        for item in issues_found:
            print(f"  Subject {item['subject_id']}: {len(item['stimuli_with_issues'])} problematic stimuli")
    
    return issues_found


def scan_extreme_artifacts(
    std_threshold: float = 500.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Scan all subjects for extreme electrode artifacts.
    
    Artifacts cause auto-scaling to make normal EEG appear as "flat lines".
    Normal EEG std is ~5-50, artifacts can push std to 1000-10000+.
    
    Args:
        std_threshold: Flag stimuli with global std above this value
        verbose: Print progress
    
    Returns:
        List of problematic subjects with affected stimuli
    """
    print("\n" + "=" * 70)
    print(f"SCANNING FOR EXTREME ARTIFACTS (std > {std_threshold})")
    print("=" * 70)
    print("This detects electrode artifacts that cause 'flat line' appearance")
    print("when auto-scaling tries to fit extreme outliers.")
    print()
    
    problematic = []
    
    for subject_id in range(1, 81):
        filepath = RAW_DATA_DIR / f"sub_{subject_id}.mat"
        if not filepath.exists():
            continue
        
        try:
            data = load_raw_mat_file(filepath)
        except Exception as e:
            print(f"Subject {subject_id:2d}: ✗ Load error: {e}")
            continue
        
        bad_stimuli = []
        max_std = 0
        
        for stim in range(28):
            eeg = data[:, :, stim, BROAD_BAND_INDEX].T  # (32, 7500)
            std = float(np.std(eeg))
            max_std = max(max_std, std)
            
            if std > std_threshold:
                # Find which channel(s) have the extreme values
                channel_stds = np.std(eeg, axis=1)
                worst_ch_idx = int(np.argmax(channel_stds))
                worst_ch_std = float(channel_stds[worst_ch_idx])
                ch_name = ALL_CHANNELS[worst_ch_idx] if worst_ch_idx < len(ALL_CHANNELS) else f"ch{worst_ch_idx}"
                
                bad_stimuli.append({
                    'stimulus': stim,
                    'std': std,
                    'range': (float(eeg.min()), float(eeg.max())),
                    'worst_channel': ch_name,
                    'worst_channel_std': worst_ch_std,
                })
        
        if bad_stimuli:
            problematic.append({
                'subject_id': subject_id,
                'n_affected': len(bad_stimuli),
                'max_std': max_std,
                'affected_stimuli': bad_stimuli,
            })
            if verbose:
                print(f"Subject {subject_id:2d}: ⚠ {len(bad_stimuli):2d}/28 stimuli affected, max_std={max_std:>10.0f}")
        else:
            if verbose:
                print(f"Subject {subject_id:2d}: ✓ OK (max_std={max_std:.1f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Subjects with extreme artifacts: {len(problematic)}/80")
    
    if problematic:
        # Sort by number of affected stimuli (descending)
        problematic.sort(key=lambda x: -x['n_affected'])
        
        print("\nSubjects to potentially exclude or inspect carefully:")
        print(f"{'Subject':>8} {'Affected':>10} {'Max Std':>12}")
        print("-" * 35)
        for item in problematic:
            print(f"{item['subject_id']:>8d} {item['n_affected']:>7d}/28 {item['max_std']:>12.0f}")
        
        # Show detailed breakdown for worst subject
        worst = problematic[0]
        print(f"\n📋 Detailed breakdown for Subject {worst['subject_id']} ({worst['n_affected']}/28 affected):")
        print(f"   {'Stim':>4} {'Std':>12} {'Range':>28} {'Worst Ch':>10}")
        for stim_info in worst['affected_stimuli'][:10]:  # Show first 10
            rng = stim_info['range']
            print(f"   {stim_info['stimulus']:>4d} {stim_info['std']:>12.0f} "
                  f"[{rng[0]:>12.0f}, {rng[1]:>12.0f}] {stim_info['worst_channel']:>10}")
        if len(worst['affected_stimuli']) > 10:
            print(f"   ... and {len(worst['affected_stimuli']) - 10} more")
    else:
        print("\n✓ No subjects with extreme artifacts detected!")
    
    return problematic


def inspect_all_stimuli(subject_id: int) -> List[Dict]:
    """Inspect all 28 stimuli for one subject."""
    print(f"\n{'=' * 70}")
    print(f"INSPECTING ALL STIMULI FOR SUBJECT {subject_id}")
    print(f"{'=' * 70}")
    
    results = []
    issues_summary = []
    
    for stimulus_idx in range(28):
        result = inspect_stimulus(subject_id, stimulus_idx, verbose=False)
        results.append(result)
        
        if result['has_issues']:
            issues_summary.append((stimulus_idx, result['issues']))
    
    # Print summary table
    print(f"\n{'Stim':<6} {'Mean':>10} {'Std':>10} {'Range':>10} {'Status':<20}")
    print("-" * 60)
    
    for i, result in enumerate(results):
        if 'error' in result:
            print(f"{i:<6} {'ERROR':>10} {'-':>10} {'-':>10} ✗ {result['error'][:15]}")
            continue
        
        stats = result['statistics']
        status = "✓ OK" if not result['has_issues'] else f"⚠ {len(result['issues'])} issues"
        print(f"{i:<6} {stats['global_mean']:>10.2f} {stats['global_std']:>10.2f} "
              f"{stats['global_range']:>10.2f} {status:<20}")
    
    # Detailed issues
    if issues_summary:
        print(f"\n⚠ STIMULI WITH ISSUES:")
        for stimulus_idx, issues in issues_summary:
            print(f"\n  Stimulus {stimulus_idx}:")
            for issue in issues:
                print(f"    • {issue}")
    else:
        print(f"\n✓ All stimuli OK for subject {subject_id}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive data quality inspection for THU-EP EEG"
    )
    parser.add_argument(
        "-s", "--subject",
        type=int,
        help="Subject ID (1-80)"
    )
    parser.add_argument(
        "-st", "--stimulus",
        type=int,
        default=0,
        help="Stimulus index (0-27), default: 0"
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan all 80 subjects for issues"
    )
    parser.add_argument(
        "--all-stimuli",
        action="store_true",
        help="Check all 28 stimuli for specified subject"
    )
    parser.add_argument(
        "--deep-scan",
        action="store_true",
        help="Deep scan: check all stimuli for all subjects (slow)"
    )
    parser.add_argument(
        "--scan-artifacts",
        action="store_true",
        help="Scan all subjects for extreme electrode artifacts (fast)"
    )
    parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=500.0,
        help="Std threshold for artifact detection (default: 500)"
    )
    
    args = parser.parse_args()
    
    if args.scan_artifacts:
        scan_extreme_artifacts(std_threshold=args.artifact_threshold, verbose=True)
    elif args.scan_all:
        scan_all_subjects(verbose=False, check_all_stimuli=False)
    elif args.deep_scan:
        scan_all_subjects(verbose=False, check_all_stimuli=True)
    elif args.subject is not None:
        if args.all_stimuli:
            inspect_all_stimuli(args.subject)
        else:
            inspect_stimulus(args.subject, args.stimulus, verbose=True)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  uv run python -m src.thu_ep.exploration.inspect_data_quality -s 1 -st 0")
        print("  uv run python -m src.thu_ep.exploration.inspect_data_quality -s 33 --all-stimuli")
        print("  uv run python -m src.thu_ep.exploration.inspect_data_quality --scan-all")
        print("  uv run python -m src.thu_ep.exploration.inspect_data_quality --scan-artifacts")


if __name__ == "__main__":
    main()
