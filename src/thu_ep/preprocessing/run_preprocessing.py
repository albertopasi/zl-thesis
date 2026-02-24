"""Run THU-EP preprocessing pipeline.

Usage:
    # Process all subjects
    uv run python -m src.thu_ep.preprocessing.run_preprocessing

    # Process specific subjects
    uv run python -m src.thu_ep.preprocessing.run_preprocessing --subjects 1 2 3

    # Validate preprocessed data
    uv run python -m src.thu_ep.preprocessing.run_preprocessing --validate

    # Process single subject with verbose output
    uv run python -m src.thu_ep.preprocessing.run_preprocessing --subjects 1 --verbose
"""

import argparse
from pathlib import Path

from .thu_ep_preprocessing_config import THUEPPreprocessingConfig
from .thu_ep_preprocessing_pipeline import THUEPPreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="THU-EP EEG preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        help='Subject IDs to process (e.g., 1 2 3). Processes all if not specified.'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate preprocessed data instead of processing.'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True).'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output.'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/thu ep/preprocessed',
        help='Output directory for preprocessed files.'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List subjects to process without actually processing.'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = THUEPPreprocessingConfig(
        preprocessed_output_dir=args.output_dir,
        verbose=not args.quiet and args.verbose
    )
    
    # Create pipeline
    pipeline = THUEPPreprocessingPipeline(config)
    
    if args.dry_run:
        # List subjects without processing
        files = pipeline.get_subject_files()
        if args.subjects:
            files = [f for f in files if pipeline.get_subject_id(f) in args.subjects]
        
        print(f"Would process {len(files)} subjects:")
        for f in files:
            print(f"  {f.name}")
        return
    
    if args.validate:
        # Validate preprocessed data
        preprocessed_files = pipeline.get_preprocessed_files()
        
        if not preprocessed_files:
            print("No preprocessed files found.")
            return
        
        print(f"Validating {len(preprocessed_files)} preprocessed files...")
        
        all_valid = True
        for filepath in preprocessed_files:
            subject_id = pipeline.get_subject_id(filepath)
            
            if args.subjects and subject_id not in args.subjects:
                continue
            
            validation = pipeline.validate_preprocessed_data(subject_id)
            
            status = "✓" if validation['valid'] else "✗"
            print(f"\n{status} Subject {subject_id}:")
            print(f"    Shape: {validation['shape']} (valid: {validation['shape_valid']})")
            print(f"    Range: [{validation['min']:.2f}, {validation['max']:.2f}]")
            print(f"    Mean: {validation['mean']:.4f}, Std: {validation['std']:.4f}")
            print(f"    Within clip range: {validation['within_clip_range']}")
            
            if not validation['valid']:
                all_valid = False
                if validation['has_nan']:
                    print("    ⚠ Contains NaN values!")
                if validation['has_inf']:
                    print("    ⚠ Contains Inf values!")
        
        print(f"\n{'='*40}")
        if all_valid:
            print("All validated subjects passed!")
        else:
            print("Some subjects failed validation.")
        
        return
    
    # Process subjects
    results = pipeline.process_all_subjects(subject_ids=args.subjects)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {results['total_subjects']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    if results['failed'] > 0:
        print("\nFailed subjects:")
        for key, result in results['subject_results'].items():
            if not result['success']:
                print(f"  {key}: {result['error']}")


if __name__ == '__main__':
    main()
