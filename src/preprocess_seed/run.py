"""Run SEED preprocessing pipeline with optional subject/session arguments.

Usage:
  python run.py                    # Process all subjects
  python run.py -s 1               # Process subject 1 (all sessions)
  python run.py -s 1 -se 1         # Process subject 1, session 1
  python run.py -s 1 -se 1 -q      # Quiet mode
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocess_seed.preprocessing_pipeline import SEEDPreprocessingPipeline
from src.preprocess_seed.preprocessing_config import SEEDPreprocessingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run SEED EEG preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # All subjects
  python run.py -s 1               # Subject 1, all sessions
  python run.py -s 1 -se 1         # Subject 1, session 1
  python run.py -s 1 -se 1 -q      # Quiet mode
        """
    )
    
    parser.add_argument('-s', '--subject', type=int, default=None,
                        help='Subject ID (1-15). Default: all')
    parser.add_argument('-se', '--session', type=int, default=None,
                        help='Session ID (1-3). Requires --subject')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Validate
    if args.session is not None and args.subject is None:
        parser.error("--session requires --subject")
    
    if args.subject is not None:
        if not (1 <= args.subject <= 15):
            parser.error("Subject ID must be 1-15")
        if args.session is not None and not (1 <= args.session <= 3):
            parser.error("Session ID must be 1-3")
    
    config = SEEDPreprocessingConfig()
    config.verbose = not args.quiet
    
    pipeline = SEEDPreprocessingPipeline(config)
    
    if args.subject is None:
        # All subjects
        print("\n" + "="*70)
        print("PROCESSING ALL SUBJECTS AND SESSIONS")
        print("="*70)
        
        all_results = pipeline.process_all_subjects()
        
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        print(f"Total subjects: {all_results['total_subjects']}")
        print(f"Total sessions: {all_results['total_sessions']}")
        print(f"Expected trials: {all_results['total_subjects'] * 3 * 15}")
        
        successful = sum(1 for r in all_results['subject_results'].values() if r['success'])
        total_trials = sum(r['num_trials'] for r in all_results['subject_results'].values())
        
        print(f"\nSuccessful processing: {successful}/{all_results['total_sessions']}")
        print(f"Total trials processed: {total_trials}")
        
        errors = [k for k, v in all_results['subject_results'].items() if v['error']]
        if errors:
            print(f"\nErrors encountered ({len(errors)}):")
            for k in errors:
                print(f"  ✗ {k}: {all_results['subject_results'][k]['error'][:50]}...")
        else:
            print(f"\n✓ All subjects processed successfully!")
    
    else:
        subjects = pipeline.loader.get_subject_sessions()
        
        if args.subject not in subjects:
            print(f"Error: Subject {args.subject} not found")
            return
        
        if args.session is None:
            # All sessions for subject
            sessions = sorted(subjects[args.subject])
            print("\n" + "="*70)
            print(f"PROCESSING: Subject {args.subject}, All Sessions ({len(sessions)})")
            print("="*70)
            
            results_list = []
            for session_id in sessions:
                result = pipeline.process_subject_session(args.subject, session_id)
                results_list.append(result)
            
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            successful = sum(1 for r in results_list if r['success'])
            total_trials = sum(r['num_trials'] for r in results_list)
            print(f"Sessions processed: {successful}/{len(sessions)}")
            print(f"Total trials: {total_trials}")
        
        else:
            # Specific subject and session
            if args.session not in subjects[args.subject]:
                print(f"Error: Session {args.session} not found for subject {args.subject}")
                return
            
            print("\n" + "="*70)
            print(f"PROCESSING: Subject {args.subject}, Session {args.session}")
            print("="*70)
            
            result = pipeline.process_subject_session(args.subject, args.session)
            
            print("\n" + "="*70)
            print("RESULT")
            print("="*70)
            print(f"Success: {result['success']}")
            print(f"Trials processed: {result['num_trials']}")
            
            if result['error']:
                print(f"Error: {result['error']}")
            else:
                print(f"✓ Completed successfully!")


if __name__ == "__main__":
    main()
