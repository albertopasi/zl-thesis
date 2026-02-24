"""Analyze mean and standard deviation for each channel in preprocessed SEED data."""
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_channel_statistics(data_dir: str = None, subject: str = "sub-01", session: str = "ses-1"):
    """
    Analyze channel statistics from preprocessed SEED npy files.
    
    Args:
        data_dir: Path to preprocessed_seed directory. If None, uses default path.
        subject: Subject ID (default: "sub-01")
        session: Session ID (default: "ses-1")
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "SEED" / "preprocessed_seed"
    else:
        data_dir = Path(data_dir)
    
    subject_dir = data_dir / subject / session
    
    if not subject_dir.exists():
        print(f"Directory not found: {subject_dir}")
        return
    
    # Load all trial files
    trial_files = sorted(subject_dir.glob("trial-*.npy"))
    print(f"Found {len(trial_files)} trial files in {subject}/{session}\n")
    
    # Analyze each trial separately
    all_stats = []
    for trial_file in trial_files:
        trial_data = np.load(trial_file)
        print(f"\n{trial_file.name}: shape = {trial_data.shape}")
        
        # Calculate statistics per channel for this trial
        means = np.mean(trial_data, axis=1)
        stdevs = np.std(trial_data, axis=1)
        
        # Create a DataFrame for this trial
        stats_df = pd.DataFrame({
            'Channel': [f"Ch {i}" for i in range(len(means))],
            'Mean': means,
            'Std Dev': stdevs,
            'Min': np.min(trial_data, axis=1),
            'Max': np.max(trial_data, axis=1),
        })
        
        print(stats_df.to_string(index=False))
        all_stats.append(stats_df)
    
    return all_stats


if __name__ == "__main__":
    all_stats = analyze_channel_statistics()
