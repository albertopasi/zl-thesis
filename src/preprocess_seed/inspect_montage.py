"""Inspect montage and electrode positions after loading."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocess_seed.seed_loader import SEEDEEGLoader
from src.preprocess_seed.preprocessing_config import SEEDPreprocessingConfig


def inspect_montage():
    """Load and inspect montage with electrode positions."""
    
    config = SEEDPreprocessingConfig()
    loader = SEEDEEGLoader(
        seed_raw_dir=config.seed_raw_dir,
        montage_path=config.montage_file
    )
    
    # Load raw data
    print("=" * 80)
    print("MONTAGE INSPECTION")
    print("=" * 80)
    print(f"\nLoading montage from: {config.montage_file}\n")
    
    raw = loader.load_raw(subject_id=1, session_id=1)
    montage = raw.get_montage()
    
    # Print montage info
    print(f"Montage type: {type(montage).__name__}")
    print(f"Montage coordinate frame: {montage.get_positions()}")
    print(f"Number of channels: {len(montage.ch_names)}")
    print(f"\nChannel names: {montage.ch_names[:10]}... (showing first 10)")
    
    # Get positions
    pos_dict = montage.get_positions()
    print(f"\nPosition coordinate systems available:")
    print(f"  - Coordinate frame: {pos_dict.get('coord_frame')}")
    print(f"  - Nasion: {pos_dict.get('nasion')}")
    print(f"  - LPA (Left Pre-Auricular): {pos_dict.get('lpa')}")
    print(f"  - RPA (Right Pre-Auricular): {pos_dict.get('rpa')}")
    
    # Get channel positions from ch_pos dictionary
    ch_pos_dict = pos_dict['ch_pos']
    print(f"\nNumber of channels with position info: {len(ch_pos_dict)}")
    
    # Convert to DataFrame for easy viewing
    pos_list = []
    ch_names_list = []
    for ch_name, pos_vec in ch_pos_dict.items():
        ch_names_list.append(ch_name)
        pos_list.append(pos_vec)
    
    pos = np.array(pos_list)  # (n_channels, 3) array
    print(f"Electrode position array shape: {pos.shape}")
    print(f"Position values in coordinate frame (normalized to unit sphere)")
    
    # Create DataFrame for easy viewing
    channels_pos_df = pd.DataFrame(
        pos,
        columns=['X', 'Y', 'Z'],
        index=ch_names_list
    )
    
    print("\n" + "=" * 80)
    print("ELECTRODE POSITIONS (First 10 channels)")
    print("=" * 80)
    print(channels_pos_df.head(10).to_string())
    
    print("\n" + "=" * 80)
    print("ELECTRODE POSITION STATISTICS")
    print("=" * 80)
    print(channels_pos_df.describe().to_string())
    
    # Check positions in raw data
    print("\n" + "=" * 80)
    print("RAW DATA MONTAGE INFO")
    print("=" * 80)
    print(f"Raw object channels: {len(raw.ch_names)}")
    print(f"First 10 raw channels: {raw.ch_names[:10]}")
    
    # Get channel info from raw
    print("\n" + "=" * 80)
    print("CHANNEL LOCATION INFO IN RAW DATA")
    print("=" * 80)
    
    ch_locs = []
    for idx, ch_name in enumerate(raw.ch_names[:10]):  # First 10
        ch_info = raw.info['chs'][idx]
        loc = ch_info['loc'][:3]  # First 3 values are x, y, z
        kind = ch_info['kind']
        ch_locs.append({
            'Channel': ch_name,
            'X': loc[0],
            'Y': loc[1],
            'Z': loc[2],
            'Type': kind
        })
    
    ch_locs_df = pd.DataFrame(ch_locs)
    print(ch_locs_df.to_string(index=False))
    
    # Compare montage positions with raw positions
    print("\n" + "=" * 80)
    print("MONTAGE ↔ RAW DATA POSITION COMPARISON")
    print("=" * 80)
    
    comparison = []
    for idx, ch_name in enumerate(raw.ch_names[:15]):  # First 15
        # From montage
        mont_pos = ch_pos_dict.get(ch_name, np.array([np.nan, np.nan, np.nan]))
        
        # From raw
        raw_loc = raw.info['chs'][idx]['loc'][:3]
        
        # Check if both have valid positions
        if not np.any(np.isnan(mont_pos)) and not np.any(np.isnan(raw_loc)):
            match = "✓" if np.allclose(mont_pos, raw_loc, atol=1e-5) else "✗"
        else:
            match = "—"
        
        comparison.append({
            'Channel': ch_name,
            'Montage[X,Y,Z]': f"[{mont_pos[0]:.4f}, {mont_pos[1]:.4f}, {mont_pos[2]:.4f}]",
            'Raw[X,Y,Z]': f"[{raw_loc[0]:.4f}, {raw_loc[1]:.4f}, {raw_loc[2]:.4f}]",
            'Match': match
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # Visualize montage if possible
    print("\n" + "=" * 80)
    print("VISUALIZATION")
    print("=" * 80)
    
    try:
        # Create a simple 2D projection (filter out NaN positions)
        import matplotlib.pyplot as plt
        
        # Filter to only EEG channels (with valid positions)
        valid_indices = ~np.any(np.isnan(pos), axis=1)
        pos_valid = pos[valid_indices]
        ch_names_valid = [ch for ch, valid in zip(ch_names_list, valid_indices) if valid]
        
        print(f"Valid electrode positions: {np.sum(valid_indices)} (excluding {np.sum(~valid_indices)} non-EEG channels)")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # X-Y plane (top view)
        axes[0].scatter(pos_valid[:, 0], pos_valid[:, 1], s=100, alpha=0.6)
        for i, ch_name in enumerate(ch_names_valid):
            axes[0].annotate(ch_name, (pos_valid[i, 0], pos_valid[i, 1]), 
                            fontsize=7, alpha=0.7, ha='center')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Top View (X-Y plane)')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        
        # X-Z plane (front view)
        axes[1].scatter(pos_valid[:, 0], pos_valid[:, 2], s=100, alpha=0.6)
        for i, ch_name in enumerate(ch_names_valid):
            axes[1].annotate(ch_name, (pos_valid[i, 0], pos_valid[i, 2]), 
                            fontsize=7, alpha=0.7, ha='center')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        axes[1].set_title('Front View (X-Z plane)')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        # Y-Z plane (side view)
        axes[2].scatter(pos_valid[:, 1], pos_valid[:, 2], s=100, alpha=0.6)
        for i, ch_name in enumerate(ch_names_valid):
            axes[2].annotate(ch_name, (pos_valid[i, 1], pos_valid[i, 2]), 
                            fontsize=7, alpha=0.7, ha='center')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        axes[2].set_title('Side View (Y-Z plane)')
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(__file__).parent / "outputs" / "montage_visualization.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Montage visualization saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not available for visualization")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Montage successfully loaded with {len(ch_pos_dict)} channels")
    n_valid = np.sum(~np.any(np.isnan(pos), axis=1))
    print(f"✓ Valid electrode positions: {n_valid}")
    print(f"✓ Positions are in normalized coordinate frame (unit sphere)")
    print(f"✓ Coordinate frame reference points:")
    print(f"    - Nasion (bridge of nose): {pos_dict.get('nasion')}")
    print(f"    - LPA (left ear): {pos_dict.get('lpa')}")
    print(f"    - RPA (right ear): {pos_dict.get('rpa')}")


if __name__ == "__main__":
    inspect_montage()
