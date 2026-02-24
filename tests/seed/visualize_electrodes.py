"""Visualize SEED electrode positions."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.preprocess_seed.seed_loader import SEEDEEGLoader
from src.preprocess_seed.seed_preprocessing_config import SEEDPreprocessingConfig


def visualize_montage():
    """Create 2D and 3D visualizations of electrode positions."""
    config = SEEDPreprocessingConfig()
    loader = SEEDEEGLoader(config.seed_raw_dir, config.montage_file)
    
    # Get montage positions
    montage = loader.montage
    pos = montage.get_positions()
    ch_pos = pos['ch_pos']
    
    print("Creating electrode position visualizations...")
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))
    
    # Extract valid positions
    valid_channels = []
    x_coords = []
    y_coords = []
    z_coords = []
    
    for ch_name in sorted(ch_pos.keys()):
        pos_3d = ch_pos[ch_name]
        if pos_3d is not None and not np.allclose(pos_3d, 0):
            valid_channels.append(ch_name)
            x_coords.append(pos_3d[0])
            y_coords.append(pos_3d[1])
            z_coords.append(pos_3d[2])
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)
    
    # 2D Top View (X-Y)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(x_coords, y_coords, s=100, alpha=0.7, c='blue', edgecolors='black', linewidth=1)
    for i, ch_name in enumerate(valid_channels):
        ax1.annotate(ch_name, (x_coords[i], y_coords[i]), 
                    fontsize=6, ha='center', va='center', alpha=0.7)
    ax1.set_xlabel('X (Left-Right)', fontsize=10)
    ax1.set_ylabel('Y (Front-Back)', fontsize=10)
    ax1.set_title('Top View (X-Y Plane)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2D Front View (X-Z)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(x_coords, z_coords, s=100, alpha=0.7, c='green', edgecolors='black', linewidth=1)
    for i, ch_name in enumerate(valid_channels):
        ax2.annotate(ch_name, (x_coords[i], z_coords[i]), 
                    fontsize=6, ha='center', va='center', alpha=0.7)
    ax2.set_xlabel('X (Left-Right)', fontsize=10)
    ax2.set_ylabel('Z (Up-Down)', fontsize=10)
    ax2.set_title('Front View (X-Z Plane)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 2D Side View (Y-Z)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(y_coords, z_coords, s=100, alpha=0.7, c='red', edgecolors='black', linewidth=1)
    for i, ch_name in enumerate(valid_channels):
        ax3.annotate(ch_name, (y_coords[i], z_coords[i]), 
                    fontsize=6, ha='center', va='center', alpha=0.7)
    ax3.set_xlabel('Y (Front-Back)', fontsize=10)
    ax3.set_ylabel('Z (Up-Down)', fontsize=10)
    ax3.set_title('Side View (Y-Z Plane)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 3D View
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(x_coords, y_coords, z_coords, s=80, alpha=0.7, c='purple', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_zlabel('Z', fontsize=10)
    ax4.set_title('3D View', fontsize=12, fontweight='bold')
    
    # Color by region (rough division)
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Define regions based on position
    colors = []
    for i, ch_name in enumerate(valid_channels):
        if ch_name.startswith(('Fp', 'AF', 'F')):
            colors.append('red')  # Frontal
        elif ch_name.startswith(('T', 'C')):
            colors.append('blue')  # Central/Temporal
        elif ch_name.startswith(('P', 'PO')):
            colors.append('green')  # Parietal
        elif ch_name.startswith(('O', 'CB')):
            colors.append('orange')  # Occipital
        else:
            colors.append('gray')
    
    ax5.scatter(x_coords, y_coords, s=100, alpha=0.7, c=colors, edgecolors='black', linewidth=1)
    ax5.set_xlabel('X (Left-Right)', fontsize=10)
    ax5.set_ylabel('Y (Front-Back)', fontsize=10)
    ax5.set_title('Channels by Region', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Frontal'),
        Patch(facecolor='blue', edgecolor='black', label='Central/Temporal'),
        Patch(facecolor='green', edgecolor='black', label='Parietal'),
        Patch(facecolor='orange', edgecolor='black', label='Occipital'),
    ]
    ax5.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
MONTAGE STATISTICS
{'='*35}

Total Channels: {len(loader.channels)}
Valid Positions: {len(valid_channels)}

Coordinate Ranges:
  X: [{x_coords.min():.3f}, {x_coords.max():.3f}]
  Y: [{y_coords.min():.3f}, {y_coords.max():.3f}]
  Z: [{z_coords.min():.3f}, {z_coords.max():.3f}]

Channel Distribution:
  Frontal: {sum(1 for ch in valid_channels if ch.startswith(('Fp', 'AF', 'F')))}
  Central/Temporal: {sum(1 for ch in valid_channels if ch.startswith(('T', 'C')))}
  Parietal: {sum(1 for ch in valid_channels if ch.startswith(('P', 'PO')))}
  Occipital: {sum(1 for ch in valid_channels if ch.startswith(('O', 'CB')))}
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "seed_electrode_positions.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Also save individual high-quality plots
    fig_top = plt.figure(figsize=(10, 8))
    ax = fig_top.add_subplot(111)
    ax.scatter(x_coords, y_coords, s=150, alpha=0.8, c='darkblue', edgecolors='black', linewidth=1.5)
    for i, ch_name in enumerate(valid_channels):
        ax.annotate(ch_name, (x_coords[i], y_coords[i]), 
                   fontsize=7, ha='center', va='center', weight='bold')
    ax.set_xlabel('X (Left ← → Right)', fontsize=12, weight='bold')
    ax.set_ylabel('Y (Back ← → Front)', fontsize=12, weight='bold')
    ax.set_title('SEED Electrode Layout - Top View', fontsize=14, weight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    top_view_path = output_dir / "seed_electrodes_topview.png"
    fig_top.savefig(top_view_path, dpi=200, bbox_inches='tight')
    print(f"✓ Top view saved to: {top_view_path}")
    
    plt.show()
    
    print(f"\nTotal channels visualized: {len(valid_channels)}")


if __name__ == "__main__":
    visualize_montage()
