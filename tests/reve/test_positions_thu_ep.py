"""
Test REVE position bank with THU-EP channel labels.

This script:
1. Loads channel labels from THU-EP label.mat
2. Checks if they exist in REVE position bank
3. Prints 3D positions
4. Plots positions in 3D and 2D from different angles

Usage:
    python -m tests.reve.test_positions_thu_ep
    uv run python -m tests.reve.test_positions_thu_ep
"""

from pathlib import Path

import numpy as np
import h5py
import torch
from transformers import AutoModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "reve_pretrained_original"
THU_EP_LABELS_FILE = Path(__file__).parent.parent.parent / "data" / "thu ep" / "Others" / "label.mat"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_thu_ep_channel_names() -> list[str]:
    """Load channel names from THU-EP label.mat."""
    strings = []
    with h5py.File(str(THU_EP_LABELS_FILE), 'r') as f:
        dataset = f['label']
        for ref in dataset[0]:
            dereferenced = f[ref]
            chars = np.array(dereferenced).flatten()
            string = ''.join(chr(c) for c in chars)
            strings.append(string)
    return strings


def load_position_bank():
    """Load REVE position bank."""
    model_path = MODELS_DIR / "reve-positions"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Position bank not found at {model_path}. "
            "Run: uv run python -m src.download_reve.download_models"
        )
    
    pos_bank = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    return pos_bank


def test_position_correspondence():
    """Test if THU-EP channels exist in REVE position bank."""
    print("=" * 70)
    print("REVE POSITION BANK vs THU-EP CHANNELS")
    print("=" * 70)
    
    # Load THU-EP channel names
    print("\n1. Loading THU-EP channel names...")
    thu_ep_channels = load_thu_ep_channel_names()
    print(f"   Found {len(thu_ep_channels)} channels: {thu_ep_channels}")
    
    # Load position bank
    print("\n2. Loading REVE position bank...")
    pos_bank = load_position_bank()
    
    # Get all registered positions
    all_reve_positions = pos_bank.get_all_positions()
    print(f"   Position bank has {len(all_reve_positions)} registered positions")
    
    # Check correspondence
    print("\n3. Checking channel correspondence...")
    found_channels = []
    missing_channels = []
    
    for ch in thu_ep_channels:
        if ch in all_reve_positions:
            found_channels.append(ch)
        else:
            missing_channels.append(ch)
    
    print(f"   Found in REVE: {len(found_channels)}/{len(thu_ep_channels)}")
    print(f"   Missing: {missing_channels if missing_channels else 'None'}")
    
    # Get 3D positions for found channels
    print("\n4. Getting 3D positions for matching channels...")
    if found_channels:
        positions = pos_bank(found_channels)
        print(f"   Position tensor shape: {positions.shape}")
        
        # Print positions
        print("\n   Channel positions (x, y, z):")
        print("   " + "-" * 50)
        positions_np = positions.detach().numpy()
        for i, ch in enumerate(found_channels):
            x, y, z = positions_np[i]
            print(f"   {ch:5s}: ({x:7.4f}, {y:7.4f}, {z:7.4f})")
        
        return found_channels, positions_np, missing_channels, pos_bank
    
    return found_channels, None, missing_channels, pos_bank


def plot_positions(channels: list[str], positions: np.ndarray):
    """Plot electrode positions in 3D and 2D views."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(x, y, z, c='blue', s=100, alpha=0.7)
    for i, ch in enumerate(channels):
        ax1.text(x[i], y[i], z[i], f' {ch}', fontsize=8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')
    
    # Top view (X-Y plane, looking down Z axis)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(x, y, c='blue', s=100, alpha=0.7)
    for i, ch in enumerate(channels):
        ax2.annotate(ch, (x[i], y[i]), fontsize=8, ha='center', va='bottom')
    ax2.set_xlabel('X (left-right)')
    ax2.set_ylabel('Y (back-front)')
    ax2.set_title('Top View (looking down)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Front view (X-Z plane, looking from front)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(x, z, c='blue', s=100, alpha=0.7)
    for i, ch in enumerate(channels):
        ax3.annotate(ch, (x[i], z[i]), fontsize=8, ha='center', va='bottom')
    ax3.set_xlabel('X (left-right)')
    ax3.set_ylabel('Z (bottom-top)')
    ax3.set_title('Front View')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Side view (Y-Z plane, looking from side)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(y, z, c='blue', s=100, alpha=0.7)
    for i, ch in enumerate(channels):
        ax4.annotate(ch, (y[i], z[i]), fontsize=8, ha='center', va='bottom')
    ax4.set_xlabel('Y (back-front)')
    ax4.set_ylabel('Z (bottom-top)')
    ax4.set_title('Side View (from right)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "thu_ep_positions_reve.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def plot_positions_detailed_3d(channels: list[str], positions: np.ndarray):
    """Create a more detailed 3D plot with multiple angles."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    # Create figure with 3D views from different angles
    fig = plt.figure(figsize=(16, 8))
    
    angles = [
        (30, 45, "Default View"),
        (90, 0, "Top View"),
        (0, 0, "Front View"),
        (0, 90, "Right Side View"),
    ]
    
    for idx, (elev, azim, title) in enumerate(angles, 1):
        ax = fig.add_subplot(1, 4, idx, projection='3d')
        ax.scatter(x, y, z, c='blue', s=80, alpha=0.7)
        
        for i, ch in enumerate(channels):
            ax.text(x[i], y[i], z[i], f' {ch}', fontsize=6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "thu_ep_positions_3d_angles.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"3D angles plot saved to: {output_path}")
    
    plt.show()


def plot_with_additional_channels(channels: list[str], positions: np.ndarray, pos_bank):
    """Plot THU-EP channels (blue) with additional T3, T4, T5, T6 channels (red)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Additional channels to show in red
    additional_channels = ['T3', 'T4', 'T5', 'T6']
    
    # Get positions for additional channels
    additional_positions = pos_bank(additional_channels).detach().numpy()
    
    print(f"\n   Additional channels (in red): {additional_channels}")
    print("   " + "-" * 50)
    for i, ch in enumerate(additional_channels):
        x, y, z = additional_positions[i]
        print(f"   {ch:5s}: ({x:7.4f}, {y:7.4f}, {z:7.4f})")
    
    # Combine data
    x_blue, y_blue, z_blue = positions[:, 0], positions[:, 1], positions[:, 2]
    x_red, y_red, z_red = additional_positions[:, 0], additional_positions[:, 1], additional_positions[:, 2]
    
    # Create figure with 3D views from different angles
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('THU-EP Channels (blue) + T3, T4, T5, T6 (red)', fontsize=14)
    
    angles = [
        (30, 45, "Default View"),
        (90, 0, "Top View"),
        (0, 0, "Front View"),
        (0, 90, "Right Side View"),
    ]
    
    for idx, (elev, azim, title) in enumerate(angles, 1):
        ax = fig.add_subplot(1, 4, idx, projection='3d')
        
        # Plot THU-EP channels in blue
        ax.scatter(x_blue, y_blue, z_blue, c='blue', s=80, alpha=0.7, label='THU-EP')
        for i, ch in enumerate(channels):
            ax.text(x_blue[i], y_blue[i], z_blue[i], f' {ch}', fontsize=6, color='blue')
        
        # Plot additional channels in red
        ax.scatter(x_red, y_red, z_red, c='red', s=120, alpha=0.9, marker='^', label='T3/T4/T5/T6')
        for i, ch in enumerate(additional_channels):
            ax.text(x_red[i], y_red[i], z_red[i], f' {ch}', fontsize=8, color='red', fontweight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        
        if idx == 1:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "thu_ep_with_T3T4T5T6.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot with additional channels saved to: {output_path}")
    
    plt.show()


def run_test():
    """Run the full position test."""
    found_channels, positions, missing_channels, pos_bank = test_position_correspondence()
    
    if positions is not None and len(found_channels) > 0:
        print("\n5. Plotting positions...")
        plot_positions(found_channels, positions)
        plot_positions_detailed_3d(found_channels, positions)
        
        # Plot with additional T3, T4, T5, T6 channels in red
        print("\n6. Plotting with additional channels (T3, T4, T5, T6 in red)...")
        plot_with_additional_channels(found_channels, positions, pos_bank)
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"THU-EP channels: {len(found_channels) + len(missing_channels)}")
        print(f"Found in REVE position bank: {len(found_channels)}")
        print(f"Missing from REVE: {len(missing_channels)}")
        if missing_channels:
            print(f"Missing channels: {missing_channels}")
            print("\nNote: Missing channels may use different naming conventions.")
            print("Check REVE position bank for alternative names.")
    else:
        print("\nNo positions to plot - no matching channels found.")


if __name__ == "__main__":
    run_test()
