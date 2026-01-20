"""
REVE Model Setup and Configuration

Utilities for configuring loaded REVE models with electrode coordinates.
"""

import torch
import numpy as np
from typing import Optional


def setup_reve_model(
    electrode_coordinates: Optional[np.ndarray] = None,
    channel_indices: Optional[list] = None,
) -> torch.Tensor:
    """
    Configure REVE model with actual electrode coordinates.
    
    REVE's 4D Positional Encoding combines:
    - Temporal dimension (timestep t)
    - Spatial 3D coordinates (x, y, z from actual electrode positions)
    
    This allows REVE to handle arbitrary electrode configurations.
    
    Args:
        model: Loaded REVE model from loader.py
        pos_bank: Loaded REVE positions model from loader.py
        electrode_coordinates: (num_channels, 3) array with 3D coordinates
        channel_indices: List of channel indices to select (optional)
    
    Returns:
        positions: (num_channels, 3) tensor of electrode positions
    """
    print(f"\n{'='*60}")
    print("Setting up REVE model with electrode coordinates")
    print(f"{'='*60}")
    
    if electrode_coordinates is None:
        raise ValueError("electrode_coordinates required")
    
    # Select channels if indices provided
    if channel_indices is not None:
        positions_array = electrode_coordinates[channel_indices]
        print(f"Using {len(channel_indices)} selected channels from {electrode_coordinates.shape[0]} total")
    else:
        positions_array = electrode_coordinates
        print(f"Using all {len(positions_array)} electrode coordinates")
    
    # Convert to tensor
    positions = torch.from_numpy(positions_array).float()
    
    if positions.shape[1] != 3:
        raise ValueError(f"Expected (num_channels, 3), got {positions.shape}")
    
    # Print stats
    print(f"\nPosition tensor shape: {positions.shape}")
    print(f"Coordinate ranges:")
    print(f"  X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
    print(f"  Y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] mm")
    print(f"  Z: [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] mm")
    
    return positions
