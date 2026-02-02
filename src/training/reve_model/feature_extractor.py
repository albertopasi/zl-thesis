"""
REVE Feature Extraction

Utilities for extracting EEG features using REVE model with arbitrary electrode coordinates.
"""

import torch
import numpy as np
from typing import Optional, Dict
import json
from pathlib import Path


def extract_eeg_features(
    model,
    eeg: np.ndarray,
    positions: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract features from EEG using REVE model
    
    Args:
        model: Loaded REVE model
        eeg: (num_channels, num_samples) or (batch_size, num_channels, num_samples)
        positions: (num_channels, 3) - 3D electrode coordinates
        device: Device to use
    
    Returns:
        features: (output_dim,) or (batch_size, output_dim) - Extracted features
    """
    model = model.to(device)
    model.eval()
    
    # Handle input shape
    if eeg.ndim == 2:
        eeg = eeg[np.newaxis, :, :]  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Normalize EEG
    eeg = (eeg - eeg.mean(axis=2, keepdims=True)) / (eeg.std(axis=2, keepdims=True) + 1e-6)
    
    # Convert to tensors
    eeg_tensor = torch.from_numpy(eeg).float().to(device)
    
    if positions.dim() == 2:
        positions = positions.unsqueeze(0).expand(eeg.shape[0], -1, -1)
    positions = positions.to(device)
    
    # Extract features
    with torch.no_grad():
        features = model(eeg_tensor, positions)
    
    features = features.cpu().numpy()
    
    if squeeze_output:
        features = features.squeeze(0)
    
    return features


def load_electrode_coordinates(filepath: Path) -> np.ndarray:
    """
    Load electrode coordinates from JSON file
    
    Args:
        filepath: Path to electrode coordinates JSON
    
    Returns:
        (num_electrodes, 3) - x, y, z coordinates in mm
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        # Format: {"electrode_name": [x, y, z], ...}
        coords = np.array([v for v in data.values() if isinstance(v, (list, tuple)) and len(v) == 3])
    else:
        # Format: [[x, y, z], [x, y, z], ...]
        coords = np.array(data)
    
    return coords.astype(np.float32)


def standardize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Standardize coordinates to zero mean, unit variance"""
    mean = coords.mean(axis=0)
    std = coords.std(axis=0) + 1e-6
    return ((coords - mean) / std).astype(np.float32)
