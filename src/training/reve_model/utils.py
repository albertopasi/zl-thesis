"""
Utilities for model management and checkpointing
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any, Tuple


def save_checkpoint(
    model,
    checkpoint_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        checkpoint_path: Where to save checkpoint
        optimizer: Optional optimizer state
        epoch: Training epoch
        metrics: Optional metrics dictionary
        **kwargs: Additional data to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics or {},
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def load_checkpoint(
    model,
    checkpoint_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return checkpoint
