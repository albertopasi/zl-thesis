"""
Data loading and preprocessing utilities for EEG classification.
"""

from functools import partial
import torch
from datasets import load_dataset
from config import BATCH_SIZE, DATASET_ID


def collate_batch(batch, positions):
    """
    Custom collate function to prepare batches of data.
    
    Args:
        batch: List of samples from the dataset
        positions: Position embeddings for EEG electrodes
        
    Returns:
        dict: Dictionary with 'sample', 'label', and 'pos' tensors
    """
    x_data = torch.stack([x["data"] for x in batch])
    y_label = torch.tensor([x["labels"] for x in batch])
    positions = positions.repeat(len(batch), 1, 1)
    
    return {
        "sample": x_data,
        "label": y_label.long(),
        "pos": positions
    }


def load_eeg_dataset(batch_size, positions):
    """
    Load the EEGMAT dataset from Hugging Face.
    
    Args:
        batch_size: Batch size for data loaders
        positions: Position embeddings for EEG electrodes
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"Loading dataset from {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)
    dataset.set_format("torch", columns=["data", "labels"])
    
    print("Dataset loaded. Creating data loaders...")
    
    # Create collate function with positions
    collate_fn = partial(collate_batch, positions=positions)
    
    # Create data loaders for each split
    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset["val"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
