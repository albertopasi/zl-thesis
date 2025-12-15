"""
Model initialization and setup for REVE-based EEG classification.
"""

import torch
from transformers import AutoModel

# Import from local config in this directory
try:
    from config import (
        REVE_MODEL_ID,
        REVE_POSITIONS_MODEL_ID,
        NUM_CHANNELS,
        SAMPLE_LENGTH,
        HIDDEN_DIM,
        NUM_CLASSES,
        EEG_POSITIONS,
    )
except ImportError:
    # Fallback for when imported from elsewhere
    REVE_MODEL_ID = "brain-bzh/reve-base-model"
    REVE_POSITIONS_MODEL_ID = "brain-bzh/reve-positions-model"
    NUM_CHANNELS = 99
    SAMPLE_LENGTH = 512
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    EEG_POSITIONS = None


def load_reve_model():
    """
    Load the REVE base model and position bank from Hugging Face.
    
    Returns:
        tuple: (model, pos_bank) - The REVE model and position bank
    """
    print(f"Loading REVE model from {REVE_MODEL_ID}...")
    model = AutoModel.from_pretrained(
        REVE_MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    
    print(f"Loading position bank from {REVE_POSITIONS_MODEL_ID}...")
    pos_bank = AutoModel.from_pretrained(
        REVE_POSITIONS_MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    
    return model, pos_bank


def setup_model(model, pos_bank):
    """
    Configure the REVE model with a classification head.
    
    Args:
        model: The REVE model
        pos_bank: The position bank for electrode positions
        
    Returns:
        tuple: (model, positions) - Modified model and position embeddings
    """
    # Get position embeddings for the EEG electrodes
    positions = pos_bank(EEG_POSITIONS)
    
    # Calculate the flattened dimension of model output
    # Output shape: [B, NUM_CHANNELS, SAMPLE_LENGTH, HIDDEN_DIM]
    # Flattened: [B, NUM_CHANNELS * SAMPLE_LENGTH * HIDDEN_DIM]
    final_dim = NUM_CHANNELS * SAMPLE_LENGTH * HIDDEN_DIM
    
    # Replace the final layer with a classification head
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(final_dim),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(final_dim, NUM_CLASSES),
    )
    
    return model, positions


def get_device():
    """
    Get the appropriate device (GPU if available, else CPU).
    
    Returns:
        torch.device: The device to use for training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def freeze_backbone(model):
    """
    Freeze all model parameters except the final layer.
    
    Args:
        model: The REVE model with classification head
    """
    for name, param in model.named_parameters():
        if "final_layer" not in name:
            param.requires_grad = False
    
    print("Backbone frozen. Only final layer will be trained.")


def inspect_model(model):
    """
    Print model architecture and parameter information.
    
    Args:
        model: The model to inspect
    """
    print("\n=== Model Architecture ===")
    print(model)
    
    print("\n=== Model Parameters ===")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"{name}: {param.shape} ({num_params:,} params)")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
