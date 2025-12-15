"""
Model initialization and setup for REVE-based EEG classification.
"""

import torch
from transformers import AutoModel

# Import from local config in this directory
from config import (
    REVE_MODEL_ID,
    REVE_POSITIONS_MODEL_ID,
    NUM_CHANNELS,
    SAMPLE_LENGTH,
    HIDDEN_DIM,
)

# 96 channels
STANDARD_POSITIONS_96 = [ ]


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


def setup_model(model, pos_bank, num_classes=None, eeg_positions=None):
    """
    Configure the REVE model with a classification head.
    Note: This keeps the original REVE model architecture intact.
    For classification, you may need to add a separate classification head layer.
    
    Args:
        model: The REVE model
        pos_bank: The position bank for electrode positions
        num_classes: Number of output classes (if None, uses config value)
        eeg_positions: List of channel position labels (if None, uses standard 10-20 system)
        
    Returns:
        tuple: (model, positions) - Model and position embeddings
    """
    if num_classes is None:
        from config import NUM_CLASSES
        num_classes = NUM_CLASSES
    
    # Use standard positions if none provided
    if eeg_positions is None or len(eeg_positions) == 0 or all(isinstance(ch, str) and ch.startswith('CH') for ch in eeg_positions):
        # Use standard 10-20 system positions
        print(f"Using standard 10-20 electrode positions for {NUM_CHANNELS} channels...")
        if NUM_CHANNELS <= len(STANDARD_POSITIONS_96):
            eeg_positions = STANDARD_POSITIONS_96[:NUM_CHANNELS]
        else:
            print(f"Warning: Requested {NUM_CHANNELS} channels but only {len(STANDARD_POSITIONS_96)} standard positions available")
            eeg_positions = STANDARD_POSITIONS_96 + [f"CH{i}" for i in range(NUM_CHANNELS - len(STANDARD_POSITIONS_96))]
    
    # Get position embeddings from the position bank
    try:
        print(f"Loading position embeddings for: {eeg_positions[:3]}... ({len(eeg_positions)} positions total)")
        positions = pos_bank(eeg_positions)
        print(f"Position embeddings loaded successfully. Shape: {positions.shape}")
    except Exception as e:
        print(f"Warning: Could not load position embeddings from position bank ({e})")
        print(f"Generating random position embeddings for {NUM_CHANNELS} channels...")
        # Fallback: generate random position embeddings
        positions = torch.randn(NUM_CHANNELS, 768)
    
    # Keep REVE model unchanged - it already has a final_layer
    # Users can adapt the output dimension as needed
    print("REVE model kept unchanged (original architecture preserved)")
    
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
