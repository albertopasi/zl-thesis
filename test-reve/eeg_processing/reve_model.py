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


def setup_model(model, pos_bank, num_classes=None, electrode_coordinates=None, eeg_positions=None, channel_indices=None):
    """
    Configure the REVE model with actual electrode coordinates.
    
    REVE's 4D Positional Encoding:
    ==============================
    The REVE model uses a 4D positional encoding that combines:
      1. Temporal dimension (timestep t)
      2. Spatial 3D coordinates (x, y, z from actual electrode positions)
    
    This design allows REVE to handle ARBITRARY electrode configurations

    Args:
        model: REVE model
        pos_bank: The position bank (not used when actual coordinates provided)
        num_classes: Number of output classes (if None, uses config value)
        electrode_coordinates: np.ndarray of shape (total_electrodes, 3) with actual 3D coordinates
                              These are typically all 96 electrodes from CapTrak system
        eeg_positions: List of channel position labels (fallback if coordinates not provided)
        channel_indices: List of indices to select specific channels from electrode_coordinates
                        If None, uses all coordinates up to NUM_CHANNELS
        
    Returns:
        tuple: (model, positions) - Model and position tensor of shape (num_used_channels, 3)
    """
    if num_classes is None:
        from config import NUM_CLASSES
        num_classes = NUM_CLASSES
    
    # Use actual 3D coordinates if provided (RECOMMENDED for arbitrary configurations)
    if electrode_coordinates is not None:
        # Handle the case where we have all electrodes but only use a subset
        if channel_indices is not None:
            # Use only the specified channel indices
            positions_array = electrode_coordinates[channel_indices]
            print(f"Using selected {len(channel_indices)} electrode coordinates from {electrode_coordinates.shape[0]} total")
        else:
            # Use the first NUM_CHANNELS electrodes
            from config import NUM_CHANNELS
            positions_array = electrode_coordinates[:NUM_CHANNELS]
            print(f"Using first {NUM_CHANNELS} electrode coordinates from {electrode_coordinates.shape[0]} total")
        
        # Convert to tensor and validate shape
        positions = torch.from_numpy(positions_array).float()
        
        if positions.shape[1] != 3:
            print(f"ERROR: Expected coordinates shape (num_channels, 3), got {positions.shape}")
            return model, None
        
        print(f"\n=== REVE 4D Positional Encoding (Arbitrary Electrode Support) ===")
        print(f"Position tensor shape: {positions.shape}")
        print(f"Utilizing actual 3D coordinates with temporal information:")
        print(f"  X (lateral):          [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
        print(f"  Y (anterior-post):    [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] mm")
        print(f"  Z (vertical):         [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] mm")
        print(f"This flexible approach enables handling of arbitrary electrode montages.")
        print()
        
        return model, positions
    
    # Fallback to position bank embeddings if no coordinates provided
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
        print(f"Loading position embeddings from pos_bank for: {eeg_positions[:3]}... ({len(eeg_positions)} positions total)")
        positions = pos_bank(eeg_positions)
        print(f"Position embeddings loaded from pos_bank. Shape: {positions.shape}")
    except Exception as e:
        print(f"Warning: Could not load position embeddings from position bank ({e})")
        print(f"Generating random position embeddings for {NUM_CHANNELS} channels...")
        # Fallback: generate random position embeddings
        positions = torch.randn(NUM_CHANNELS, 3)
    
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
