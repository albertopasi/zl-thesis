"""
Test loading locally downloaded REVE models.

This script verifies that the downloaded REVE models can be loaded correctly
and prints model architecture, parameters, positions, etc.

Usage:
    python -m tests.reve.test_load_models
    uv run python -m tests.reve.test_load_models
"""

from pathlib import Path

import torch
from transformers import AutoModel


# Path to locally downloaded models
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "reve_pretrained_original"


def test_load_reve_base():
    """Load and inspect REVE base model."""
    print("=" * 70)
    print("REVE BASE MODEL")
    print("=" * 70)
    
    model_path = MODELS_DIR / "reve-base"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: uv run python -m src.download_reve.download_models --model reve-base")
        return None
    
    print(f"Loading from: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    
    # Model architecture
    print("\n--- Architecture ---")
    print(model)
    
    # Parameters
    print("\n--- Parameters ---")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Summary ---")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def test_load_reve_large():
    """Load and inspect REVE large model."""
    print("\n" + "=" * 70)
    print("REVE LARGE MODEL")
    print("=" * 70)
    
    model_path = MODELS_DIR / "reve-large"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: uv run python -m src.download_reve.download_models --model reve-large")
        return None
    
    print(f"Loading from: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    
    # Model architecture
    print("\n--- Architecture ---")
    print(model)
    
    # Parameters (just summary for large model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Summary ---")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Layer names only (not full shapes to keep output manageable)
    print("\n--- Layer names ---")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    return model


def test_load_position_bank():
    """Load and inspect REVE position bank."""
    print("\n" + "=" * 70)
    print("REVE POSITION BANK")
    print("=" * 70)
    
    model_path = MODELS_DIR / "reve-positions"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: uv run python -m src.download_reve.download_models --model reve-positions")
        return None
    
    print(f"Loading from: {model_path}")
    pos_bank = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    
    # Architecture
    print("\n--- Architecture ---")
    print(pos_bank)
    
    # Get all registered positions
    print("\n--- All Registered Positions ---")
    all_positions = pos_bank.get_all_positions()
    print(f"  Total positions: {len(all_positions)}")
    print(f"  Positions: {all_positions}")
    
    # Test getting specific electrode positions
    print("\n--- Test Electrode Lookup ---")
    test_electrodes = ["Fp1", "Fp2", "Fz", "Cz", "Pz", "O1", "O2"]
    try:
        positions = pos_bank(test_electrodes)
        print(f"  Electrodes: {test_electrodes}")
        print(f"  Position tensor shape: {positions.shape}")
        print(f"  Position values:\n{positions}")
    except Exception as e:
        print(f"  ERROR looking up electrodes: {e}")
    
    # Parameters
    total_params = sum(p.numel() for p in pos_bank.parameters())
    print(f"\n--- Summary ---")
    print(f"  Total parameters: {total_params:,}")
    
    return pos_bank


def test_model_forward_pass():
    """Test a forward pass through the model with dummy data."""
    print("\n" + "=" * 70)
    print("FORWARD PASS TEST")
    print("=" * 70)
    
    # Load models
    model_path = MODELS_DIR / "reve-base"
    pos_path = MODELS_DIR / "reve-positions"
    
    if not model_path.exists() or not pos_path.exists():
        print("ERROR: Models not found. Download them first.")
        return
    
    print("Loading models...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    pos_bank = AutoModel.from_pretrained(
        pos_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    
    # Create dummy input (similar to EEGMAT dataset format)
    # 20 channels, 5 seconds at 200Hz = 1000 samples
    batch_size = 2
    n_channels = 20
    n_samples = 1000  # 5s * 200Hz
    
    # Dummy EEG data
    dummy_eeg = torch.randn(batch_size, n_channels, n_samples)
    
    # Get positions for 20 standard electrodes
    electrodes = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4", 
                  "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A2"]
    positions = pos_bank(electrodes)
    positions = positions.repeat(batch_size, 1, 1)  # [B, n_channels, pos_dim]
    
    print(f"\n--- Input Shapes ---")
    print(f"  EEG data: {dummy_eeg.shape}")
    print(f"  Positions: {positions.shape}")
    
    # Forward pass
    print("\n--- Forward Pass ---")
    model.eval()
    with torch.inference_mode():
        output = model(dummy_eeg, positions)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Expected output for classification head setup
    # From notebook: [B, n_channels, n_seconds, hidden_dim] = [B, 20, 5, 512]
    print(f"\n--- Expected dimensions for classification ---")
    print(f"  For classifier head: flatten to {output.shape[0]} x {output[0].numel()} = {output.numel() // batch_size}")
    
    print("\nForward pass successful!")


def run_all_tests():
    """Run all model loading tests."""
    print("\n" + "=" * 70)
    print("REVE MODEL LOADING TESTS")
    print("=" * 70)
    print(f"Models directory: {MODELS_DIR.absolute()}")
    print("=" * 70)
    
    results = {}
    
    # Test base model
    try:
        model = test_load_reve_base()
        results["reve-base"] = "OK" if model is not None else "NOT FOUND"
        del model
    except Exception as e:
        results["reve-base"] = f"FAILED: {e}"
    
    # Test large model
    try:
        model = test_load_reve_large()
        results["reve-large"] = "OK" if model is not None else "NOT FOUND"
        del model
    except Exception as e:
        results["reve-large"] = f"FAILED: {e}"
    
    # Test position bank
    try:
        pos_bank = test_load_position_bank()
        results["reve-positions"] = "OK" if pos_bank is not None else "NOT FOUND"
        del pos_bank
    except Exception as e:
        results["reve-positions"] = f"FAILED: {e}"
    
    # Test forward pass
    try:
        test_model_forward_pass()
        results["forward-pass"] = "OK"
    except Exception as e:
        results["forward-pass"] = f"FAILED: {e}"
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {name}: {status}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_tests()
