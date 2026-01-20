"""
LoRA (Low-Rank Adaptation) for REVE Model

Uses the peft library for parameter-efficient fine-tuning.
All configuration from src/config.py - no hardcoded values.
"""

from peft import LoraConfig, get_peft_model, PeftModel
import torch
from typing import Optional, List
from pathlib import Path
from ..config import LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, OPTIMIZER_LR, OPTIMIZER_WEIGHT_DECAY


def get_lora_config(task_type: str = "FEATURE_EXTRACTION") -> LoraConfig:
    """
    Create LoRA configuration for REVE from src/config.py settings.
    
    REVE structure:
    - 22 transformer layers with attention (to_qkv, to_out)
    
    Returns:
        LoraConfig: Configuration using LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS
    """

    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_qkv", "to_out"],
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=task_type,
    )


def apply_lora(model) -> PeftModel:
    """
    Apply LoRA adapters to REVE model.
    
    Configuration from src/config.py:
    - LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS
    
    Args:
        model: Loaded REVE model
    
    Returns:
        PeftModel: Model with LoRA adapters
    """

    lora_config = get_lora_config()
    
    # Apply LoRA
    lora_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    _print_trainable_params(lora_model)
    
    return lora_model


def _print_trainable_params(model) -> None:
    """Print model parameter statistics."""

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print("LoRA Model Parameters:")
    print(f"{'='*60}")
    print(f"Trainable params:  {trainable_params:,}")
    print(f"All params:        {all_params:,}")
    print(f"Trainable %:       {trainable_params/all_params*100:.2f}%")
    print(f"{'='*60}\n")


def save_lora_adapter(
    model: PeftModel,
    save_path: Path,
) -> None:
    """
    Save LoRA adapter weights only (not base model).
    
    Args:
        model: PeftModel with LoRA adapters
        save_path: Where to save adapter weights
    
    Example:
        save_lora_adapter(lora_model, Path("checkpoints/adapter_weights"))
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_path)
    print(f"LoRA adapter saved to: {save_path}")


def load_lora_adapter(
    model,
    adapter_path: Path,
) -> PeftModel:
    """
    Load LoRA adapter weights onto base model.
    
    Args:
        model: Base model (must be same architecture as adapter was created from (REVE BASE))
        adapter_path: Path to saved adapter weights
    
    Returns:
        PeftModel: Model with loaded adapter
    
    Example:
        model = load_reve_model()
        model = load_lora_adapter(model, Path("checkpoints/adapter_weights"))
    """

    lora_model = PeftModel.from_pretrained(model, adapter_path)
    return lora_model


def freeze_base_model(model) -> None:
    """Freeze all base model parameters (only LoRA trainable)."""

    for param in model.base_model.parameters():
        param.requires_grad = False
    print("Base model parameters frozen (LoRA only trainable)")


def unfreeze_base_model(model) -> None:
    """Unfreeze all base model parameters."""

    for param in model.base_model.parameters():
        param.requires_grad = True
    print("Base model parameters unfrozen")


def merge_adapters(model: PeftModel) -> torch.nn.Module:
    """
    Merge LoRA adapters into base model weights.
    Useful for inference to avoid overhead.
    
    Args:
        model: PeftModel with LoRA adapters
    
    Returns:
        torch.nn.Module: Base model with merged weights
    
    Example:
        merged_model = merge_adapters(lora_model)
        # Use merged_model for inference without LoRA overhead
    """

    merged_model = model.merge_and_unload()
    print("LoRA adapters merged into base model")
    return merged_model


def get_optimizer_groups(model: PeftModel) -> List[dict]:
    """
    Get parameter groups for optimizer from config.
    
    Uses OPTIMIZER_LR and OPTIMIZER_WEIGHT_DECAY from src/config.py.
    Standard practice: weight decay on dense layers only, not on biases/norms.
    
    Args:
        model: PeftModel with LoRA
    
    Returns:
        List of parameter groups for torch.optim.AdamW
    
    Example:
        optimizer_groups = get_optimizer_groups(lora_model)
        optimizer = torch.optim.AdamW(optimizer_groups)
    """
    
    decay = set()
    no_decay = set()
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or "bias" in name or "norm" in name:
            no_decay.add(name)
        else:
            decay.add(name)
    
    return [
        {
            "params": [p for n, p in model.named_parameters() if n in decay],
            "weight_decay": OPTIMIZER_WEIGHT_DECAY,
            "lr": OPTIMIZER_LR,
        },
        {
            "params": [p for n, p in model.named_parameters() if n in no_decay],
            "weight_decay": 0.0,
            "lr": OPTIMIZER_LR,
        },
    ]