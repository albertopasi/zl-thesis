"""
REVE Model Module

Simple, modular wrapper for HuggingFace REVE models.
- Downloads and caches models locally in models/ folder
"""

from .model_loader import load_reve_model, load_reve_positions, load_reve_models
from .lora import (
    apply_lora,
    save_lora_adapter,
    load_lora_adapter,
    merge_adapters,
    get_optimizer_groups,
)

__all__ = [
    "load_reve_model",
    "load_reve_positions",
    "load_reve_models",
    "apply_lora",
    "save_lora_adapter",
    "load_lora_adapter",
    "merge_adapters",
    "get_optimizer_groups",
    "ProjectionClassifierPipeline",
    "create_pipeline",
    "create_train_test_dataloaders",
    "compute_class_weights",
    "get_loss_function",
    "get_optimizer",
    "train_epoch",
    "evaluate",
    "train_pipeline",
    "apply_lora",
    "save_lora_adapter",
    "load_lora_adapter",
    "merge_adapters",
    "get_optimizer_groups",
]
