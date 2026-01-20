"""
REVE Model Module

Simple, modular wrapper for HuggingFace REVE models.
- Downloads and caches models locally in models/ folder
"""

from .loader import load_reve_model, load_reve_positions, load_reve_models
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
]
