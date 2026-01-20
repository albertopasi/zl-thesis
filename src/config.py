"""
Project Configuration

Central place for all project-wide settings.
"""

import torch
from pathlib import Path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache directory
MODELS_CACHE_DIR = Path(__file__).parent.parent / "models" / "pretrained"

# PyTorch dtype
TORCH_DTYPE = "auto"

# Trust remote code from HuggingFace (required for REVE)
TRUST_REMOTE_CODE = True

# REVE Model configuration
REVE_MODEL_SIZE = "base"  # Options: "base" or "large"
# LoRA (Low-Rank Adaptation) configuration for fine-tuning
LORA_RANK = 16  # LoRA rank (r) - lower values = more efficient, typical 4-16
LORA_ALPHA = 32  # LoRA scaling factor - typically 2*rank
LORA_DROPOUT = 0.05  # Dropout in LoRA layers
LORA_BIAS = "none"  # Options: "none", "all", "lora_only"

# Optimizer configuration
OPTIMIZER_LR = 1e-4  # Learning rate
OPTIMIZER_WEIGHT_DECAY = 0.01  # Weight decay for regularization
