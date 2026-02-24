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

