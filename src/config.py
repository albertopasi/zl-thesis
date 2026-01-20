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

# ============================================================================
# PROJECTION HEAD CONFIGURATION
# ============================================================================

# REVE feature dimensions by model size
REVE_FEATURE_DIMS = {
    "base": 393216,      # REVE-base output dimension
    "large": 786432,     # REVE-large output dimension (approx)
}

PROJECTION_HEAD_INPUT_DIM = REVE_FEATURE_DIMS[REVE_MODEL_SIZE]
PROJECTION_HEAD_OUTPUT_DIM = 512  # Reduced representation dimension
PROJECTION_HEAD_HIDDEN_DIMS = [2048, 1024, 512]  # Progressive reduction
PROJECTION_HEAD_USE_BATCH_NORM = True
PROJECTION_HEAD_USE_LAYER_NORM = False
PROJECTION_HEAD_ACTIVATION = "relu"
PROJECTION_HEAD_DROPOUT = 0.1
PROJECTION_HEAD_USE_PROJECTION_LOSS = False  # Contrastive loss during fine-tuning

# ============================================================================
# DOWNSTREAM CLASSIFIER CONFIGURATION
# ============================================================================

CLASSIFIER_INPUT_DIM = PROJECTION_HEAD_OUTPUT_DIM  # Matches projection head output
CLASSIFIER_NUM_CLASSES = 2  # Binary: task vs no-task (workload classification)
CLASSIFIER_HIDDEN_DIMS = [256, 128]  # Hidden layer architecture
CLASSIFIER_USE_BATCH_NORM = True
CLASSIFIER_USE_LAYER_NORM = False
CLASSIFIER_ACTIVATION = "relu"
CLASSIFIER_DROPOUT = 0.2
CLASSIFIER_USE_CLASS_WEIGHTS = False  # Weight by class imbalance
CLASSIFIER_LABEL_SMOOTHING = 0.0  # Label smoothing (0.0 = disabled)
CLASSIFIER_FOCAL_LOSS_GAMMA = 0.0  # Focal loss gamma (0.0 = disabled)
CLASSIFIER_TASK_NAME = "workload_classification"
