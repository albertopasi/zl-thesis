"""
Downstream Classifier Configuration

Settings for the classification head that performs workload prediction
and other downstream tasks.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ClassifierConfig:
    """Configuration for downstream classifier architecture."""
    
    # Input/Output dimensions
    input_dim: int = 512  # Matches projection head output
    num_classes: int = 2  # Binary: task vs no-task (default)
    
    # Architecture
    hidden_dims: List[int] = None  # Intermediate layer dimensions
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    activation: str = "relu"  # relu, gelu, elu, silu
    dropout_rate: float = 0.2
    
    # Training and regularization
    use_class_weights: bool = False  # Weight loss by class frequency
    label_smoothing: float = 0.0  # Label smoothing factor (0.0 = disabled)
    focal_loss_gamma: float = 0.0  # Focal loss gamma parameter (0.0 = disabled)
    
    # Task-specific
    task_name: str = "workload_classification"  # For logging and identification
    
    def __post_init__(self):
        """Set default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            # Simple 2-layer architecture for classification
            self.hidden_dims = [256, 128]
