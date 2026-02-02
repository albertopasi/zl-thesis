"""
Projection Head Configuration

Settings for the projection head module that reduces REVE feature dimensionality.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ProjectionHeadConfig:
    """Configuration for projection head architecture."""
    
    # Input/Output dimensions
    input_dim: int = 512  # REVE backbone embedding dimension (D_enc)
    output_dim: int = 128  # Contrastive embedding dimension (compressing for SupCon)
    
    # Architecture
    hidden_dims: List[int] = None  # Intermediate layer dimensions
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    activation: str = "relu"  # relu, gelu, elu
    dropout_rate: float = 0.1
    
    # Training
    use_projection_loss: bool = False  # Whether to apply contrastive loss during fine-tuning
    temperature: float = 0.07  # Temperature for contrastive loss (if used)
    
    def __post_init__(self):
        """Set default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            # Non-linear MLP with information buffer (input -> 512 -> output)
            # Intermediate dim = input_dim preserves information before compression
            # Allows head to absorb invariance constraints without degrading backbone
            self.hidden_dims = [512, self.output_dim]
