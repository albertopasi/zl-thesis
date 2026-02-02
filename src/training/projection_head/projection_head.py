"""
Projection Head Implementation

Neural network module for projecting high-dimensional REVE features
into a lower-dimensional space suitable for contrastive loss computation.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .config import ProjectionHeadConfig


class ProjectionHead(nn.Module):
    """
    Non-linear projection head for Supervised Contrastive Learning (SupCon).
    
    Acts as an "information buffer" to absorb invariance constraints from SupCon loss
    while preserving rich representations in the backbone for downstream classification.
    
    Transforms REVE backbone embeddings (h, D_enc=512) into normalized contrastive
    embeddings (z, typically 128-256 dims) on the unit hypersphere.
    
    Architecture:
    - Non-linear MLP with ReLU/GELU activations (critical for preserving information)
    - Optional batch normalization and layer normalization
    - Dropout for regularization
    - L2 normalization in forward pass for Cosine Similarity metric
    
    Why non-linearity matters:
    Without the non-linear transformations, the head acts as a passive linear projection.
    The non-linearity allows the head to learn task-relevant transformations that make
    the SupCon loss effective without destroying features needed by the classifier.
    """
    
    def __init__(self, config: ProjectionHeadConfig):
        """
        Initialize projection head.
        
        Args:
            config: ProjectionHeadConfig instance with architecture parameters
        """
        super().__init__()
        self.config = config
        
        # Get activation function
        activation_fn = self._get_activation(config.activation)
        
        # Build layer stack
        layers = []
        dims = [config.input_dim] + config.hidden_dims
        
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Normalization (if not the last layer)
            if i < len(dims) - 2:
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                elif config.use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                
                # Activation
                layers.append(activation_fn)
                
                # Dropout
                if config.dropout_rate > 0:
                    layers.append(nn.Dropout(config.dropout_rate))
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to lower-dimensional space and normalize to unit hypersphere.
        
        CRITICAL: L2 normalization ensures the embedding lies on the unit hypersphere,
        enabling Cosine Similarity computation required for SupCon loss.        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            L2-normalized embedding tensor of shape (batch_size, output_dim)
            with ||z||_2 = 1 for proper Cosine Similarity metric
        """
        z = self.projection(x)
        # L2 normalization to unit hypersphere for Cosine Similarity
        return torch.nn.functional.normalize(z, p=2, dim=1)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
        }
        
        if activation not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(activations.keys())}"
            )
        
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the projection head."""
        return self.config.output_dim


def create_projection_head(
    input_dim: int = 512,
    output_dim: int = 128,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> ProjectionHead:
    """
    Factory function to create a SupCon-aligned projection head.
    
    Args:
        input_dim: REVE backbone embedding dimension (default: 512 = D_enc)
        output_dim: Contrastive embedding dimension (default: 128 for compression)
        hidden_dims: List of intermediate layer dimensions. If None, defaults to [512, output_dim]
        **kwargs: Additional config parameters (use_batch_norm, dropout_rate, activation, etc.)
    
    Returns:
        Initialized ProjectionHead instance with L2 normalization enabled
        
    Example:
        head = create_projection_head(input_dim=512, output_dim=256)
        z = head(h)  # h is REVE backbone output, z is normalized contrastive embedding
    """
    config = ProjectionHeadConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )
    return ProjectionHead(config)
