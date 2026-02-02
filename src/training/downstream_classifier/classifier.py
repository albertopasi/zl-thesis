"""
Downstream Classifier Implementation

Neural network module for classification tasks on REVE extracted features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from .config import ClassifierConfig


class DownstreamClassifier(nn.Module):
    """
    Classification head for downstream tasks on REVE extracted features.
    
    Architecture:
    - Configurable hidden layers
    - Optional batch normalization and layer normalization
    - Dropout for regularization
    - Flexible activation functions
    - Support for class weights and focal loss
    """
    
    def __init__(self, config: ClassifierConfig):
        """
        Initialize downstream classifier.
        
        Args:
            config: ClassifierConfig instance with architecture parameters
        """
        super().__init__()
        self.config = config
        
        # Get activation function
        activation_fn = self._get_activation(config.activation)
        
        # Build hidden layer stack
        layers = []
        dims = [config.input_dim] + config.hidden_dims
        
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Normalization
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            # Activation
            layers.append(activation_fn)
            
            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (classification)
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify projected features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        return logits
    
    def forward_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning intermediate features and logits.
        
        Useful for analysis, visualization, and feature extraction.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Dictionary with:
                - 'hidden': Hidden layer outputs (batch_size, hidden_dims[-1])
                - 'logits': Classification logits (batch_size, num_classes)
                - 'probs': Softmax probabilities (batch_size, num_classes)
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        probs = F.softmax(logits, dim=1)
        
        return {
            'hidden': hidden,
            'logits': logits,
            'probs': probs
        }
    
    def predict(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Get class predictions from input features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_probs: If True, return probabilities instead of class indices
        
        Returns:
            Predictions: class indices (batch_size,) or probabilities (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            if return_probs:
                return F.softmax(logits, dim=1)
            else:
                return torch.argmax(logits, dim=1)
    
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
    
    def get_num_classes(self) -> int:
        """Get the number of output classes."""
        return self.config.num_classes
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_classifier(
    input_dim: int = 512,
    num_classes: int = 2,
    hidden_dims: Optional[List[int]] = None,
    task_name: str = "workload_classification",
    **kwargs
) -> DownstreamClassifier:
    """
    Factory function to create a downstream classifier with custom parameters.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of intermediate layer dimensions
        task_name: Name of the downstream task
        **kwargs: Additional config parameters (dropout_rate, use_batch_norm, etc.)
    
    Returns:
        Initialized DownstreamClassifier instance
    """
    config = ClassifierConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        task_name=task_name,
        **kwargs
    )
    return DownstreamClassifier(config)
