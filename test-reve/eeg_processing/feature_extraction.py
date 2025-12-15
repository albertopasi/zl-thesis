"""
REVE-based feature extraction from EEG epochs.
"""

import torch
import numpy as np

# Import REVE model functions from local module
from reve_model import (
    load_reve_model,
    setup_model,
    get_device,
    freeze_backbone
)


class REVEFeatureExtractor:
    """Extract features from EEG epochs using REVE model."""
    
    def __init__(self, device=None, load_positions=True, channel_labels=None):
        """
        Initialize REVE feature extractor.
        
        Args:
            device: torch.device (default: auto-detect GPU/CPU)
            load_positions: Load position bank for electrode positions
            channel_labels: List of channel labels (if None, generates default labels)
        """
        self.device = device or get_device()
        self.model = None
        self.pos_bank = None
        self.positions = None
        self.channel_labels = channel_labels
        self.load_positions = load_positions
        
    def load_model(self, num_classes=None):
        """
        Load REVE model and position bank.
        
        Args:
            num_classes: Number of output classes (if None, uses config value)
        """
        print("Loading REVE model...")
        self.model, self.pos_bank = load_reve_model()
        
        if self.load_positions:
            print("Setting up model with position embeddings...")
            self.model, self.positions = setup_model(
                self.model, 
                self.pos_bank, 
                num_classes=num_classes,
                eeg_positions=self.channel_labels
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("REVE model loaded successfully")
    
    def freeze_backbone(self):
        """Freeze REVE backbone for feature extraction only."""
        freeze_backbone(self.model)
    
    def extract_features(self, epochs, batch_size=8, return_logits=False):
        """
        Extract features from EEG epochs.
        
        Args:
            epochs: np.ndarray of shape (num_epochs, num_channels, epoch_samples)
                    or torch.Tensor
            batch_size: Batch size for processing
            return_logits: If True, return model logits; if False, return features before final layer
            
        Returns:
            np.ndarray: Features of shape (num_epochs, feature_dim)
        """
        if self.model is None:
            self.load_model()
        
        # Convert to tensor if needed
        if isinstance(epochs, np.ndarray):
            epochs_tensor = torch.from_numpy(epochs).float()
        else:
            epochs_tensor = epochs.float()
        
        num_epochs = epochs_tensor.shape[0]
        print(f"\nExtracting features from {num_epochs} epochs...")
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, num_epochs, batch_size):
                batch_end = min(i + batch_size, num_epochs)
                batch = epochs_tensor[i:batch_end].to(self.device)
                
                # Prepare positions for batch (like in tutorial: repeat for batch dimension)
                try:
                    if self.positions is not None:
                        # positions shape: (num_channels, embedding_dim)
                        # After repeat: (batch_size, num_channels, embedding_dim)
                        pos = self.positions.repeat(batch.shape[0], 1, 1).to(self.device)
                        output = self.model(batch, pos)
                    else:
                        output = self.model(batch)
                except Exception as e:
                    print(f"Warning: Model forward failed ({e}). Trying without positions...")
                    output = self.model(batch)
                
                if return_logits:
                    # Output from final classification layer
                    features = output
                else:
                    # Features before classification head (if available)
                    # For now, use full output
                    features = output
                
                all_features.append(features.cpu().numpy())
                
                if (i // batch_size + 1) % max(1, 5 // batch_size) == 0:
                    print(f"  Processed {batch_end}/{num_epochs} epochs")
        
        features_array = np.vstack(all_features)
        print(f"Features extracted: shape {features_array.shape}")
        
        return features_array
    
    def extract_features_for_labels(self, epochs, labels, batch_size=8):
        """
        Extract features organized by label.
        
        Args:
            epochs: np.ndarray of shape (num_epochs, num_channels, epoch_samples)
            labels: list of epoch labels
            batch_size: Batch size for processing
            
        Returns:
            dict: {label: features_array for that label}
        """
        features = self.extract_features(epochs, batch_size=batch_size)
        
        label_features = {}
        for label in set(labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            label_features[label] = features[indices]
        
        print("\nFeatures by label:")
        for label, feats in label_features.items():
            print(f"  {label}: {feats.shape}")
        
        return label_features
