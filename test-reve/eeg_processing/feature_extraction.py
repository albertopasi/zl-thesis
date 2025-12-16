"""
REVE-based feature extraction from EEG epochs.
Uses actual measured 3D electrode coordinates (no position embedding lookup).
REVE can handle arbitrary electrode configurations using the actual 3D coordinates.
"""

import torch
import numpy as np
import json
from pathlib import Path

# Import REVE model functions from local module
from reve_model import (
    load_reve_model,
    setup_model,
    get_device,
    freeze_backbone
)


class REVEFeatureExtractor:
    """Extract features from EEG epochs using REVE model with actual 3D electrode coordinates."""
    
    def __init__(self, device=None, channel_labels=None):
        """
        Initialize REVE feature extractor.
        
        Args:
            device: torch.device (default: auto-detect GPU/CPU)
            channel_labels: List of channel labels (if None, generates default labels)
        """
        self.device = device or get_device()
        self.model = None
        self.pos_bank = None
        self.positions = None
        self.channel_labels = channel_labels or [str(i+1) for i in range(96)]
        self.electrode_coordinates = None  # Actual 3D coordinates from XDF
        
    def _load_electrode_coordinates(self):
        """Load actual 3D electrode coordinates from the XDF extraction."""
        coords_path = Path(__file__).parent / "electrodes_pos" / "electrode_positions.json"
        
        if not coords_path.exists():
            print(f"ERROR: Electrode coordinates not found at {coords_path}")
            print("Run: python inspect_electrodes/generate_electrode_positions.py first!")
            return False
        
        with open(coords_path, 'r') as f:
            positions_dict = json.load(f)
        
        # Convert to numpy array: shape (96, 3) with x, y, z for each electrode
        self.electrode_coordinates = np.zeros((96, 3))
        
        for electrode_num in range(1, 97):
            key = str(electrode_num)
            if key in positions_dict:
                coord = positions_dict[key]
                self.electrode_coordinates[electrode_num - 1] = [coord['x'], coord['y'], coord['z']]
            else:
                print(f"WARNING: Electrode {electrode_num} not found in coordinates")
        
        print(f"[OK] Loaded 3D electrode coordinates: shape {self.electrode_coordinates.shape}")
        print(f"  Coordinate ranges:")
        print(f"    X: [{self.electrode_coordinates[:, 0].min():.2f}, {self.electrode_coordinates[:, 0].max():.2f}] mm")
        print(f"    Y: [{self.electrode_coordinates[:, 1].min():.2f}, {self.electrode_coordinates[:, 1].max():.2f}] mm")
        print(f"    Z: [{self.electrode_coordinates[:, 2].min():.2f}, {self.electrode_coordinates[:, 2].max():.2f}] mm")
        
        return True
        
    def load_model(self, num_classes=None):
        """
        Load REVE model and prepare to use actual 3D electrode coordinates.
        
        Args:
            num_classes: Number of output classes (if None, uses config value)
        """
        print("Loading REVE model...")
        self.model, self.pos_bank = load_reve_model()
        
        print("\nSetting up model with actual 3D electrode coordinates...")
        print("REVE will handle arbitrary electrode configurations using the measured 3D coordinates...")
        
        # Load the actual 3D coordinates from XDF
        if not self._load_electrode_coordinates():
            print("ERROR: Could not load electrode coordinates!")
            return False
        
        self.model, self.positions = setup_model(
            self.model, 
            self.pos_bank, 
            num_classes=num_classes,
            electrode_coordinates=self.electrode_coordinates  # Pass actual 3D coords
        )
        print("[OK] Model setup complete with 3D electrode coordinates")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("REVE model loaded successfully")
        return True
    
    def freeze_backbone(self):
        """Freeze REVE backbone for feature extraction only."""
        freeze_backbone(self.model)
    
    def extract_features(self, epochs, batch_size=8, return_logits=False):
        """
        Extract features from EEG epochs using REVE with actual 3D electrode coordinates.
        
        REVE's approach:
        ================
        The REVE model's forward pass requires both:
        1. EEG signals: (batch_size, num_channels, sequence_length)
        2. Position coordinates: (num_channels, 3) - actual 3D electrode locations
        
        The model's 4D positional encoding combines:
        - Spatial: 3D coordinates (x, y, z) from electrode_positions.json
        - Temporal: timestep information (handled internally)
        
        This allows the model to handle arbitrary electrode montages without
        learned position embeddings.
        
        Args:
            epochs: np.ndarray of shape (num_epochs, num_channels, epoch_samples)
                    or torch.Tensor
            batch_size: Batch size for processing
            return_logits: Not used (REVE returns full model output)
            
        Returns:
            np.ndarray: Features of shape (num_epochs, feature_dim)
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        # Convert to tensor if needed
        if isinstance(epochs, np.ndarray):
            epochs_tensor = torch.from_numpy(epochs).float()
        else:
            epochs_tensor = epochs.float()
        
        num_epochs = epochs_tensor.shape[0]
        print(f"\nExtracting features from {num_epochs} epochs with 3D electrode coordinates...")
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, num_epochs, batch_size):
                batch_end = min(i + batch_size, num_epochs)
                batch = epochs_tensor[i:batch_end].to(self.device)
                
                # Pass positions (3D coordinates) to REVE model
                if self.positions is not None:
                    # self.positions shape: (num_channels, 3) - the actual 3D coordinates
                    # REVE expects: (batch_size, num_channels, 3)
                    # The model will add temporal dimension internally
                    
                    current_batch_size = batch.shape[0]
                    pos = self.positions.unsqueeze(0).expand(current_batch_size, -1, -1).to(self.device)
                    # pos shape after expand: (batch_size, num_channels, 3)
                    
                    # REVE forward pass: model(eeg_signals, positions)
                    # The model uses 4D positional encoding with both spatial and temporal info
                    output = self.model(batch, pos)
                else:
                    print("ERROR: Positions not loaded!")
                    return None
                
                # REVE output is already a feature representation
                # Flatten if needed to get feature vector per epoch
                if output.dim() > 2:
                    features = output.reshape(output.shape[0], -1)
                else:
                    features = output
                
                all_features.append(features.cpu().numpy())
                
                if (i // batch_size + 1) % max(1, 5 // batch_size) == 0:
                    print(f"  Processed {batch_end}/{num_epochs} epochs")
        
        features_array = np.vstack(all_features)
        print(f"Features extracted: shape {features_array.shape}")
        
        return features_array
    
    def extract_features_for_labels(self, features, labels):
        """
        Organize already-extracted features by label.
        
        Args:
            features: np.ndarray of shape (num_epochs, feature_dim) - already extracted
            labels: list of epoch labels
            
        Returns:
            dict: {label: features_array for that label}
        """
        # Validate that labels and features match
        if len(labels) != features.shape[0]:
            print(f"\nWARNING: Label/feature mismatch!")
            print(f"  Labels: {len(labels)}")
            print(f"  Features: {features.shape[0]}")
            print(f"  Using only first {features.shape[0]} labels")
            labels = labels[:features.shape[0]]
        
        label_features = {}
        for label in set(labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            label_features[label] = features[indices]
        
        print("\nFeatures by label:")
        for label, feats in label_features.items():
            print(f"  {label}: {feats.shape}")
        
        return label_features
