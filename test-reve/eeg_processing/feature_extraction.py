"""
REVE-based feature extraction from EEG epochs.
Uses pre-mapped electrode positions from electrode_mapping_to_standard.json
to leverage REVE's pre-trained position embeddings consistently across subjects.
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
    """Extract features from EEG epochs using REVE model with mapped position embeddings."""
    
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
        self.electrode_mapping = None
        self.reve_positions_dict = None
        
    def _load_electrode_mapping(self):
        """Load the pre-computed electrode-to-REVE position mapping."""
        mapping_path = Path(__file__).parent / "electrodes_pos" / "electrode_mapping_to_standard.json"
        
        if not mapping_path.exists():
            print(f"WARNING: Mapping file not found at {mapping_path}")
            print("Run python inspect_electrodes/map_to_standard_positions.py first!")
            return False
        
        with open(mapping_path, 'r') as f:
            self.electrode_mapping = json.load(f)
        
        print(f"[OK] Loaded electrode mapping: {len(self.electrode_mapping)} electrodes")
        return True
    
    def _load_reve_positions(self):
        """Load all REVE position embeddings from pre-computed file."""
        positions_path = Path(__file__).parent / "electrodes_pos" / "reve_all_positions.json"
        
        if not positions_path.exists():
            print(f"WARNING: REVE positions file not found at {positions_path}")
            return False
        
        with open(positions_path, 'r') as f:
            self.reve_positions_dict = json.load(f)
        
        print(f"[OK] Loaded REVE positions: {len(self.reve_positions_dict)} total positions available")
        return True
    
    def _get_reve_position_names(self):
        """
        Get the REVE position names for all 96 electrodes in order.
        Uses the pre-computed electrode-to-REVE position mapping.
        
        Returns:
            list: REVE position names (one for each electrode 1-96)
        """
        if self.electrode_mapping is None:
            if not self._load_electrode_mapping():
                return None
        
        # Build position name list for our 96 channels in order
        # Order: electrode 1, 2, 3, ..., 96
        position_names = []
        
        for electrode_num in range(1, 97):  # 1-indexed like the mapping
            electrode_key = str(electrode_num)
            
            if electrode_key in self.electrode_mapping:
                mapped_position_name = self.electrode_mapping[electrode_key]['standard_position']
                position_names.append(mapped_position_name)
            else:
                print(f"WARNING: Electrode {electrode_num} not in mapping, using default")
                position_names.append(f"CH{electrode_num}")
        
        print(f"[OK] Created position name list: {len(position_names)} positions")
        print(f"  Sample: {position_names[:5]} ... {position_names[-3:]}")
        
        return position_names
        
    def load_model(self, num_classes=None):
        """
        Load REVE model with mapped position embeddings.
        
        Args:
            num_classes: Number of output classes (if None, uses config value)
        """
        print("Loading REVE model...")
        self.model, self.pos_bank = load_reve_model()
        
        print("\nSetting up model with mapped position embeddings...")
        print("Using pre-computed electrode-to-REVE position mapping...")
        
        # Get REVE position names from our mapping
        position_names = self._get_reve_position_names()
        
        if position_names is not None:
            self.model, self.positions = setup_model(
                self.model, 
                self.pos_bank, 
                num_classes=num_classes,
                eeg_positions=position_names  # Pass position names, not tensors
            )
            print("[OK] Model setup complete with mapped positions")
        else:
            print("WARNING: Could not load position mapping, continuing without positions")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("REVE model loaded successfully")
    
    def freeze_backbone(self):
        """Freeze REVE backbone for feature extraction only."""
        freeze_backbone(self.model)
    
    def extract_features(self, epochs, batch_size=8, return_logits=False):
        """
        Extract features from EEG epochs using REVE with mapped position embeddings.
        
        Args:
            epochs: np.ndarray of shape (num_epochs, num_channels, epoch_samples)
                    or torch.Tensor
            batch_size: Batch size for processing
            return_logits: Not used (REVE returns full model output)
            
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
        print(f"\nExtracting features from {num_epochs} epochs with mapped positions...")
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, num_epochs, batch_size):
                batch_end = min(i + batch_size, num_epochs)
                batch = epochs_tensor[i:batch_end].to(self.device)
                
                # Prepare positions for batch
                if self.positions is not None:
                    # positions shape: (num_channels, embedding_dim)
                    # After repeat: (batch_size, num_channels, embedding_dim)
                    pos = self.positions.repeat(batch.shape[0], 1, 1).to(self.device)
                    
                    # REVE model always requires positions argument
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
