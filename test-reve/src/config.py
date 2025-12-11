"""
Configuration and hyperparameters for REVE EEG classification training.
"""

# Training parameters
BATCH_SIZE = 64
N_EPOCHS = 20
LEARNING_RATE = 1e-3

# Model parameters
HIDDEN_DIM = 512
NUM_CHANNELS = 20
SAMPLE_LENGTH = 5
NUM_CLASSES = 2

# EEG electrode positions (10-20 system)
EEG_POSITIONS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", 
    "T3", "T4", "C3", "C4", "T5", "T6", 
    "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A2"
]

# Model identifiers
REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITIONS_MODEL_ID = "brain-bzh/reve-positions"

# Dataset identifier
DATASET_ID = "brain-bzh/eegmat-prepro"

# Random seed for reproducibility
RANDOM_SEED = 42
