"""
REVE Model Loader - Download and load REVE models from HuggingFace

Simple, modular interface for loading REVE models.
Models are cached locally in the models/ folder.
REVE is downloaded in both its BASE and LARGE version. (default used = BASE)
"""

from transformers import AutoModel
from pathlib import Path
from ..config import DEVICE, MODELS_CACHE_DIR, TORCH_DTYPE, TRUST_REMOTE_CODE, REVE_MODEL_SIZE


# HuggingFace model IDs
REVE_MODELS = {
    "base": "brain-bzh/reve-base",
    "large": "brain-bzh/reve-large",
}
REVE_POSITIONS_MODEL_ID = "brain-bzh/reve-positions"


def load_reve_model(size: str = None, cache_dir: Path = None):
    """
    Load REVE model from HuggingFace (with local caching).
    
    Uses device and other settings from src/config.py
    
    Args:
        size: Model size ("base" or "large"). Default from config.
        cache_dir: Where to cache models (default: from config)
    
    Returns:
        model: REVE model
    """
    if size is None:
        size = REVE_MODEL_SIZE
    
    if size not in REVE_MODELS:
        raise ValueError(f"Unknown model size: {size}. Options: {list(REVE_MODELS.keys())}")
    
    model_id = REVE_MODELS[size]
    
    if cache_dir is None:
        cache_dir = MODELS_CACHE_DIR
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Loading REVE {size} model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")
    print("Downloading from HuggingFace (this may take a few minutes)...")
    
    # Load model from HuggingFace (auto-caches locally)
    model = AutoModel.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=TORCH_DTYPE,
    )
    
    print(f"REVE {size} model loaded successfully")
    
    # Move to device from config
    print(f"Moving model to device: {DEVICE}...")
    model = model.to(DEVICE)
    print(f"Model moved to device: {DEVICE}")
    
    return model


def load_reve_positions(cache_dir: Path = None):
    """
    Load REVE positions model from HuggingFace (with local caching).
    
    Uses device and other settings from src/config.py
    
    Args:
        cache_dir: Where to cache models (default: from config)
    
    Returns:
        pos_bank: REVE positions model/bank
    """
    if cache_dir is None:
        cache_dir = MODELS_CACHE_DIR
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Loading REVE positions model: {REVE_POSITIONS_MODEL_ID}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}")
    
    # Load position bank from HuggingFace (auto-caches locally)
    pos_bank = AutoModel.from_pretrained(
        REVE_POSITIONS_MODEL_ID,
        cache_dir=str(cache_dir),
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=TORCH_DTYPE,
    )
    
    print(f"REVE positions model loaded successfully")
    
    # Move to device from config
    pos_bank = pos_bank.to(DEVICE)
    print(f"Positions model moved to device: {DEVICE}")
    
    return pos_bank


def load_reve_models(size: str = None, cache_dir: Path = None):
    """
    Load both REVE model and positions model.
    
    Convenience function to load both models at once.
    Uses device and other settings from src/config.py
    
    Args:
        size: Model size ("base" or "large"). Default from config.
        cache_dir: Where to cache models (default: from config)
    
    Returns:
        (model, pos_bank): Both REVE models
    """
    model = load_reve_model(size, cache_dir)
    pos_bank = load_reve_positions(cache_dir)
    
    return model, pos_bank


if __name__ == "__main__":
    """
    Download (once) and load all models (base and large)
    """
    print("Testing REVE model loader...")
    
    # Download base model
    print("\n[1/3] Loading base model...")
    model_base = load_reve_model(size="base")
    
    # Download large model
    print("\n[2/3] Loading large model...")
    model_large = load_reve_model(size="large")
    
    # Load positions
    print("\n[3/3] Loading positions model...")
    pos_bank = load_reve_positions()
    
    print("\n✓ All models loaded and cached successfully!")

