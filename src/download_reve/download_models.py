"""
Download REVE models from Hugging Face and save them locally.

This script downloads the REVE models (base, large, and position bank)
from Hugging Face and saves them to a local directory in a portable format
with all weights and configuration files.

Usage:
    python -m src.download_reve.download_models
    uv run python -m src.download_reve.download_models
    
Or import and use:
    from src.download_reve import download_all_reve_models
    download_all_reve_models()
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel
from huggingface_hub import snapshot_download


# Default paths
DEFAULT_SAVE_DIR = Path(__file__).parent.parent.parent / "models" / "reve_pretrained_original"

# Model repository IDs on Hugging Face
REVE_MODELS = {
    "reve-base": "brain-bzh/reve-base",
    "reve-large": "brain-bzh/reve-large",
    "reve-positions": "brain-bzh/reve-positions",
}


def download_reve_model(
    model_name: str,
    save_dir: Optional[Path] = None,
    torch_dtype: str = "auto",
    force_download: bool = False,
) -> Path:
    """
    Download a single REVE model from Hugging Face and save it locally.
    
    Args:
        model_name: Name of the model ("reve-base", "reve-large", or "reve-positions")
        save_dir: Directory to save the model. Defaults to models/reve_pretrained_original
        torch_dtype: PyTorch dtype for model weights. Default is "auto"
        force_download: If True, re-download even if model exists locally
        
    Returns:
        Path to the saved model directory
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in REVE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(REVE_MODELS.keys())}"
        )
    
    save_dir = save_dir or DEFAULT_SAVE_DIR
    save_dir = Path(save_dir)
    model_save_path = save_dir / model_name
    
    # Check if model already exists
    if model_save_path.exists() and not force_download:
        print(f"Model {model_name} already exists at {model_save_path}")
        print("Use force_download=True to re-download")
        return model_save_path
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    repo_id = REVE_MODELS[model_name]
    print(f"Downloading {model_name} from {repo_id}...")
    
    # Method 1: Download full repository snapshot (includes all files)
    # get all custom code, configs, and weights
    temp_snapshot_dir = save_dir / f".temp_{model_name}"
    
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_dir=temp_snapshot_dir,
        )
        
        # Move to final location
        if model_save_path.exists():
            shutil.rmtree(model_save_path)
        shutil.move(str(temp_snapshot_dir), str(model_save_path))
        
        print(f"Successfully saved {model_name} to {model_save_path}")
            
    except Exception as e:
        # Clean up temp directory on failure
        if temp_snapshot_dir.exists():
            shutil.rmtree(temp_snapshot_dir)
        raise RuntimeError(f"Failed to download {model_name}: {e}") from e
    
    return model_save_path


def download_all_reve_models(
    save_dir: Optional[Path] = None,
    torch_dtype: str = "auto",
    force_download: bool = False,
) -> dict[str, Path]:
    """
    Download all REVE models (base, large, and position bank).
    
    Args:
        save_dir: Directory to save models. Defaults to models/reve_pretrained_original
        torch_dtype: PyTorch dtype for model weights. Default is "auto"
        force_download: If True, re-download even if models exist locally
        
    Returns:
        Dictionary mapping model names to their save paths
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR
    save_dir = Path(save_dir)
    
    print("=" * 60)
    print("REVE Model Downloader")
    print("=" * 60)
    print(f"Save directory: {save_dir.absolute()}")
    print(f"Models to download: {list(REVE_MODELS.keys())}")
    print("=" * 60)
    
    results = {}
    
    for model_name in REVE_MODELS:
        print(f"\n[{model_name}]")
        try:
            path = download_reve_model(
                model_name=model_name,
                save_dir=save_dir,
                torch_dtype=torch_dtype,
                force_download=force_download,
            )
            results[model_name] = path
        except Exception as e:
            print(f"ERROR downloading {model_name}: {e}")
            results[model_name] = None
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)
    
    return results


def load_local_reve_model(
    model_name: str,
    save_dir: Optional[Path] = None,
    torch_dtype: str = "auto",
    device: Optional[str] = None,
):
    """
    Load a locally saved REVE model.
    
    Args:
        model_name: Name of the model ("reve-base", "reve-large", or "reve-positions")
        save_dir: Directory where models are saved. Defaults to models/reve_pretrained_original
        torch_dtype: PyTorch dtype for model weights
        device: Device to load model on (e.g., "cuda", "cpu")
        
    Returns:
        The loaded model
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR
    model_path = Path(save_dir) / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run download_reve_model('{model_name}') first."
        )
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    
    if device:
        model = model.to(device)
    
    return model


def load_all_local_models(
    save_dir: Optional[Path] = None,
    torch_dtype: str = "auto",
    device: Optional[str] = None,
) -> dict:
    """
    Load all locally saved REVE models.
    
    Args:
        save_dir: Directory where models are saved
        torch_dtype: PyTorch dtype for model weights
        device: Device to load models on
        
    Returns:
        Dictionary with 'base', 'large', and 'positions' models
    """
    return {
        "base": load_local_reve_model("reve-base", save_dir, torch_dtype, device),
        "large": load_local_reve_model("reve-large", save_dir, torch_dtype, device),
        "positions": load_local_reve_model("reve-positions", save_dir, torch_dtype, device),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download REVE models from Hugging Face"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory to save models",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(REVE_MODELS.keys()) + ["all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="PyTorch dtype for weights (default: auto)",
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        download_all_reve_models(
            save_dir=args.save_dir,
            torch_dtype=args.dtype,
            force_download=args.force,
        )
    else:
        download_reve_model(
            model_name=args.model,
            save_dir=args.save_dir,
            torch_dtype=args.dtype,
            force_download=args.force,
        )
