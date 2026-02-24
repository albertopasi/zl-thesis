"""
Quick script to download all REVE models.

Run this script to download all REVE models to models/reve_pretrained_original:

    python -m src.download_reve.run
    uv run python -m src.download_reve.run

"""

from .download_models import download_all_reve_models


if __name__ == "__main__":
    print("Starting REVE model download...")
    print("Note: You may need to authenticate with Hugging Face first.")
    print("Run: huggingface-cli login")
    print()
    
    download_all_reve_models()
