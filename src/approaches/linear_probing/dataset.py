"""
dataset.py — LP-specific dataset classes.

Provides:
  - EmbeddedDataset: thin wrapper over pre-computed 512-D REVE embedding tensors.

Shared dataset utilities (THUEPWindowDataset, label maps, excluded subjects/stimuli)
live in src.thu_ep.dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


# EmbeddedDataset
class EmbeddedDataset(Dataset):
    """
    Thin wrapper around a pre-computed embedding .pt file.

    The file must be a dict with keys:
        'embeddings': Tensor of shape (N, 512)
        'labels':     Tensor of shape (N,) with integer class labels

    Args:
        embeddings_path: Path to the .pt file produced by EmbeddingExtractor.
    """

    def __init__(self, embeddings_path: Path) -> None:
        super().__init__()
        embeddings_path = Path(embeddings_path)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        payload = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.embeddings: Tensor = payload["embeddings"]  # (N, 512)
        self.labels: Tensor = payload["labels"]          # (N,)

        assert self.embeddings.shape[0] == self.labels.shape[0], (
            "Mismatch between number of embeddings and labels."
        )

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.embeddings[idx], self.labels[idx]
