"""
model.py — REVE Embedding Extractor + Linear Probing Lightning Module.

Provides:
  - EmbeddingExtractor: utility class that runs the frozen REVE encoder over a
    THUEPWindowDataset once and caches the resulting 512-D vectors to disk.
  - LinearProber: Lightning Module that trains a single linear layer on top of
    pre-computed REVE embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
import torchmetrics
from transformers import AutoModel

from src.thu_ep.config import THUEPConfig
from src.thu_ep.dataset import THUEPWindowDataset


# EmbeddingExtractor
class EmbeddingExtractor:
    """
    Pre-computation utility: runs the frozen REVE encoder once over an entire
    THUEPWindowDataset and saves the resulting 512-D embeddings to a .pt file.

    REVE embedding extraction (two-step process)
    --------------------------------------------
    Step 1 — forward pass (return_output=False, the default):
        out_4d = reve_model(eeg, pos)
        # eeg:    (B, C, T)      — C=30 channels, T=1600 timepoints
        # pos:    (B, C, 3)      — 3-D electrode coordinates broadcast per batch
        # out_4d: (B, C, H, 512) — H = number of time patches

    Step 2 — attention pooling to collapse spatial+temporal into a single vector:
        emb = reve_model.attention_pooling(out_4d)
        # emb: (B, 512)          — the globally-pooled representation

    Args:
        reve_model_path:    Local path to the reve-base model directory.
        reve_pos_path:      Local path to the reve-positions model directory.
        config:             THUEPConfig instance (supplies channel names for pos bank).
        device:             Torch device string, e.g. 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        reve_model_path: Path,
        reve_pos_path: Path,
        config: THUEPConfig,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.config = config

        print("Loading REVE model from local path …")
        self.reve = AutoModel.from_pretrained(
            str(reve_model_path),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.reve.eval()
        self.reve.to(self.device)

        # Freeze all REVE parameters to only use it as a feature extractor
        for param in self.reve.parameters():
            param.requires_grad_(False)

        print("Loading REVE position bank from local path …")
        self.pos_bank = AutoModel.from_pretrained(
            str(reve_pos_path),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.pos_bank.eval()
        self.pos_bank.to(self.device)

        # Pre-compute electrode positions once using the 30 final channel names
        # pos_bank(channel_names) -> Tensor of shape (30, 3)
        # expand to (B, 30, 3) at inference time.
        channel_names: list[str] = config.final_channels  # 30 names, A1/A2 removed
        with torch.no_grad():
            self._pos_1d: Tensor = self.pos_bank(channel_names)  # (30, 3)
        print(f"  Electrode positions cached for {len(channel_names)} channels.")

    # Public API
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataset: THUEPWindowDataset,
        batch_size: int = 64,
        use_pooling: bool = True,
        no_pool_mode: str = "mean",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Pass every window in `dataset` through the frozen REVE encoder.

        Args:
            dataset:      THUEPWindowDataset (windows already in RAM).
            batch_size:   How many windows to process per GPU forward pass.
            use_pooling:  If True, apply REVE attention_pooling → (N, 512).
                          If False, return raw patch embeddings flattened
                          according to `no_pool_mode`.
            no_pool_mode: Only used when use_pooling=False.
                          "mean" → mean over channels, flatten time patches → (N, H*512).
                          "flat" → full flatten → (N, C*H*512).

        Returns:
            embeddings:       Float32 Tensor on CPU, shape (N, D).
            labels:           Int64 Tensor of shape (N,) on CPU.
            stimulus_indices: Int64 Tensor of shape (N,) on CPU — 0-indexed stimulus ID per window.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # preserve order for reproducibility
            num_workers=0,  # dataset already in RAM; no I/O benefit from workers
            pin_memory=True,
        )

        all_embeddings: list[Tensor] = []
        all_labels: list[Tensor] = []

        print(f"Extracting REVE embeddings for {len(dataset):,} windows …")
        for eeg_batch, label_batch in tqdm(loader, unit="batch"):
            # eeg_batch: (B, 30, 1600)  float32
            # label_batch: (B,)         int
            B = eeg_batch.shape[0]
            eeg_batch = eeg_batch.to(self.device)

            # Broadcast cached electrode positions to match batch size.
            # _pos_1d shape: (30, 3)  →  pos shape: (B, 30, 3)
            pos = self._pos_1d.unsqueeze(0).expand(B, -1, -1)  # (B, 30, 3)

            # Step 1: REVE forward pass
            #   return_output=False (default) → already rearranged to (B, C, H, E)
            #   where C=30 channels, H=time patches, E=512
            out_4d: Tensor = self.reve(eeg_batch, pos)  # (B, 30, H, 512)

            # Step 2: Pool or flatten depending on use_pooling
            if use_pooling:
                emb = self.reve.attention_pooling(out_4d)   # (B, 512)
            elif no_pool_mode == "mean":
                emb = out_4d.mean(dim=1).reshape(B, -1)     # (B, H*512)
            else:  # "flat"
                emb = out_4d.reshape(B, -1)                  # (B, C*H*512)

            all_embeddings.append(emb.cpu())
            all_labels.append(label_batch.long())

        embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
        labels = torch.cat(all_labels, dim=0)          # (N,)

        # Recover stimulus index for each window from the dataset's flat index.
        # Order is preserved because the DataLoader uses shuffle=False.
        stimulus_indices = torch.tensor(
            [dataset.index[i][1] for i in range(len(dataset))],
            dtype=torch.long,
        )  # (N,)

        print(f"  Done. Embeddings shape: {embeddings.shape}")
        return embeddings, labels, stimulus_indices

    @staticmethod
    def save_embeddings(
        embeddings: Tensor,
        labels: Tensor,
        save_path: Path,
        stimulus_indices: Tensor | None = None,
    ) -> None:
        """
        Persist pre-computed embeddings to disk.

        Args:
            embeddings:       Tensor of shape (N, D).
            labels:           Tensor of shape (N,).
            save_path:        Destination .pt file path (parent dirs created if needed).
            stimulus_indices: Optional Tensor of shape (N,) — 0-indexed stimulus ID per window.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Tensor] = {"embeddings": embeddings, "labels": labels}
        if stimulus_indices is not None:
            payload["stimulus_indices"] = stimulus_indices
        torch.save(payload, save_path)
        print(f"  Saved {embeddings.shape[0]:,} embeddings → {save_path}")


# LinearProber  (Lightning Module)
class LinearProber(L.LightningModule):
    """
    A single linear layer trained on top of pre-computed REVE embeddings.

    Architecture:
        Input: (B, 512) float32 embeddings
        Output: (B, num_classes) logits
        Loss: Cross-entropy

    Metrics tracked (train + val):
        - Accuracy (macro)
        - ROC-AUC  (macro, one-vs-rest)
        - F1-Score (macro)

    Args:
        num_classes: Number of output classes (2 for binary, 9 for 9-class).
        embed_dim:   Dimensionality of input embeddings (default 512).
        lr:          Adam learning rate (default 1e-3).
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        lr: float = 1e-3,
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.normalize_features = normalize_features

        # LINEAR PROBE, the only trainable component
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Metrics (separate instances for train vs val to avoid state mixing)
        task = "binary" if num_classes == 2 else "multiclass"
        metric_kwargs = dict(task=task, num_classes=num_classes, average="macro")

        self.train_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc   = torchmetrics.Accuracy(**metric_kwargs)

        # AUROC needs probability inputs (softmax), not raw logits
        auroc_kwargs = dict(task=task, num_classes=num_classes, average="macro")
        self.train_auroc = torchmetrics.AUROC(**auroc_kwargs)
        self.val_auroc   = torchmetrics.AUROC(**auroc_kwargs)

        self.train_f1 = torchmetrics.F1Score(**metric_kwargs)
        self.val_f1   = torchmetrics.F1Score(**metric_kwargs)

    # Forward
    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 512) → logits: (B, num_classes)"""
        if self.normalize_features:
            x = F.normalize(x, dim=-1)
        return self.classifier(x)

    # Training / Validation steps
    def _shared_step(self, batch: Tuple[Tensor, Tensor], prefix: str) -> Tensor:
        embeddings, labels = batch
        labels = labels.long()

        logits = self(embeddings)                        # (B, num_classes)
        loss = F.cross_entropy(logits, labels)

        probs = torch.softmax(logits, dim=-1)            # (B, num_classes)
        preds = torch.argmax(logits, dim=-1)             # (B,)

        # For binary AUROC, torchmetrics expects shape (B,) — the positive-class
        # probability only. For multiclass it expects the full (B, C) matrix.
        auroc_preds = probs[:, 1] if self.num_classes == 2 else probs

        # Retrieve the correct metric objects based on split prefix
        acc   = getattr(self, f"{prefix}_acc")
        auroc = getattr(self, f"{prefix}_auroc")
        f1    = getattr(self, f"{prefix}_f1")

        acc(preds, labels)
        auroc(auroc_preds, labels)
        f1(preds, labels)

        self.log(f"{prefix}/loss",  loss,        on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc",   acc,          on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/auroc", auroc,        on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/f1",    f1,           on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, prefix="val")

    # Optimiser
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
