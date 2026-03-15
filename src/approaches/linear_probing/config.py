"""
config.py — Configuration and hyperparameters for Linear Probing.

All defaults, paths, and hardware constants live here. The LPConfig dataclass
is populated from CLI arguments in train_lp.py and passed to run_fold()
and summary functions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT       = PROJECT_ROOT / "data" / "thu ep" / "preprocessed"
EMBEDDINGS_DIR  = PROJECT_ROOT / "data" / "thu ep" / "embeddings"
REVE_MODEL_PATH = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-base"
REVE_POS_PATH   = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-positions"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "lp_checkpoints"

# ── W&B ───────────────────────────────────────────────────────────────────────

USE_WANDB     = True
WANDB_PROJECT = "eeg-lp-thu-ep"
WANDB_ENTITY  = "zl-tudelft-thesis"

# ── Hardware ──────────────────────────────────────────────────────────────────

SAMPLING_RATE = 200
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ACCELERATOR   = "gpu"  if torch.cuda.is_available() else "cpu"
NUM_WORKERS   = 0 if sys.platform == "win32" else 4


@dataclass
class LPConfig:
    """Mutable run configuration populated from CLI arguments."""

    # Task
    task_mode: str = "binary"
    fold: int | None = None
    generalization: bool = False
    gen_seeds: list[int] = field(default_factory=lambda: [123])

    # Window
    window_size: int = 1600    # 8 s at 200 Hz
    stride: int = 800          # 4 s at 200 Hz

    # Training
    max_epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3

    # Features
    normalize_features: bool = False
    use_pooling: bool = True
    no_pool_mode: str = "mean"

    # ── Derived helpers ───────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        return 2 if self.task_mode == "binary" else 9

    @property
    def window_tag(self) -> str:
        w_s  = round(self.window_size / SAMPLING_RATE)
        st_s = round(self.stride / SAMPLING_RATE)
        return f"w{w_s}s{st_s}"

    @property
    def pool_tag(self) -> str:
        return "pool" if self.use_pooling else f"nopool_{self.no_pool_mode}"

    @property
    def norm_tag(self) -> str:
        return "_norm" if self.normalize_features else ""

    def run_name(self, fold_idx: int, gen_seed: int | None = None) -> str:
        gen = f"_gen_s{gen_seed}" if gen_seed is not None else ""
        return f"lp_{self.task_mode}_{self.window_tag}_{self.pool_tag}{self.norm_tag}{gen}_fold_{fold_idx}"

    def group_name(self) -> str:
        gen = "_gen" if self.generalization else ""
        return f"lp_{self.task_mode}_{self.window_tag}_{self.pool_tag}{self.norm_tag}{gen}"

    def hparams_dict(
        self,
        fold_idx: int,
        n_folds: int,
        n_train_subjects: int,
        n_val_subjects: int,
        n_train_windows: int,
        n_val_windows: int,
        embed_dim: int,
        gen_seed: int | None = None,
    ) -> dict:
        return {
            "task_mode":          self.task_mode,
            "fold":               fold_idx,
            "n_folds":            n_folds,
            "batch_size":         self.batch_size,
            "lr":                 self.lr,
            "max_epochs":         self.max_epochs,
            "window_size":        self.window_size,
            "stride":             self.stride,
            "n_train_subjects":   n_train_subjects,
            "n_val_subjects":     n_val_subjects,
            "n_train_windows":    n_train_windows,
            "n_val_windows":      n_val_windows,
            "normalize_features": self.normalize_features,
            "use_pooling":        self.use_pooling,
            "no_pool_mode":       self.no_pool_mode,
            "embed_dim":          embed_dim,
            "generalization":     self.generalization,
            "gen_seed":           gen_seed,
        }
