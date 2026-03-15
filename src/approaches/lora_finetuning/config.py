"""
config.py — Configuration and hyperparameters for LoRA fine-tuning.

All defaults, paths, and trainer knobs live here. The LoRAConfig dataclass
is populated from CLI arguments in train_lora.py and passed to run_fold()
and summary functions — no module-level globals needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT       = PROJECT_ROOT / "data" / "thu ep" / "preprocessed"
REVE_MODEL_PATH = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-base"
REVE_POS_PATH   = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-positions"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "lora_checkpoints"

# ── W&B ───────────────────────────────────────────────────────────────────────

USE_WANDB     = True
WANDB_PROJECT = "eeg-lora-thu-ep"
WANDB_ENTITY  = "zl-tudelft-thesis"

# ── Hardware ──────────────────────────────────────────────────────────────────

SAMPLING_RATE = 200
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
ACCELERATOR   = "gpu"  if torch.cuda.is_available() else "cpu"
NUM_WORKERS   = 0 if sys.platform == "win32" else 4

# ── Trainer knobs ─────────────────────────────────────────────────────────────

ACCUMULATE_GRAD_BATCHES = 1
GRADIENT_CLIP_VAL       = 1.0
PRECISION               = "16-mixed"


@dataclass
class LoRAConfig:
    """Mutable run configuration populated from CLI arguments."""

    # Task
    task_mode: str = "binary"
    fold: int | None = None
    generalization: bool = False
    gen_seeds: list[int] = field(default_factory=lambda: [123])

    # Window
    window_size: int = 2000    # 10 s at 200 Hz
    stride: int = 2000         # 10 s at 200 Hz

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Head
    head_dropout: float = 0.1

    # Optimiser
    lr_head: float = 1e-3
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64

    # Training
    max_epochs: int = 80
    phase1_epochs: int = 10
    unfreeze_cls: bool = False
    mixup_alpha: float = 0.0

    # Pooling
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
    def cls_tag(self) -> str:
        return "_cls" if self.unfreeze_cls else ""

    @property
    def rank_tag(self) -> str:
        return f"r{self.lora_rank}"

    def run_name(self, fold_idx: int, gen_seed: int | None = None) -> str:
        gen = f"_gen_s{gen_seed}" if gen_seed is not None else ""
        return (
            f"lora_{self.task_mode}_{self.window_tag}_{self.pool_tag}"
            f"{self.cls_tag}_{self.rank_tag}{gen}_fold_{fold_idx}"
        )

    def group_name(self) -> str:
        gen = "_gen" if self.generalization else ""
        return (
            f"lora_{self.task_mode}_{self.window_tag}_{self.pool_tag}"
            f"{self.cls_tag}_{self.rank_tag}{gen}"
        )

    def hparams_dict(
        self,
        fold_idx: int,
        n_folds: int,
        n_train_subjects: int,
        n_val_subjects: int,
        n_train_windows: int,
        n_val_windows: int,
        gen_seed: int | None = None,
    ) -> dict:
        return {
            "task_mode":        self.task_mode,
            "fold":             fold_idx,
            "n_folds":          n_folds,
            "batch_size":       self.batch_size,
            "accumulate_grad":  ACCUMULATE_GRAD_BATCHES,
            "effective_batch":  self.batch_size * ACCUMULATE_GRAD_BATCHES,
            "lr_head":          self.lr_head,
            "lr_lora":          self.lr_lora,
            "weight_decay":     self.weight_decay,
            "max_epochs":       self.max_epochs,
            "phase1_epochs":    self.phase1_epochs,
            "lora_rank":        self.lora_rank,
            "lora_alpha":       self.lora_alpha,
            "lora_dropout":     self.lora_dropout,
            "head_dropout":     self.head_dropout,
            "unfreeze_cls":     self.unfreeze_cls,
            "mixup_alpha":      self.mixup_alpha,
            "use_pooling":      self.use_pooling,
            "no_pool_mode":     self.no_pool_mode,
            "window_size":      self.window_size,
            "stride":           self.stride,
            "gradient_clip":    GRADIENT_CLIP_VAL,
            "precision":        PRECISION,
            "generalization":   self.generalization,
            "gen_seed":         gen_seed,
            "n_train_subjects": n_train_subjects,
            "n_val_subjects":   n_val_subjects,
            "n_train_windows":  n_train_windows,
            "n_val_windows":    n_val_windows,
        }
