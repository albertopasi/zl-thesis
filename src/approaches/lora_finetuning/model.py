"""
model.py — REVE LoRA Fine-Tuning Lightning Module.

Provides:
  - REVELoRAModule: wraps the frozen REVE encoder with LoRA adapters on the
    attention QKVO layers, a configurable pooling step, and a classification head.
    Implements the two-phase training strategy from the REVE paper:
      Phase 1 — head-only (encoder frozen, LoRA frozen)
      Phase 2 — LoRA + head (LoRA adapters unfrozen, original weights still frozen)

  - WarmupSchedulerCallback: Lightning callback implementing the REVE authors'
    per-phase LR schedule: 10% linear warmup → 80% peak → 10% linear cooldown
    to 1% of peak, applied within the first epoch of each training phase.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import lightning as L
import torchmetrics
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

from src.thu_ep.config import THUEPConfig


class REVELoRAModule(L.LightningModule):
    """
    Two-phase LoRA fine-tuning of the REVE encoder for emotion classification.

    Phase 1 (epochs 0 .. phase1_epochs-1):
        Only the classification head is trainable. LoRA adapters exist but are
        frozen, so the encoder behaves identically to the pretrained model.

    Phase 2 (epochs phase1_epochs .. end):
        LoRA adapters on to_qkv and to_out (attention QKVO) are unfrozen.
        Optionally, the cls_query_token is also unfrozen.
        Original REVE weights remain frozen throughout.

    Pooling strategies:
        pool         — attention pooling via cls_query_token → (B, 512)
        nopool_mean  — mean over channels → (B, H*512)
        nopool_flat  — full flatten → (B, C*H*512)
    """

    def __init__(
        self,
        reve_model_path: str | Path,
        reve_pos_path: str | Path,
        config: THUEPConfig,
        num_classes: int,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        head_dropout: float = 0.1,
        lr_head: float = 1e-3,
        lr_lora: float = 1e-4,
        weight_decay: float = 0.01,
        phase1_epochs: int = 10,
        unfreeze_cls: bool = False,
        mixup_alpha: float = 0.0,
        use_pooling: bool = True,
        no_pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config", "reve_model_path", "reve_pos_path"])

        self.phase = 1
        self._head_initialized = False

        # Load REVE encoder
        reve = AutoModel.from_pretrained(
            str(reve_model_path),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        for param in reve.parameters():
            param.requires_grad_(False)

        # Apply LoRA to attention QKVO
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["to_qkv", "to_out"],
            bias="none",
        )
        self.reve = get_peft_model(reve, lora_cfg)

        # Direct reference to the unwrapped REVE model (avoids fragile
        # self.reve.base_model.model chains through the PEFT wrapper).
        self._reve_unwrapped = self.reve.base_model.model

        # Freeze LoRA adapters for Phase 1 (head-only)
        for name, param in self.reve.named_parameters():
            if "lora_" in name:
                param.requires_grad_(False)

        # Position bank
        pos_bank = AutoModel.from_pretrained(
            str(reve_pos_path),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        pos_bank.eval()
        channel_names: list[str] = config.final_channels
        with torch.no_grad():
            pos_1d = pos_bank(channel_names)  # (30, 3)
        self.register_buffer("_pos_1d", pos_1d)
        del pos_bank

        self.embed_dim = self._reve_unwrapped.embed_dim  # 512

        # Classification head: initialized lazily for nopool modes
        # (embed_dim depends on window size which is only known at first forward)
        self.head: nn.Module | None = None
        if use_pooling:
            self._init_head(self.embed_dim)

        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        metric_kwargs = dict(task=task, num_classes=num_classes, average="macro")

        self.train_acc   = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc     = torchmetrics.Accuracy(**metric_kwargs)
        self.train_auroc = torchmetrics.AUROC(**metric_kwargs)
        self.val_auroc   = torchmetrics.AUROC(**metric_kwargs)
        self.train_f1    = torchmetrics.F1Score(**metric_kwargs)
        self.val_f1      = torchmetrics.F1Score(**metric_kwargs)

    def _init_head(self, input_dim: int) -> None:
        self.head = nn.Sequential(
            nn.Dropout(self.hparams.head_dropout),
            nn.Linear(input_dim, self.hparams.num_classes),
        )
        self._head_initialized = True

    def _pool(self, x: Tensor) -> Tensor:
        """
        Aggregate the 4-D encoder output into a 1-D embedding.

        Args:
            x: (B, C, H, E) — channels x temporal patches x embed_dim

        Returns:
            (B, D) where D depends on pooling strategy:
              pool:         D = E (512)
              nopool_mean:  D = H * E
              nopool_flat:  D = C * H * E
        """
        if self.hparams.use_pooling:
            return self._reve_unwrapped.attention_pooling(x)  # (B, E)

        B, C, H, E = x.shape
        if self.hparams.no_pool_mode == "mean":
            return x.mean(dim=1).reshape(B, H * E)       # (B, H*E)
        else:  # flat
            return x.reshape(B, C * H * E)               # (B, C*H*E)

    def _ensure_head(self, emb: Tensor) -> None:
        """Lazily initialize head on first forward pass for nopool modes."""
        if not self._head_initialized:
            self._init_head(emb.shape[-1])
            self.head = self.head.to(emb.device)

    def forward(self, eeg: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            eeg: (B, 30, T) raw EEG windows.
        Returns:
            logits: (B, num_classes)
            emb:    (B, D) pooled embedding
        """
        B = eeg.shape[0]
        pos = self._pos_1d.unsqueeze(0).expand(B, -1, -1)  # (B, 30, 3)
        out_4d = self.reve(eeg, pos)                         # (B, 30, H, 512)
        emb = self._pool(out_4d)                             # (B, D)
        self._ensure_head(emb)
        logits = self.head(emb)                              # (B, num_classes)
        return logits, emb

    # Training / Validation

    def _shared_step(self, batch: tuple[Tensor, Tensor], prefix: str) -> Tensor:
        eeg, labels = batch
        labels = labels.long()
        B = eeg.shape[0]

        logits, emb = self(eeg)

        # Mixup: only in Phase 2, only during training, only if enabled
        if (
            prefix == "train"
            and self.phase == 2
            and self.hparams.mixup_alpha > 0
        ):
            lam = torch.distributions.Beta(
                self.hparams.mixup_alpha, self.hparams.mixup_alpha
            ).sample().to(emb.device)
            perm = torch.randperm(B, device=emb.device)
            mixed_emb = lam * emb + (1.0 - lam) * emb[perm]
            logits_mixed = self.head(mixed_emb)
            loss = lam * F.cross_entropy(logits_mixed, labels) + \
                   (1.0 - lam) * F.cross_entropy(logits_mixed, labels[perm])
            logits_for_metrics = logits
        else:
            loss = F.cross_entropy(logits, labels)
            logits_for_metrics = logits

        probs = torch.softmax(logits_for_metrics, dim=-1)
        preds = torch.argmax(logits_for_metrics, dim=-1)

        num_classes = self.hparams.num_classes
        auroc_preds = probs[:, 1] if num_classes == 2 else probs

        acc   = getattr(self, f"{prefix}_acc")
        auroc = getattr(self, f"{prefix}_auroc")
        f1    = getattr(self, f"{prefix}_f1")

        acc(preds, labels)
        auroc(auroc_preds, labels)
        f1(preds, labels)

        self.log(f"{prefix}/loss",  loss,  on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc",   acc,   on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/f1",    f1,    on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, prefix="val")

    # Phase switching

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.phase1_epochs and self.phase == 1:
            self.phase = 2
            n_unfrozen = 0
            for name, param in self.reve.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    n_unfrozen += param.numel()

            if self.hparams.unfreeze_cls:
                cls_token = self._reve_unwrapped.cls_query_token
                cls_token.requires_grad_(True)
                n_unfrozen += cls_token.numel()

            print(
                f"\n>>> Phase 2 started at epoch {self.current_epoch}  "
                f"({n_unfrozen:,} LoRA params unfrozen)"
            )

        self.log("phase", float(self.phase), on_step=False, on_epoch=True)

    # Optimiser

    def configure_optimizers(self):
        wd = self.hparams.weight_decay

        head_params = list(self.head.parameters()) if self.head is not None else []
        lora_params = [
            p for n, p in self.reve.named_parameters() if "lora_" in n
        ]

        param_groups = [
            {"params": head_params, "lr": self.hparams.lr_head, "weight_decay": wd, "name": "head"},
            {"params": lora_params, "lr": self.hparams.lr_lora, "weight_decay": wd, "name": "lora"},
        ]

        if self.hparams.unfreeze_cls:
            cls_token = self._reve_unwrapped.cls_query_token
            param_groups.append(
                {"params": [cls_token], "lr": self.hparams.lr_lora, "weight_decay": 0.0, "name": "cls"}
            )

        optimizer = torch.optim.AdamW(param_groups)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }


class WarmupSchedulerCallback(L.Callback):
    """
    REVE authors' per-phase LR warmup schedule.

    Within the first epoch of each training phase:
      - 10% of batches: linear warmup from 1% to 100% of target LR
      - 80% of batches: hold at peak (target LR)
      - 10% of batches: linear cooldown from 100% to 1% of target LR

    After the warmup epoch, LR is left at the peak value for the optimizer's
    ReduceLROnPlateau to manage.

    Args:
        phase1_epochs: number of Phase 1 epochs (Phase 2 starts at this epoch).
        lr_head:       target LR for the head param group.
        lr_lora:       target LR for the lora/cls param groups.
    """

    def __init__(self, phase1_epochs: int, lr_head: float, lr_lora: float) -> None:
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.lr_head = lr_head
        self.lr_lora = lr_lora
        self._warmup_active = False
        self._total_batches: int | None = None
        self._current_phase_start_epoch: int | None = None

    def _is_warmup_epoch(self, trainer: L.Trainer, pl_module: L.LightningModule) -> bool:
        epoch = trainer.current_epoch
        # Phase 1 warmup: epoch 0
        if epoch == 0:
            return True
        # Phase 2 warmup: first epoch after phase switch
        if epoch == self.phase1_epochs:
            return True
        return False

    def _get_target_lr(self, group_name: str) -> float:
        if group_name == "head":
            return self.lr_head
        return self.lr_lora  # lora, cls

    def _warmup_factor(self, step: int, total: int) -> float:
        """Return LR multiplier (0.01 to 1.0) for the REVE warmup schedule."""
        if total <= 0:
            return 1.0

        warmup_end = int(total * 0.10)
        peak_end   = int(total * 0.90)

        if step < warmup_end:
            # Linear warmup: 1% → 100%
            t = step / max(warmup_end, 1)
            return 0.01 + 0.99 * t
        elif step < peak_end:
            # Peak: 100%
            return 1.0
        else:
            # Linear cooldown: 100% → 1%
            t = (step - peak_end) / max(total - peak_end, 1)
            return 1.0 - 0.99 * t

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._warmup_active = self._is_warmup_epoch(trainer, pl_module)
        if self._warmup_active:
            self._total_batches = trainer.num_training_batches
            phase_label = "Phase 1" if trainer.current_epoch < self.phase1_epochs else "Phase 2"
            print(f"    [{phase_label}] LR warmup active for epoch {trainer.current_epoch} "
                  f"({self._total_batches} batches)")

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        batch, batch_idx: int,
    ) -> None:
        if not self._warmup_active:
            return

        factor = self._warmup_factor(batch_idx, self._total_batches)
        optimizer = trainer.optimizers[0]

        for pg in optimizer.param_groups:
            name = pg.get("name", "")
            target_lr = self._get_target_lr(name)
            pg["lr"] = target_lr * factor

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._warmup_active:
            # Ensure LR is at peak after warmup epoch
            optimizer = trainer.optimizers[0]
            for pg in optimizer.param_groups:
                name = pg.get("name", "")
                pg["lr"] = self._get_target_lr(name)
            self._warmup_active = False
