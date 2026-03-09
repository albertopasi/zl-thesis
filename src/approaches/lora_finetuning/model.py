"""
model.py — REVE LoRA Fine-Tuning Lightning Module.

Provides:
  - REVELoRAModule: wraps the frozen REVE encoder with LoRA adapters on the
    attention QKVO layers, an attention-pooling step, and a classification head.
    Implements the two-phase training strategy from the REVE paper:
      Phase 1 — head-only (encoder frozen, LoRA frozen)
      Phase 2 — LoRA + head (LoRA adapters unfrozen, original weights still frozen)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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

    Args:
        reve_model_path:  Path to the reve-base model directory.
        reve_pos_path:    Path to the reve-positions model directory.
        config:           THUEPConfig (supplies channel names for position bank).
        num_classes:      2 (binary) or 9 (9-class).
        lora_rank:        LoRA rank r.
        lora_alpha:       LoRA scaling factor alpha.
        lora_dropout:     Dropout on LoRA adapter outputs.
        head_dropout:     Dropout before the classification head.
        lr_head:          Learning rate for the classification head.
        lr_lora:          Learning rate for LoRA adapters + cls_query_token.
        phase1_epochs:    Number of head-only epochs before unfreezing LoRA.
        warmup_epochs:    Linear LR warmup epochs at the start of Phase 2 (0 = disabled).
        unfreeze_cls:     Whether to unfreeze cls_query_token in Phase 2.
        mixup_alpha:      Beta distribution param for Mixup (0.0 = disabled).
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
        phase1_epochs: int = 10,
        warmup_epochs: int = 3,
        unfreeze_cls: bool = False,
        mixup_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config", "reve_model_path", "reve_pos_path"])

        self.phase = 1

        # Load REVE encoder
        reve = AutoModel.from_pretrained(
            str(reve_model_path),
            trust_remote_code=True,
            torch_dtype="auto",
        )
        # Freeze all original weights
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

        # Classification head
        embed_dim = self.reve.base_model.model.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        metric_kwargs = dict(task=task, num_classes=num_classes, average="macro")
        auroc_kwargs  = dict(task=task, num_classes=num_classes, average="macro")

        self.train_acc   = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc     = torchmetrics.Accuracy(**metric_kwargs)
        self.train_auroc = torchmetrics.AUROC(**auroc_kwargs)
        self.val_auroc   = torchmetrics.AUROC(**auroc_kwargs)
        self.train_f1    = torchmetrics.F1Score(**metric_kwargs)
        self.val_f1      = torchmetrics.F1Score(**metric_kwargs)

    # Forward

    def forward(self, eeg: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            eeg: (B, 30, T) raw EEG windows.
            pos: (B, 30, 3) electrode positions.
        Returns:
            logits: (B, num_classes)
            emb:    (B, 512) pooled embedding (for optional mixup)
        """
        out_4d = self.reve(eeg, pos)  # (B, 30, H, 512)
        emb = self.reve.base_model.model.attention_pooling(out_4d)  # (B, 512)
        logits = self.head(emb)  # (B, num_classes)
        return logits, emb

    # Training / Validation

    def _shared_step(self, batch: Tuple[Tensor, Tensor], prefix: str) -> Tensor:
        eeg, labels = batch
        labels = labels.long()
        B = eeg.shape[0]

        pos = self._pos_1d.unsqueeze(0).expand(B, -1, -1)  # (B, 30, 3)
        logits, emb = self(eeg, pos)

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
            # Use un-mixed logits for metrics (cleaner evaluation)
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

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
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

            # Optionally unfreeze cls_query_token
            if self.hparams.unfreeze_cls:
                cls_token = self.reve.base_model.model.cls_query_token
                cls_token.requires_grad_(True)
                n_unfrozen += cls_token.numel()

            print(
                f"\n>>> Phase 2 started at epoch {self.current_epoch}  "
                f"({n_unfrozen:,} LoRA params unfrozen)"
            )
            if self.hparams.warmup_epochs > 0:
                print(
                    f"    LR warmup: {self.hparams.warmup_epochs} epochs  "
                    f"(lr_lora: {self.hparams.lr_lora * (1 / self.hparams.warmup_epochs):.2e} → "
                    f"{self.hparams.lr_lora:.2e})\n"
                )

        # Linear warmup for LoRA/cls params at the start of Phase 2
        if self.phase == 2 and self.hparams.warmup_epochs > 0:
            phase2_epoch = self.current_epoch - self.hparams.phase1_epochs
            if phase2_epoch < self.hparams.warmup_epochs:
                warmup_factor = (phase2_epoch + 1) / self.hparams.warmup_epochs
                optimizer = self.optimizers().optimizer
                for pg in optimizer.param_groups:
                    if pg.get("name") in ("lora", "cls"):
                        pg["lr"] = self.hparams.lr_lora * warmup_factor

        self.log("phase", float(self.phase), on_step=False, on_epoch=True)

    # Optimiser + LR schedule

    def configure_optimizers(self):
        head_params = list(self.head.parameters())
        lora_params = [
            p for n, p in self.reve.named_parameters() if "lora_" in n
        ]

        param_groups = [
            {"params": head_params, "lr": self.hparams.lr_head, "weight_decay": 0.01, "name": "head"},
            {"params": lora_params, "lr": self.hparams.lr_lora, "weight_decay": 0.01, "name": "lora"},
        ]

        # Add cls_query_token to optimizer if it may be unfrozen
        if self.hparams.unfreeze_cls:
            cls_token = self.reve.base_model.model.cls_query_token
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
