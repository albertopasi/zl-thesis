"""
train_lora.py — Main execution script for the LoRA Fine-Tuning run.

Workflow
--------
1. Build subject list (79 subjects).
2. Apply 10-fold cross-subject KFold split.
3. For each fold:
   a. Create THUEPWindowDataset for train/val subjects (raw EEG windows).
   b. Instantiate REVELoRAModule (loads fresh REVE from disk each fold).
   c. Phase 1: train head only (LoRA frozen).
   d. Phase 2: unfreeze LoRA adapters, continue training.
   e. Save best checkpoint + adapter weights + results JSON.
4. Print cross-fold summary (mean ± std) and save aggregate JSON.

Run with:
    # Binary, all folds (default)
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary

    # 9-class, single fold (dry run)
    uv run python -m src.approaches.lora_finetuning.train_lora --task 9-class --fold 1

    # Custom rank and window
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary --rank 16 --window 10 --stride 5

"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import wandb

from src.thu_ep.config import THUEPConfig
from src.thu_ep.dataset import EXCLUDED_SUBJECTS, THUEPWindowDataset
from src.thu_ep.folds import get_all_subjects, get_kfold_splits, N_FOLDS
from src.thu_ep.callbacks import EpochSummaryCallback, fmt_dur, fmt_metric, COL_W
from src.approaches.lora_finetuning.model import REVELoRAModule


# Configuration

TASK_MODE      = "binary"
MAX_EPOCHS     = 80
PHASE1_EPOCHS  = 10
BATCH_SIZE     = 64  # MODIFY BATCH SIZE
LR_HEAD        = 1e-3
LR_LORA        = 1e-4
LORA_RANK      = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.1
HEAD_DROPOUT   = 0.1
UNFREEZE_CLS   = False
WARMUP_EPOCHS  = 3
MIXUP_ALPHA    = 0.0

DRY_RUN_FOLD   = None

# Paths
DATA_ROOT       = PROJECT_ROOT / "data" / "thu ep" / "preprocessed"
REVE_MODEL_PATH = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-base"
REVE_POS_PATH   = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-positions"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "lora_checkpoints"

# W&B
USE_WANDB     = True
WANDB_PROJECT = "eeg-lora-thu-ep"
WANDB_ENTITY  = "zl-tudelft-thesis"

# Window parameters
SAMPLING_RATE = 200
WINDOW_SIZE   = 1600   # 8 s at 200 Hz
STRIDE        = 800    # 4 s at 200 Hz

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
ACCELERATOR = "gpu"  if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if sys.platform == "win32" else 4

# Trainer knobs
ACCUMULATE_GRAD_BATCHES = 1 # 4 # MODIFY ACCUMULATE FACTOR
GRADIENT_CLIP_VAL       = 1.0
PRECISION               = "16-mixed"


# Cross-fold summary

def _print_fold_summary(fold_results: list[dict]) -> None:
    print(f"\n{'═' * COL_W}")
    print(f"  CROSS-FOLD SUMMARY  ({len(fold_results)} folds  |  task={TASK_MODE}  |  LoRA r={LORA_RANK})")
    print(f"{'═' * COL_W}")

    col = f"{'Fold':>5}  {'ValAcc':>8}  {'ValAUROC':>9}  {'ValF1':>8}  {'Epochs':>7}  {'BestEp':>7}"
    print(col)
    print("─" * len(col))

    accs, aurocs, f1s = [], [], []
    for r in fold_results:
        acc   = r.get("val_acc")
        auroc = r.get("val_auroc")
        f1    = r.get("val_f1")
        if acc   is not None: accs.append(acc)
        if auroc is not None: aurocs.append(auroc)
        if f1    is not None: f1s.append(f1)

        print(
            f"{r['fold']:>5}  "
            f"{fmt_metric(acc   if acc   is not None else float('nan')):>8}  "
            f"{fmt_metric(auroc if auroc is not None else float('nan')):>9}  "
            f"{fmt_metric(f1    if f1    is not None else float('nan')):>8}  "
            f"{r.get('epochs_trained', 0):>7}  "
            f"{r.get('best_epoch', 0):>7}"
        )

    print("─" * len(col))

    def _stat(vals: list[float]) -> tuple[str, str]:
        if not vals:
            return "n/a", "n/a"
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.4f}", f"{std:.4f}"

    acc_mean,   acc_std   = _stat(accs)
    auroc_mean, auroc_std = _stat(aurocs)
    f1_mean,    f1_std    = _stat(f1s)

    print(f"{'Mean':>5}  {acc_mean:>8}  {auroc_mean:>9}  {f1_mean:>8}")
    print(f"{'Std':>5}  {acc_std:>8}  {auroc_std:>9}  {f1_std:>8}")
    print(f"{'═' * COL_W}")

    # Save aggregate JSON
    w_s  = round(WINDOW_SIZE / SAMPLING_RATE)
    st_s = round(STRIDE / SAMPLING_RATE)
    cls_tag = "_cls" if UNFREEZE_CLS else ""
    agg_path = OUTPUT_DIR / f"summary_{TASK_MODE}_w{w_s}s{st_s}_r{LORA_RANK}{cls_tag}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = {
        "task_mode":    TASK_MODE,
        "lora_rank":    LORA_RANK,
        "unfreeze_cls": UNFREEZE_CLS,
        "completed_at": datetime.datetime.now().isoformat(),
        "n_folds_run":  len(fold_results),
        "mean": {
            "val_acc":   round(statistics.mean(accs),   4) if accs   else None,
            "val_auroc": round(statistics.mean(aurocs), 4) if aurocs else None,
            "val_f1":    round(statistics.mean(f1s),    4) if f1s    else None,
        },
        "std": {
            "val_acc":   round(statistics.stdev(accs),   4) if len(accs)   > 1 else 0.0,
            "val_auroc": round(statistics.stdev(aurocs), 4) if len(aurocs) > 1 else 0.0,
            "val_f1":    round(statistics.stdev(f1s),    4) if len(f1s)    > 1 else 0.0,
        },
        "folds": fold_results,
    }
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate results saved → {agg_path}")


# Per-fold training

def run_fold(
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
    config: THUEPConfig,
) -> dict:
    """Train REVELoRAModule for one cross-subject fold."""

    print(f"\n{'#' * COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  task={TASK_MODE}  |  "
        f"LoRA r={LORA_RANK}  |  device={DEVICE}"
    )
    print(
        f"  train: {len(train_subject_ids)} subjects  |  "
        f"val: {len(val_subject_ids)} subjects"
    )
    print(f"{'#' * COL_W}")

    # Datasets
    t_data = time.time()
    print("Building datasets …", end="  ", flush=True)
    train_ds = THUEPWindowDataset(
        subject_ids=train_subject_ids,
        task_mode=TASK_MODE,
        data_root=DATA_ROOT,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    val_ds = THUEPWindowDataset(
        subject_ids=val_subject_ids,
        task_mode=TASK_MODE,
        data_root=DATA_ROOT,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    print(
        f"done in {fmt_dur(time.time() - t_data)}  "
        f"(train={len(train_ds):,}  val={len(val_ds):,})"
    )

    pin = DEVICE == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    # Model (fresh REVE from disk each fold)
    print("Loading REVE + applying LoRA …", end="  ", flush=True)
    t_model = time.time()
    model = REVELoRAModule(
        reve_model_path=REVE_MODEL_PATH,
        reve_pos_path=REVE_POS_PATH,
        config=config,
        num_classes=2 if TASK_MODE == "binary" else 9,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        lr_head=LR_HEAD,
        lr_lora=LR_LORA,
        phase1_epochs=PHASE1_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        unfreeze_cls=UNFREEZE_CLS,
        mixup_alpha=MIXUP_ALPHA,
    )
    print(f"done in {fmt_dur(time.time() - t_model)}")
    model.reve.print_trainable_parameters()

    # Logging + callbacks
    w_s  = round(WINDOW_SIZE / SAMPLING_RATE)
    st_s = round(STRIDE / SAMPLING_RATE)
    cls_tag = "_cls" if UNFREEZE_CLS else ""
    run_name = f"lora_{TASK_MODE}_w{w_s}s{st_s}_r{LORA_RANK}{cls_tag}_fold_{fold_idx}"
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hparams = {
        "task_mode":       TASK_MODE,
        "fold":            fold_idx,
        "n_folds":         N_FOLDS,
        "batch_size":      BATCH_SIZE,
        "accumulate_grad": ACCUMULATE_GRAD_BATCHES,
        "effective_batch":  BATCH_SIZE * ACCUMULATE_GRAD_BATCHES,
        "lr_head":         LR_HEAD,
        "lr_lora":         LR_LORA,
        "max_epochs":      MAX_EPOCHS,
        "phase1_epochs":   PHASE1_EPOCHS,
        "lora_rank":       LORA_RANK,
        "lora_alpha":      LORA_ALPHA,
        "lora_dropout":    LORA_DROPOUT,
        "head_dropout":    HEAD_DROPOUT,
        "unfreeze_cls":    UNFREEZE_CLS,
        "warmup_epochs":   WARMUP_EPOCHS,
        "mixup_alpha":     MIXUP_ALPHA,
        "window_size":     WINDOW_SIZE,
        "stride":          STRIDE,
        "gradient_clip":   GRADIENT_CLIP_VAL,
        "precision":       PRECISION,
        "n_train_subjects": len(train_subject_ids),
        "n_val_subjects":   len(val_subject_ids),
        "n_train_windows":  len(train_ds),
        "n_val_windows":    len(val_ds),
    }

    if USE_WANDB:
        logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=f"lora_{TASK_MODE}_w{w_s}s{st_s}_r{LORA_RANK}{cls_tag}",
            config=hparams,
            log_model=False,
            reinit=True,
        )
        print(f"Logging to W&B  project={WANDB_PROJECT}  entity={WANDB_ENTITY}")
    else:
        logger = CSVLogger(save_dir=str(ckpt_dir), name="csv_logs")
        print(f"W&B disabled. Logging to CSV → {ckpt_dir / 'csv_logs'}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        verbose=False,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/acc",
        patience=20,
        mode="max",
        verbose=False,
    )

    def _phase_column(trainer, pl_module):
        return f"P{pl_module.phase}"

    summary_cb = EpochSummaryCallback(
        output_dir=ckpt_dir,
        fold_idx=fold_idx,
        task_mode=TASK_MODE,
        train_subjects=train_subject_ids,
        val_subjects=val_subject_ids,
        hparams=hparams,
        extra_columns=_phase_column,
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=1,
        precision=PRECISION,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, summary_cb],
        enable_progress_bar=False,
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    )

    # Train
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    finally:
        del train_ds, val_ds, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if USE_WANDB:
            wandb.finish()

    # Save adapter + head weights
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        best_model = REVELoRAModule.load_from_checkpoint(
            best_ckpt,
            reve_model_path=REVE_MODEL_PATH,
            reve_pos_path=REVE_POS_PATH,
            config=config,
            weights_only=False,  # PyTorch 2.6+ requires this for pathlib paths
        )
        # Save LoRA adapter weights
        adapter_path = ckpt_dir / "lora_adapter_weights.pt"
        adapter_sd = {
            n: p.cpu() for n, p in best_model.reve.named_parameters() if "lora_" in n
        }
        torch.save(adapter_sd, adapter_path)
        print(f"LoRA adapter weights saved → {adapter_path}")

        # Save head weights
        head_path = ckpt_dir / "head_weights.pt"
        torch.save(best_model.head.state_dict(), head_path)
        print(f"Head weights saved → {head_path}")

        del best_model
    else:
        print("Warning: no checkpoint found, weights not saved.")

    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best val/acc:    {checkpoint_cb.best_model_score:.4f}")

    # Cleanup model before next fold
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return best metrics for aggregation
    valid_rows = [r for r in summary_cb.epoch_history if r["val_acc"] is not None]
    best_row   = max(valid_rows, key=lambda r: r["val_acc"]) if valid_rows else {}
    return {
        "fold":           fold_idx,
        "val_acc":        best_row.get("val_acc"),
        "val_auroc":      best_row.get("val_auroc"),
        "val_f1":         best_row.get("val_f1"),
        "best_epoch":     best_row.get("epoch"),
        "epochs_trained": len(summary_cb.epoch_history),
    }


# Main

def main() -> None:
    global TASK_MODE, DRY_RUN_FOLD, WINDOW_SIZE, STRIDE
    global LORA_RANK, LORA_ALPHA, PHASE1_EPOCHS, WARMUP_EPOCHS, UNFREEZE_CLS
    global LR_HEAD, LR_LORA, MIXUP_ALPHA, BATCH_SIZE

    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning of REVE on THU-EP"
    )
    parser.add_argument(
        "--task", choices=["binary", "9-class"], default=TASK_MODE,
        help="Classification task (default: %(default)s)",
    )
    parser.add_argument(
        "--fold", type=int, default=DRY_RUN_FOLD, metavar="N",
        help="Run only this fold index (1-10). Omit to run all folds.",
    )
    parser.add_argument(
        "--window", type=float, default=WINDOW_SIZE / SAMPLING_RATE, metavar="S",
        help="Window length in seconds (default: %(default)s s).",
    )
    parser.add_argument(
        "--stride", type=float, default=STRIDE / SAMPLING_RATE, metavar="S",
        help="Stride between windows in seconds (default: %(default)s s).",
    )
    parser.add_argument(
        "--rank", type=int, default=LORA_RANK, metavar="R",
        help="LoRA rank (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha", type=int, default=LORA_ALPHA, metavar="A",
        help="LoRA alpha scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--phase1-epochs", type=int, default=PHASE1_EPOCHS, metavar="N",
        help="Head-only epochs before unfreezing LoRA (default: %(default)s).",
    )
    parser.add_argument(
        "--phase2-warmup", type=int, default=WARMUP_EPOCHS, metavar="N",
        help="Linear LR warmup epochs at start of Phase 2 (default: %(default)s; 0 = off).",
    )
    parser.add_argument(
        "--unfreeze-cls", action="store_true", default=UNFREEZE_CLS,
        help="Unfreeze cls_query_token in Phase 2.",
    )
    parser.add_argument(
        "--lr-head", type=float, default=LR_HEAD, metavar="F",
        help="Head learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--lr-lora", type=float, default=LR_LORA, metavar="F",
        help="LoRA adapter learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=MIXUP_ALPHA, metavar="F",
        help="Mixup Beta distribution param; 0 = disabled (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, metavar="N",
        help="Batch size per GPU (default: %(default)s).",
    )
    args = parser.parse_args()

    TASK_MODE     = args.task
    DRY_RUN_FOLD  = args.fold
    WINDOW_SIZE   = round(args.window * SAMPLING_RATE)
    STRIDE        = round(args.stride * SAMPLING_RATE)
    LORA_RANK     = args.rank
    LORA_ALPHA    = args.alpha
    PHASE1_EPOCHS = args.phase1_epochs
    WARMUP_EPOCHS = args.phase2_warmup
    UNFREEZE_CLS  = args.unfreeze_cls
    LR_HEAD       = args.lr_head
    LR_LORA       = args.lr_lora
    MIXUP_ALPHA   = args.mixup_alpha
    BATCH_SIZE    = args.batch_size

    L.seed_everything(42, workers=True)

    config = THUEPConfig()
    all_subjects = get_all_subjects()

    print(f"\nTotal valid subjects: {len(all_subjects)}  (excluded: {EXCLUDED_SUBJECTS})")
    print(
        f"Device: {DEVICE}  |  task_mode: {TASK_MODE}  |  dry_run_fold: {DRY_RUN_FOLD}  |  "
        f"window: {WINDOW_SIZE / SAMPLING_RATE}s ({WINDOW_SIZE} pts)  "
        f"stride: {STRIDE / SAMPLING_RATE}s ({STRIDE} pts)"
    )
    print(
        f"LoRA: rank={LORA_RANK}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}  "
        f"phase1_epochs={PHASE1_EPOCHS}  warmup_epochs={WARMUP_EPOCHS}  unfreeze_cls={UNFREEZE_CLS}"
    )

    # Fold splits
    folds = get_kfold_splits(all_subjects)
    folds_to_run = (
        [(DRY_RUN_FOLD, folds[DRY_RUN_FOLD - 1])]
        if DRY_RUN_FOLD is not None
        else [(i + 1, folds[i]) for i in range(N_FOLDS)]
    )

    # Train fold(s)
    fold_results: list[dict] = []
    for fold_idx, (train_idx, val_idx) in folds_to_run:
        train_subjects = [all_subjects[i] for i in train_idx]
        val_subjects   = [all_subjects[i] for i in val_idx]
        result = run_fold(fold_idx, train_subjects, val_subjects, config)
        fold_results.append(result)

    print("\nAll folds complete.")

    if len(fold_results) > 1:
        _print_fold_summary(fold_results)


if __name__ == "__main__":
    main()
