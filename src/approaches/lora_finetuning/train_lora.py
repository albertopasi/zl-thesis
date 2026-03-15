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
4. Print cross-fold summary (mean +/- std) and save aggregate JSON.

Run with:
    # Binary, all folds (default), attention pooling
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary

    # 9-class, single fold (dry run)
    uv run python -m src.approaches.lora_finetuning.train_lora --task 9-class --fold 1

    # Custom rank and window
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary --rank 16 --window 10 --stride 10

    # No-pool flat mode
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary --no-pooling --no-pool-mode flat

    # Generalization (held-out stimuli)
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary --generalization

    # Multi-seed generalization
    uv run python -m src.approaches.lora_finetuning.train_lora --task binary --generalization --gen-seeds 123 456 789 101 202
"""

from __future__ import annotations

import argparse
import gc
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
from src.thu_ep.folds import (
    get_all_subjects, get_kfold_splits, get_stimulus_generalization_split, N_FOLDS,
)
from src.thu_ep.callbacks import EpochSummaryCallback, fmt_dur, COL_W

from src.approaches.lora_finetuning.model import REVELoRAModule, WarmupSchedulerCallback
from src.approaches.lora_finetuning.config import (
    LoRAConfig,
    DATA_ROOT, REVE_MODEL_PATH, REVE_POS_PATH, OUTPUT_DIR,
    USE_WANDB, WANDB_PROJECT, WANDB_ENTITY,
    SAMPLING_RATE, DEVICE, ACCELERATOR, NUM_WORKERS,
    ACCUMULATE_GRAD_BATCHES, GRADIENT_CLIP_VAL, PRECISION,
)
from src.approaches.lora_finetuning.summary import (
    print_fold_summary, print_cross_seed_summary,
)


# ── Per-fold training ─────────────────────────────────────────────────────────

def run_fold(
    cfg: LoRAConfig,
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
    thu_config: THUEPConfig,
    train_stimuli: set[int] | None = None,
    val_stimuli: set[int] | None = None,
    gen_seed: int | None = None,
) -> dict:
    """Train REVELoRAModule for one cross-subject fold."""

    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""
    gen_tag = f"  |  GENERALIZATION (held-out stimuli){seed_label}" if cfg.generalization else ""

    print(f"\n{'#' * COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  task={cfg.task_mode}  |  "
        f"LoRA {cfg.rank_tag}  |  {cfg.pool_tag}{cfg.cls_tag}  |  device={DEVICE}{gen_tag}"
    )
    print(
        f"  train: {len(train_subject_ids)} subjects  |  "
        f"val: {len(val_subject_ids)} subjects"
    )
    if train_stimuli is not None:
        print(
            f"  train stimuli: {sorted(train_stimuli)}  |  "
            f"val stimuli: {sorted(val_stimuli)}"
        )
    print(f"{'#' * COL_W}")

    # ── Datasets ──────────────────────────────────────────────────────────
    t_data = time.time()
    print("Building datasets ...", end="  ", flush=True)
    train_ds = THUEPWindowDataset(
        subject_ids=train_subject_ids,
        task_mode=cfg.task_mode,
        data_root=DATA_ROOT,
        window_size=cfg.window_size,
        stride=cfg.stride,
        stimulus_filter=train_stimuli,
    )
    val_ds = THUEPWindowDataset(
        subject_ids=val_subject_ids,
        task_mode=cfg.task_mode,
        data_root=DATA_ROOT,
        window_size=cfg.window_size,
        stride=cfg.stride,
        stimulus_filter=val_stimuli,
    )
    print(
        f"done in {fmt_dur(time.time() - t_data)}  "
        f"(train={len(train_ds):,}  val={len(val_ds):,})"
    )

    pin = DEVICE == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("Loading REVE + applying LoRA ...", end="  ", flush=True)
    t_model = time.time()
    model = REVELoRAModule(
        reve_model_path=REVE_MODEL_PATH,
        reve_pos_path=REVE_POS_PATH,
        config=thu_config,
        num_classes=cfg.num_classes,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        head_dropout=cfg.head_dropout,
        lr_head=cfg.lr_head,
        lr_lora=cfg.lr_lora,
        weight_decay=cfg.weight_decay,
        phase1_epochs=cfg.phase1_epochs,
        unfreeze_cls=cfg.unfreeze_cls,
        mixup_alpha=cfg.mixup_alpha,
        use_pooling=cfg.use_pooling,
        no_pool_mode=cfg.no_pool_mode,
    )
    print(f"done in {fmt_dur(time.time() - t_model)}")
    model.reve.print_trainable_parameters()

    # ── Logging + callbacks ───────────────────────────────────────────────
    run_name = cfg.run_name(fold_idx, gen_seed)
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hparams = cfg.hparams_dict(
        fold_idx=fold_idx,
        n_folds=N_FOLDS,
        n_train_subjects=len(train_subject_ids),
        n_val_subjects=len(val_subject_ids),
        n_train_windows=len(train_ds),
        n_val_windows=len(val_ds),
        gen_seed=gen_seed,
    )

    if USE_WANDB:
        logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=cfg.group_name(),
            config=hparams,
            log_model=False,
            reinit=True,
        )
        print(f"Logging to W&B  project={WANDB_PROJECT}  entity={WANDB_ENTITY}")
    else:
        logger = CSVLogger(save_dir=str(ckpt_dir), name="csv_logs")
        print(f"W&B disabled. Logging to CSV -> {ckpt_dir / 'csv_logs'}")

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
        task_mode=cfg.task_mode,
        train_subjects=train_subject_ids,
        val_subjects=val_subject_ids,
        hparams=hparams,
        extra_columns=_phase_column,
    )

    warmup_cb = WarmupSchedulerCallback(
        phase1_epochs=cfg.phase1_epochs,
        lr_head=cfg.lr_head,
        lr_lora=cfg.lr_lora,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=ACCELERATOR,
        devices=1,
        precision=PRECISION,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, summary_cb, warmup_cb],
        enable_progress_bar=False,
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    finally:
        del train_ds, val_ds, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if USE_WANDB:
            wandb.finish()

    # ── Save adapter + head weights ───────────────────────────────────────
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        best_model = REVELoRAModule.load_from_checkpoint(
            best_ckpt,
            reve_model_path=REVE_MODEL_PATH,
            reve_pos_path=REVE_POS_PATH,
            config=thu_config,
            weights_only=False,
        )
        adapter_path = ckpt_dir / "lora_adapter_weights.pt"
        adapter_sd = {
            n: p.cpu() for n, p in best_model.reve.named_parameters() if "lora_" in n
        }
        torch.save(adapter_sd, adapter_path)
        print(f"LoRA adapter weights saved -> {adapter_path}")

        if best_model.head is not None:
            head_path = ckpt_dir / "head_weights.pt"
            torch.save(best_model.head.state_dict(), head_path)
            print(f"Head weights saved -> {head_path}")

        del best_model
    else:
        print("Warning: no checkpoint found, weights not saved.")

    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best val/acc:    {checkpoint_cb.best_model_score:.4f}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> LoRAConfig:
    """Parse CLI arguments and return a populated LoRAConfig."""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning of REVE on THU-EP"
    )
    parser.add_argument(
        "--task", choices=["binary", "9-class"], default="binary",
        help="Classification task (default: %(default)s)",
    )
    parser.add_argument(
        "--fold", type=int, default=None, metavar="N",
        help="Run only this fold index (1-10). Omit to run all folds.",
    )
    parser.add_argument(
        "--window", type=float, default=10.0, metavar="S",
        help="Window length in seconds (default: %(default)s s).",
    )
    parser.add_argument(
        "--stride", type=float, default=10.0, metavar="S",
        help="Stride between windows in seconds (default: %(default)s s).",
    )
    parser.add_argument(
        "--rank", type=int, default=8, metavar="R",
        help="LoRA rank (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha", type=int, default=16, metavar="A",
        help="LoRA alpha scaling (default: %(default)s).",
    )
    parser.add_argument(
        "--phase1-epochs", type=int, default=10, metavar="N",
        help="Head-only epochs before unfreezing LoRA (default: %(default)s).",
    )
    parser.add_argument(
        "--unfreeze-cls", action="store_true", default=False,
        help="Unfreeze cls_query_token in Phase 2.",
    )
    parser.add_argument(
        "--lr-head", type=float, default=1e-3, metavar="F",
        help="Head learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--lr-lora", type=float, default=1e-4, metavar="F",
        help="LoRA adapter learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, metavar="F",
        help="Weight decay for AdamW (default: %(default)s).",
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.0, metavar="F",
        help="Mixup Beta distribution param; 0 = disabled (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Batch size per GPU (default: %(default)s).",
    )
    parser.add_argument(
        "--no-pooling", dest="no_pooling", action="store_true", default=False,
        help="Bypass attention pooling; use raw patch embeddings flattened by --no-pool-mode.",
    )
    parser.add_argument(
        "--no-pool-mode", dest="no_pool_mode", choices=["mean", "flat"], default="mean",
        help="How to flatten raw patches when --no-pooling is set: "
             "'mean' (avg over channels -> H*512, default) or 'flat' (C*H*512).",
    )
    parser.add_argument(
        "--generalization", action="store_true", default=False,
        help="Stimulus-generalization evaluation: train on 2/3 of stimuli per emotion, "
             "test on held-out 1/3 stimuli from unseen subjects.",
    )
    parser.add_argument(
        "--gen-seeds", dest="gen_seeds", type=int, nargs="+", default=[123],
        help="Stimulus split seeds for generalization mode (default: [123]). "
             "Pass multiple seeds for robustness check, e.g. --gen-seeds 123 456 789 101 202",
    )
    args = parser.parse_args()

    return LoRAConfig(
        task_mode=args.task,
        fold=args.fold,
        window_size=round(args.window * SAMPLING_RATE),
        stride=round(args.stride * SAMPLING_RATE),
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        phase1_epochs=args.phase1_epochs,
        unfreeze_cls=args.unfreeze_cls,
        lr_head=args.lr_head,
        lr_lora=args.lr_lora,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup_alpha,
        batch_size=args.batch_size,
        use_pooling=not args.no_pooling,
        no_pool_mode=args.no_pool_mode,
        generalization=args.generalization,
        gen_seeds=args.gen_seeds,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()

    L.seed_everything(42, workers=True)

    thu_config = THUEPConfig()
    all_subjects = get_all_subjects()

    print(f"\nTotal valid subjects: {len(all_subjects)}  (excluded: {EXCLUDED_SUBJECTS})")
    print(
        f"Device: {DEVICE}  |  task_mode: {cfg.task_mode}  |  fold: {cfg.fold}  |  "
        f"window: {cfg.window_size / SAMPLING_RATE}s ({cfg.window_size} pts)  "
        f"stride: {cfg.stride / SAMPLING_RATE}s ({cfg.stride} pts)  "
        f"generalization: {cfg.generalization}"
    )
    print(
        f"LoRA: rank={cfg.lora_rank}  alpha={cfg.lora_alpha}  dropout={cfg.lora_dropout}  "
        f"phase1_epochs={cfg.phase1_epochs}  unfreeze_cls={cfg.unfreeze_cls}  "
        f"pooling={cfg.pool_tag}"
    )

    # Fold splits
    folds = get_kfold_splits(all_subjects)
    folds_to_run = (
        [(cfg.fold, folds[cfg.fold - 1])]
        if cfg.fold is not None
        else [(i + 1, folds[i]) for i in range(N_FOLDS)]
    )

    # Seed list
    seed_list: list[int | None] = [None]
    if cfg.generalization:
        seed_list = cfg.gen_seeds
        print(f"\nGeneralization mode: {len(seed_list)} seed(s) = {seed_list}")

    # Outer loop over seeds, inner loop over folds
    seed_summaries: list[dict] = []

    for seed in seed_list:
        train_stimuli: set[int] | None = None
        val_stimuli: set[int] | None = None
        gen_seed: int | None = None

        if seed is not None:
            gen_seed = seed
            train_stimuli, val_stimuli = get_stimulus_generalization_split(
                cfg.task_mode, seed=seed,
            )
            print(f"\n{'=' * COL_W}")
            print(f"  SEED {seed}  |  {len(train_stimuli)} train stimuli, {len(val_stimuli)} held-out stimuli")
            print(f"  Train: {sorted(train_stimuli)}")
            print(f"  Test:  {sorted(val_stimuli)}")
            print(f"{'=' * COL_W}")

        fold_results: list[dict] = []
        for fold_idx, (train_idx, val_idx) in folds_to_run:
            train_subjects = [all_subjects[i] for i in train_idx]
            val_subjects   = [all_subjects[i] for i in val_idx]
            result = run_fold(
                cfg, fold_idx, train_subjects, val_subjects, thu_config,
                train_stimuli=train_stimuli, val_stimuli=val_stimuli,
                gen_seed=gen_seed,
            )
            fold_results.append(result)

        if len(fold_results) > 1:
            print_fold_summary(cfg, fold_results, gen_seed=gen_seed)

        if gen_seed is not None:
            accs   = [r["val_acc"]   for r in fold_results if r.get("val_acc")   is not None]
            aurocs = [r["val_auroc"] for r in fold_results if r.get("val_auroc") is not None]
            f1s    = [r["val_f1"]    for r in fold_results if r.get("val_f1")    is not None]
            seed_summaries.append({
                "seed":       gen_seed,
                "mean_acc":   round(statistics.mean(accs),   4) if accs   else None,
                "mean_auroc": round(statistics.mean(aurocs), 4) if aurocs else None,
                "mean_f1":    round(statistics.mean(f1s),    4) if f1s    else None,
                "folds":      fold_results,
            })

    print("\nAll folds complete.")

    if len(seed_summaries) > 1:
        print_cross_seed_summary(cfg, seed_summaries)


if __name__ == "__main__":
    main()
