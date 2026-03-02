"""
train_lp.py — Main execution script for the Linear Probing (LP) run.

Workflow
--------
1. Build subject list (79 subjects).
2. Pre-compute REVE embeddings ONCE per subject (skipped if already cached).
   Stored as: data/thu ep/embeddings/{task_mode}/sub_XX.pt
   Each file contains the 512-D embeddings for ALL valid windows of that subject.
   This is done only once — fold splits just select from the per-subject cache.
3. Apply 10-fold cross-subject KFold split.
4. DRY RUN: only process Fold 1.
5. Load embeddings for train/val subjects from cache (fast concat, no REVE re-run).
6. Train a linear classifier on the combined embeddings.
7. Log to Weights & Biases, save classifier weights and a results JSON.

Run with:
    # Binary, all folds (default)
    uv run python -m src.approaches.linear_probing.train_lp --task binary

    # 9-class, all folds
    uv run python -m src.approaches.linear_probing.train_lp --task 9-class

    # Dry run: binary, fold 1 only
    uv run python -m src.approaches.linear_probing.train_lp --task binary --fold 1

    # Specify window size and stride
    uv run python -m src.approaches.linear_probing.train_lp --window 8 --stride 4   # default

"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import wandb

from src.thu_ep.config import THUEPConfig
from src.thu_ep.dataset import EXCLUDED_SUBJECTS, THUEPWindowDataset
from src.thu_ep.folds import get_all_subjects, get_kfold_splits, N_FOLDS
from src.approaches.linear_probing.model import EmbeddingExtractor, LinearProber


# Configuration

TASK_MODE    = "binary"   # 'binary' or '9-class'
MAX_EPOCHS   = 80
BATCH_SIZE   = 64
LR           = 1e-3
EMBED_DIM    = 512

# Only fold 1 is run as a dry run; set to None to run all folds.
DRY_RUN_FOLD = None

# Paths (relative to project root)
DATA_ROOT       = PROJECT_ROOT / "data" / "thu ep" / "preprocessed"
EMBEDDINGS_DIR  = PROJECT_ROOT / "data" / "thu ep" / "embeddings"
REVE_MODEL_PATH = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-base"
REVE_POS_PATH   = PROJECT_ROOT / "models" / "reve_pretrained_original" / "reve-positions"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "lp_checkpoints"

# W&B — set USE_WANDB=False to skip W&B and log to CSV only.
USE_WANDB     = True
WANDB_PROJECT = "eeg-lp-thu-ep"
WANDB_ENTITY  = "zl-tudelft-thesis"   # e.g. "my-team". None = W&B picks the default entity.


# Window parameters (timepoints = seconds × SAMPLING_RATE)
SAMPLING_RATE = 200  # Hz
WINDOW_SIZE = 1600   # 8 s at 200 Hz
STRIDE      = 800    # 4 s at 200 Hz

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
ACCELERATOR = "gpu"  if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if sys.platform == "win32" else 4


# Terminal output helpers

_COL_W = 90
_SEP   = "─" * _COL_W
_HEADER = (
    f"{'Epoch':>6}  {'EpTime':>7}  {'Elapsed':>8}  "
    f"{'TrLoss':>8}  {'TrAcc':>7}  "
    f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaAUROC':>8}  {'VaF1':>7}"
)


def _fmt_dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _v(t) -> float:
    if isinstance(t, torch.Tensor):
        return float(t.item())
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return float("nan")
    return float(t)


def _fmt(val: float, width: int = 8, decimals: int = 4) -> str:
    if math.isnan(val):
        return f"{'n/a':>{width}}"
    return f"{val:>{width}.{decimals}f}"


# EpochSummaryCallback
class EpochSummaryCallback(L.Callback):
    """
    Prints a per-epoch metrics table to the terminal and saves a results JSON
    at the end of training.
    """

    def __init__(
        self,
        output_dir: Path,
        fold_idx: int,
        task_mode: str,
        train_subjects: list[int],
        val_subjects: list[int],
        hparams: dict,
    ) -> None:
        self.output_dir     = Path(output_dir)
        self.fold_idx       = fold_idx
        self.task_mode      = task_mode
        self.train_subjects = train_subjects
        self.val_subjects   = val_subjects
        self.hparams        = hparams
        self.epoch_history: list[dict] = []
        self._fit_start:   float | None = None
        self._epoch_start: float | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._fit_start = time.time()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.time()
        # Print header after epoch 0 starts — this fires after the sanity-check
        # warnings, so the header always appears directly above the first row.
        if trainer.current_epoch == 0:
            print(f"\n{_SEP}")
            print(_HEADER)
            print(_SEP)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        epoch    = trainer.current_epoch + 1
        m        = trainer.callback_metrics
        ep_time  = time.time() - self._epoch_start
        elapsed  = time.time() - self._fit_start

        tr_loss  = _v(m.get("train/loss"))
        tr_acc   = _v(m.get("train/acc"))
        va_loss  = _v(m.get("val/loss"))
        va_acc   = _v(m.get("val/acc"))
        va_auroc = _v(m.get("val/auroc"))
        va_f1    = _v(m.get("val/f1"))

        avg_ep    = elapsed / epoch
        remaining = avg_ep * (trainer.max_epochs - epoch)
        eta_str   = f"ETA {_fmt_dur(remaining)}" if epoch < trainer.max_epochs else "done"

        print(
            f"{epoch:>6}  {_fmt_dur(ep_time):>7}  {_fmt_dur(elapsed):>8}  "
            f"{_fmt(tr_loss):>8}  {_fmt(tr_acc):>7}  "
            f"{_fmt(va_loss):>8}  {_fmt(va_acc):>7}  "
            f"{_fmt(va_auroc):>8}  {_fmt(va_f1):>7}  ({eta_str})"
        )

        self.epoch_history.append({
            "epoch":       epoch,
            "epoch_time_s": round(ep_time, 2),
            "train_loss":  None if math.isnan(tr_loss)  else round(tr_loss,  4),
            "train_acc":   None if math.isnan(tr_acc)   else round(tr_acc,   4),
            "val_loss":    None if math.isnan(va_loss)  else round(va_loss,  4),
            "val_acc":     None if math.isnan(va_acc)   else round(va_acc,   4),
            "val_auroc":   None if math.isnan(va_auroc) else round(va_auroc, 4),
            "val_f1":      None if math.isnan(va_f1)    else round(va_f1,    4),
        })

    def on_fit_end(self, trainer, pl_module) -> None:
        total_time = time.time() - self._fit_start
        valid_rows = [r for r in self.epoch_history if r["val_acc"] is not None]

        print(_SEP)
        print(
            f"Training complete — {len(self.epoch_history)} epochs  |  "
            f"total time: {_fmt_dur(total_time)}"
        )
        if valid_rows:
            best = max(valid_rows, key=lambda r: r["val_acc"])
            print(
                f"Best  epoch={best['epoch']:>3}  val_acc={best['val_acc']:.4f}  "
                f"val_auroc={best['val_auroc']:.4f}  val_f1={best['val_f1']:.4f}"
            )
        print(_SEP)
        self._save_results(total_time)

    def _save_results(self, total_time: float) -> None:
        valid_rows = [r for r in self.epoch_history if r["val_acc"] is not None]
        best = max(valid_rows, key=lambda r: r["val_acc"]) if valid_rows else {}

        results = {
            "fold":          self.fold_idx,
            "task_mode":     self.task_mode,
            "completed_at":  datetime.datetime.now().isoformat(),
            "hyperparams":   self.hparams,
            "train_subjects": self.train_subjects,
            "val_subjects":   self.val_subjects,
            "best": {
                "epoch":     best.get("epoch"),
                "val_acc":   best.get("val_acc"),
                "val_auroc": best.get("val_auroc"),
                "val_f1":    best.get("val_f1"),
            },
            "total_time_s":   round(total_time, 2),
            "epochs_trained": len(self.epoch_history),
            "epoch_history":  self.epoch_history,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results JSON saved → {json_path}")


# Cross-fold summary
def _print_fold_summary(fold_results: list[dict]) -> None:
    """
    Print a per-fold metrics table and mean ± std across all completed folds.
    Also saves an aggregate JSON to OUTPUT_DIR/summary_{TASK_MODE}.json.
    """
    print(f"\n{'═' * _COL_W}")
    print(f"  CROSS-FOLD SUMMARY  ({len(fold_results)} folds  |  task={TASK_MODE})")
    print(f"{'═' * _COL_W}")

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
            f"{_fmt(acc   if acc   is not None else float('nan')):>8}  "
            f"{_fmt(auroc if auroc is not None else float('nan')):>9}  "
            f"{_fmt(f1    if f1    is not None else float('nan')):>8}  "
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
    print(f"{'═' * _COL_W}")

    # Save aggregate JSON
    agg_path = OUTPUT_DIR / f"summary_{TASK_MODE}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agg = {
        "task_mode":    TASK_MODE,
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


# Per-subject embedding cache helpers

def subject_cache_path(subject_id: int, task_mode: str) -> Path:
    """Return the path for a single subject's pre-computed embedding file."""
    return EMBEDDINGS_DIR / task_mode / f"ws{WINDOW_SIZE}_st{STRIDE}" / f"sub_{subject_id:02d}.pt"


def precompute_all_subjects(
    all_subjects: list[int],
    config: THUEPConfig,
) -> None:
    """
    Pre-compute REVE embeddings for every subject that is not yet cached.

    Embeddings are stored per-subject at:
        EMBEDDINGS_DIR / task_mode / ws{WINDOW_SIZE}_st{STRIDE} / sub_XX.pt

    This function is called once before any fold loop. Subsequent fold runs
    simply concatenate the already-saved per-subject tensors — no REVE re-runs.
    """
    missing = [
        sid for sid in all_subjects
        if not subject_cache_path(sid, TASK_MODE).exists()
    ]

    if not missing:
        print(f"All {len(all_subjects)} subject embedding caches found. Skipping REVE extraction.")
        return

    print(f"\n{len(missing)}/{len(all_subjects)} subjects need embedding extraction.")

    extractor = EmbeddingExtractor(
        reve_model_path=REVE_MODEL_PATH,
        reve_pos_path=REVE_POS_PATH,
        config=config,
        device=DEVICE,
    )

    t0 = time.time()
    try:
        for i, sid in enumerate(missing, 1):
            t_sub = time.time()
            print(f"\n[{i}/{len(missing)}] Subject {sid:02d}", end="  ")

            dataset = THUEPWindowDataset(
                subject_ids=[sid],
                task_mode=TASK_MODE,
                data_root=DATA_ROOT,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
            n_win = len(dataset)
            print(f"({n_win} windows)", end="  ", flush=True)

            embeddings, labels = extractor.extract_embeddings(
                dataset, batch_size=BATCH_SIZE
            )
            EmbeddingExtractor.save_embeddings(
                embeddings, labels, subject_cache_path(sid, TASK_MODE)
            )

            elapsed_sub = time.time() - t_sub
            avg_per_sub = (time.time() - t0) / i
            eta         = avg_per_sub * (len(missing) - i)
            print(f"done in {_fmt_dur(elapsed_sub)}  (ETA {_fmt_dur(eta)})")

            del dataset, embeddings, labels
            gc.collect()

    finally:
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll subjects done. Total extraction time: {_fmt_dur(time.time() - t0)}")


def load_subjects_embeddings(
    subject_ids: list[int],
    task_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load and concatenate per-subject embedding files for the given subjects.

    Args:
        subject_ids: List of subject IDs to load.
        task_mode:   'binary' or '9-class' (determines which cache folder to use).

    Returns:
        embeddings: Tensor of shape (N, 512)
        labels:     Tensor of shape (N,)
    """
    all_embs:   list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for sid in subject_ids:
        path = subject_cache_path(sid, task_mode)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        all_embs.append(payload["embeddings"])
        all_labels.append(payload["labels"])

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


# run_fold

def run_fold(
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
) -> None:
    """
    Train the linear classifier for one fold using pre-computed embeddings.
    Embeddings are loaded from per-subject cache files (fast concat, no REVE).
    """
    print(f"\n{'#' * _COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  task={TASK_MODE}  |  "
        f"max_epochs={MAX_EPOCHS}  |  device={DEVICE}"
    )
    print(
        f"  train: {len(train_subject_ids)} subjects  |  "
        f"val: {len(val_subject_ids)} subjects"
    )
    print(f"{'#' * _COL_W}")

    # Load pre-computed embeddings from per-subject cache files
    t_load = time.time()
    print("Loading embeddings from cache …", end="  ", flush=True)
    train_embs, train_lbls = load_subjects_embeddings(train_subject_ids, TASK_MODE)
    val_embs,   val_lbls   = load_subjects_embeddings(val_subject_ids,   TASK_MODE)
    print(
        f"done in {_fmt_dur(time.time() - t_load)}  "
        f"(train={train_embs.shape[0]:,}  val={val_embs.shape[0]:,})"
    )

    # Build DataLoaders directly from tensors (no disk I/O during training)
    train_loader = DataLoader(
        TensorDataset(train_embs, train_lbls),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        TensorDataset(val_embs, val_lbls),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Model, logger, callbacks
    num_classes = 2 if TASK_MODE == "binary" else 9
    model = LinearProber(num_classes=num_classes, embed_dim=EMBED_DIM, lr=LR)

    run_name = f"lp_{TASK_MODE}_fold_{fold_idx}"
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hparams = {
        "task_mode":        TASK_MODE,
        "fold":             fold_idx,
        "n_folds":          N_FOLDS,
        "batch_size":       BATCH_SIZE,
        "lr":               LR,
        "max_epochs":       MAX_EPOCHS,
        "window_size":      WINDOW_SIZE,
        "stride":           STRIDE,
        "n_train_subjects": len(train_subject_ids),
        "n_val_subjects":   len(val_subject_ids),
        "n_train_windows":  int(train_embs.shape[0]),
        "n_val_windows":    int(val_embs.shape[0]),
    }

    # Build logger: W&B if enabled, otherwise fall back to CSV.
    if USE_WANDB:
        logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=f"lp_{TASK_MODE}",
            config=hparams,
            log_model=False,
            reinit=True,    # allow multiple runs in the same process (one per fold)
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
        patience=10,
        mode="max",
        verbose=False,
    )

    summary_cb = EpochSummaryCallback(
        output_dir=ckpt_dir,
        fold_idx=fold_idx,
        task_mode=TASK_MODE,
        train_subjects=train_subject_ids,
        val_subjects=val_subject_ids,
        hparams=hparams,
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, summary_cb],
        log_every_n_steps=1,
        enable_progress_bar=False,   # replaced by EpochSummaryCallback table
        enable_model_summary=True,
    )

    # Train

    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    finally:
        del train_embs, train_lbls, val_embs, val_lbls, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Explicitly close the W&B run so the next fold can open a fresh one.
        if USE_WANDB:
            wandb.finish()

    # Save best classifier weights

    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        best_model = LinearProber.load_from_checkpoint(best_ckpt)
        weights_path = ckpt_dir / "classifier_weights.pt"
        torch.save(best_model.classifier.state_dict(), weights_path)
        print(f"Classifier weights (best epoch) saved → {weights_path}")
        del best_model
    else:
        print("Warning: no checkpoint found, weights not saved.")

    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best val/acc:    {checkpoint_cb.best_model_score:.4f}")

    # Return best metrics for cross-fold aggregation
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


def main() -> None:
    global TASK_MODE, DRY_RUN_FOLD, WINDOW_SIZE, STRIDE

    parser = argparse.ArgumentParser(description="Linear Probing on THU-EP with REVE embeddings")
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
    args = parser.parse_args()
    TASK_MODE    = args.task
    DRY_RUN_FOLD = args.fold
    WINDOW_SIZE  = round(args.window * SAMPLING_RATE)
    STRIDE       = round(args.stride * SAMPLING_RATE)

    L.seed_everything(42, workers=True)

    config = THUEPConfig()
    all_subjects = get_all_subjects()

    print(f"\nTotal valid subjects: {len(all_subjects)}  (excluded: {EXCLUDED_SUBJECTS})")
    print(
        f"Device: {DEVICE}  |  task_mode: {TASK_MODE}  |  dry_run_fold: {DRY_RUN_FOLD}  |  "
        f"window: {WINDOW_SIZE/SAMPLING_RATE}s ({WINDOW_SIZE} pts)  "
        f"stride: {STRIDE/SAMPLING_RATE}s ({STRIDE} pts)"
    )

    # Step 1: Pre-compute ONCE per subject (skips subjects already cached)
    precompute_all_subjects(all_subjects, config)

    # Step 2: 10-fold cross-subject split
    folds = get_kfold_splits(all_subjects)

    folds_to_run = (
        [(DRY_RUN_FOLD, folds[DRY_RUN_FOLD - 1])]
        if DRY_RUN_FOLD is not None
        else [(i + 1, folds[i]) for i in range(N_FOLDS)]
    )

    # Step 3: Train fold(s) — collect per-fold metrics for final summary
    fold_results: list[dict] = []
    for fold_idx, (train_idx, val_idx) in folds_to_run:
        train_subjects = [all_subjects[i] for i in train_idx]
        val_subjects   = [all_subjects[i] for i in val_idx]
        result = run_fold(fold_idx, train_subjects, val_subjects)
        fold_results.append(result)

    print("\nAll folds complete.")

    # Print and save cross-fold summary only when more than one fold was run
    if len(fold_results) > 1:
        _print_fold_summary(fold_results)


if __name__ == "__main__":
    main()
