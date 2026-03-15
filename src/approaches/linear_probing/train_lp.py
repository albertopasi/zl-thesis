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
    # Binary, all folds (default) (pooled embeddings) (Bx512)
    uv run python -m src.approaches.linear_probing.train_lp --task binary

    # 9-class, all folds
    uv run python -m src.approaches.linear_probing.train_lp --task 9-class

    # Dry run: binary, fold 1 only
    uv run python -m src.approaches.linear_probing.train_lp --task binary --fold 1

    # Specify window size and stride
    uv run python -m src.approaches.linear_probing.train_lp --window 8 --stride 4   # default

    Non pooled embeddings:
    # No-pool, mean over channels (B, Hx512) (H is the number of time patches)
    uv run python -m src.approaches.linear_probing.train_lp --task binary --no-pooling --fold 1

    # No-pool, full flatten (B, CxHx512)
    uv run python -m src.approaches.linear_probing.train_lp --task binary --no-pooling --no-pool-mode flat

    Generalization (held-out stimuli):
    # Single seed (default)
    uv run python -m src.approaches.linear_probing.train_lp --task binary --generalization

    # Multi-seed robustness check (5 seeds)
    uv run python -m src.approaches.linear_probing.train_lp --task binary --generalization --gen-seeds 123 456 789 101 202
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
from torch.utils.data import DataLoader, TensorDataset
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

from src.approaches.linear_probing.model import EmbeddingExtractor, LinearProber
from src.approaches.linear_probing.config import (
    LPConfig,
    DATA_ROOT, EMBEDDINGS_DIR, REVE_MODEL_PATH, REVE_POS_PATH, OUTPUT_DIR,
    USE_WANDB, WANDB_PROJECT, WANDB_ENTITY,
    SAMPLING_RATE, DEVICE, ACCELERATOR, NUM_WORKERS,
)
from src.approaches.linear_probing.summary import (
    print_fold_summary, print_cross_seed_summary,
)


# ── Embedding cache helpers ──────────────────────────────────────────────────

def subject_cache_path(cfg: LPConfig, subject_id: int) -> Path:
    """Return the path for a single subject's pre-computed embedding file."""
    return (
        EMBEDDINGS_DIR / cfg.task_mode
        / f"ws{cfg.window_size}_st{cfg.stride}_{cfg.pool_tag}"
        / f"sub_{subject_id:02d}.pt"
    )


def precompute_all_subjects(
    cfg: LPConfig,
    all_subjects: list[int],
    thu_config: THUEPConfig,
) -> None:
    """
    Pre-compute REVE embeddings for every subject that is not yet cached.

    Embeddings are stored per-subject at:
        EMBEDDINGS_DIR / task_mode / ws{W}_st{S}_{pool_tag} / sub_XX.pt

    This function is called once before any fold loop. Subsequent fold runs
    simply concatenate the already-saved per-subject tensors.
    """
    missing = [
        sid for sid in all_subjects
        if not subject_cache_path(cfg, sid).exists()
    ]

    if not missing:
        print(f"All {len(all_subjects)} subject embedding caches found. Skipping REVE extraction.")
        return

    print(f"\n{len(missing)}/{len(all_subjects)} subjects need embedding extraction.")

    extractor = EmbeddingExtractor(
        reve_model_path=REVE_MODEL_PATH,
        reve_pos_path=REVE_POS_PATH,
        config=thu_config,
        device=DEVICE,
    )

    t0 = time.time()
    try:
        for i, sid in enumerate(missing, 1):
            t_sub = time.time()
            print(f"\n[{i}/{len(missing)}] Subject {sid:02d}", end="  ")

            dataset = THUEPWindowDataset(
                subject_ids=[sid],
                task_mode=cfg.task_mode,
                data_root=DATA_ROOT,
                window_size=cfg.window_size,
                stride=cfg.stride,
            )
            n_win = len(dataset)
            print(f"({n_win} windows)", end="  ", flush=True)

            embeddings, labels, stim_indices = extractor.extract_embeddings(
                dataset, batch_size=cfg.batch_size,
                use_pooling=cfg.use_pooling, no_pool_mode=cfg.no_pool_mode,
            )
            EmbeddingExtractor.save_embeddings(
                embeddings, labels, subject_cache_path(cfg, sid),
                stimulus_indices=stim_indices,
            )

            elapsed_sub = time.time() - t_sub
            avg_per_sub = (time.time() - t0) / i
            eta         = avg_per_sub * (len(missing) - i)
            print(f"done in {fmt_dur(elapsed_sub)}  (ETA {fmt_dur(eta)})")

            del dataset, embeddings, labels
            gc.collect()

    finally:
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll subjects done. Total extraction time: {fmt_dur(time.time() - t0)}")


def load_subjects_embeddings(
    cfg: LPConfig,
    subject_ids: list[int],
    stimulus_filter: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load and concatenate per-subject embedding files for the given subjects.

    Args:
        cfg:              LPConfig (determines cache path).
        subject_ids:      List of subject IDs to load.
        stimulus_filter:  If provided, only include windows whose stimulus index
                          is in this set.

    Returns:
        embeddings: Tensor of shape (N, D)
        labels:     Tensor of shape (N,)
    """
    all_embs:   list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for sid in subject_ids:
        path = subject_cache_path(cfg, sid)
        payload = torch.load(path, map_location="cpu", weights_only=True)

        embs   = payload["embeddings"]
        labels = payload["labels"]

        if stimulus_filter is not None:
            if "stimulus_indices" not in payload:
                raise RuntimeError(
                    f"Cache file {path} does not contain 'stimulus_indices'. "
                    "Delete the embedding cache and re-run to regenerate it."
                )
            stim_idx = payload["stimulus_indices"]
            mask = torch.tensor(
                [int(s.item()) in stimulus_filter for s in stim_idx],
                dtype=torch.bool,
            )
            embs   = embs[mask]
            labels = labels[mask]

        all_embs.append(embs)
        all_labels.append(labels)

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


# ── Per-fold training ─────────────────────────────────────────────────────────

def run_fold(
    cfg: LPConfig,
    fold_idx: int,
    train_subject_ids: list[int],
    val_subject_ids: list[int],
    train_stimuli: set[int] | None = None,
    val_stimuli: set[int] | None = None,
    gen_seed: int | None = None,
) -> dict:
    """
    Train the linear classifier for one fold using pre-computed embeddings.
    """
    seed_label = f"  |  gen_seed={gen_seed}" if gen_seed is not None else ""
    gen_tag = f"  |  GENERALIZATION (held-out stimuli){seed_label}" if cfg.generalization else ""

    print(f"\n{'#' * COL_W}")
    print(
        f"  Fold {fold_idx}/{N_FOLDS}  |  task={cfg.task_mode}  |  "
        f"max_epochs={cfg.max_epochs}  |  device={DEVICE}{gen_tag}"
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

    # ── Load embeddings ───────────────────────────────────────────────────
    t_load = time.time()
    print("Loading embeddings from cache ...", end="  ", flush=True)
    train_embs, train_lbls = load_subjects_embeddings(
        cfg, train_subject_ids, stimulus_filter=train_stimuli,
    )
    val_embs, val_lbls = load_subjects_embeddings(
        cfg, val_subject_ids, stimulus_filter=val_stimuli,
    )
    print(
        f"done in {fmt_dur(time.time() - t_load)}  "
        f"(train={train_embs.shape[0]:,}  val={val_embs.shape[0]:,})"
    )

    train_loader = DataLoader(
        TensorDataset(train_embs, train_lbls),
        batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        TensorDataset(val_embs, val_lbls),
        batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    embed_dim = train_embs.shape[1]
    model = LinearProber(
        num_classes=cfg.num_classes, embed_dim=embed_dim, lr=cfg.lr,
        normalize_features=cfg.normalize_features,
    )

    # ── Logging + callbacks ───────────────────────────────────────────────
    run_name = cfg.run_name(fold_idx, gen_seed)
    ckpt_dir = OUTPUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hparams = cfg.hparams_dict(
        fold_idx=fold_idx,
        n_folds=N_FOLDS,
        n_train_subjects=len(train_subject_ids),
        n_val_subjects=len(val_subject_ids),
        n_train_windows=int(train_embs.shape[0]),
        n_val_windows=int(val_embs.shape[0]),
        embed_dim=embed_dim,
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
        patience=15,
        mode="max",
        verbose=False,
    )

    summary_cb = EpochSummaryCallback(
        output_dir=ckpt_dir,
        fold_idx=fold_idx,
        task_mode=cfg.task_mode,
        train_subjects=train_subject_ids,
        val_subjects=val_subject_ids,
        hparams=hparams,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=ACCELERATOR,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, summary_cb],
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    finally:
        del train_embs, train_lbls, val_embs, val_lbls, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if USE_WANDB:
            wandb.finish()

    # Save best classifier weights
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        best_model = LinearProber.load_from_checkpoint(best_ckpt)
        weights_path = ckpt_dir / "classifier_weights.pt"
        torch.save(best_model.classifier.state_dict(), weights_path)
        print(f"Classifier weights (best epoch) saved -> {weights_path}")
        del best_model
    else:
        print("Warning: no checkpoint found, weights not saved.")

    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best val/acc:    {checkpoint_cb.best_model_score:.4f}")

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

def parse_args() -> LPConfig:
    """Parse CLI arguments and return a populated LPConfig."""
    parser = argparse.ArgumentParser(
        description="Linear Probing on THU-EP with REVE embeddings"
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
        "--window", type=float, default=8.0, metavar="S",
        help="Window length in seconds (default: %(default)s s).",
    )
    parser.add_argument(
        "--stride", type=float, default=4.0, metavar="S",
        help="Stride between windows in seconds (default: %(default)s s).",
    )
    parser.add_argument(
        "--normalize", action="store_true", default=False,
        help="Apply L2 feature normalization before the linear classifier.",
    )
    parser.add_argument(
        "--no-pooling", dest="no_pooling", action="store_true", default=False,
        help="Bypass attention pooling; use raw patch embeddings flattened by --no-pool-mode.",
    )
    parser.add_argument(
        "--no-pool-mode", dest="no_pool_mode", choices=["mean", "flat"], default="mean",
        help="How to flatten raw patches when --no-pooling is set: "
             "'mean' (avg over channels -> Hx512, default) or 'flat' (CxHx512).",
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

    return LPConfig(
        task_mode=args.task,
        fold=args.fold,
        window_size=round(args.window * SAMPLING_RATE),
        stride=round(args.stride * SAMPLING_RATE),
        normalize_features=args.normalize,
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

    # Step 1: Pre-compute ONCE per subject (skips subjects already cached)
    precompute_all_subjects(cfg, all_subjects, thu_config)

    # Step 2: 10-fold cross-subject split
    folds = get_kfold_splits(all_subjects)
    folds_to_run = (
        [(cfg.fold, folds[cfg.fold - 1])]
        if cfg.fold is not None
        else [(i + 1, folds[i]) for i in range(N_FOLDS)]
    )

    # Step 3: Seed list
    seed_list: list[int | None] = [None]
    if cfg.generalization:
        seed_list = cfg.gen_seeds
        print(f"\nGeneralization mode: {len(seed_list)} seed(s) = {seed_list}")

    # Step 4: Outer loop over seeds, inner loop over folds
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
                cfg, fold_idx, train_subjects, val_subjects,
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
