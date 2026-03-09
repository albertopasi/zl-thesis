"""
callbacks.py — Shared Lightning callbacks for THU-EP training pipelines.

Provides:
  - EpochSummaryCallback: prints a per-epoch metrics table and saves results JSON.
  - fmt_dur: format seconds into a human-readable duration string.
"""

from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path

import torch
import lightning as L


# ── Formatting helpers ──────────────────────────────────────────────────────

COL_W = 90
SEP   = "─" * COL_W
HEADER = (
    f"{'Epoch':>6}  {'EpTime':>7}  {'Elapsed':>8}  "
    f"{'TrLoss':>8}  {'TrAcc':>7}  "
    f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaAUROC':>8}  {'VaF1':>7}"
)


def fmt_dur(seconds: float) -> str:
    """Format a duration in seconds into a compact human-readable string."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _v(t) -> float:
    """Extract a plain float from a metric value (Tensor, None, or float)."""
    if isinstance(t, torch.Tensor):
        return float(t.item())
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return float("nan")
    return float(t)


def fmt_metric(val: float, width: int = 8, decimals: int = 4) -> str:
    """Format a metric value, showing 'n/a' for NaN."""
    if math.isnan(val):
        return f"{'n/a':>{width}}"
    return f"{val:>{width}.{decimals}f}"


# ── EpochSummaryCallback ───────────────────────────────────────────────────

class EpochSummaryCallback(L.Callback):
    """
    Prints a per-epoch metrics table to the terminal and saves a results JSON
    at the end of training.

    Works with any Lightning module that logs the standard metric keys:
        train/loss, train/acc, val/loss, val/acc, val/auroc, val/f1

    An optional *extra_columns* callable can be provided to append custom
    columns (e.g. training phase) to each row of the terminal table.
    """

    def __init__(
        self,
        output_dir: Path,
        fold_idx: int,
        task_mode: str,
        train_subjects: list[int],
        val_subjects: list[int],
        hparams: dict,
        extra_columns: "callable | None" = None,
    ) -> None:
        self.output_dir     = Path(output_dir)
        self.fold_idx       = fold_idx
        self.task_mode      = task_mode
        self.train_subjects = train_subjects
        self.val_subjects   = val_subjects
        self.hparams        = hparams
        self.extra_columns  = extra_columns
        self.epoch_history: list[dict] = []
        self._fit_start:   float | None = None
        self._epoch_start: float | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._fit_start = time.time()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.time()
        if trainer.current_epoch == 0:
            print(f"\n{SEP}")
            print(HEADER)
            print(SEP)

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
        eta_str   = f"ETA {fmt_dur(remaining)}" if epoch < trainer.max_epochs else "done"

        extra = ""
        if self.extra_columns is not None:
            extra = "  " + self.extra_columns(trainer, pl_module)

        print(
            f"{epoch:>6}  {fmt_dur(ep_time):>7}  {fmt_dur(elapsed):>8}  "
            f"{fmt_metric(tr_loss):>8}  {fmt_metric(tr_acc):>7}  "
            f"{fmt_metric(va_loss):>8}  {fmt_metric(va_acc):>7}  "
            f"{fmt_metric(va_auroc):>8}  {fmt_metric(va_f1):>7}  ({eta_str}){extra}"
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

        print(SEP)
        print(
            f"Training complete — {len(self.epoch_history)} epochs  |  "
            f"total time: {fmt_dur(total_time)}"
        )
        if valid_rows:
            best = max(valid_rows, key=lambda r: r["val_acc"])
            print(
                f"Best  epoch={best['epoch']:>3}  val_acc={best['val_acc']:.4f}  "
                f"val_auroc={best['val_auroc']:.4f}  val_f1={best['val_f1']:.4f}"
            )
        print(SEP)
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
