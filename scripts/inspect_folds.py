"""
inspect_folds.py — Diagnostic script that visualises fold construction.

For a given task mode and window configuration, prints:
  - Per-fold: train/val subjects, stimuli per subject, windows per stimulus
  - Optionally: stimulus generalization split (--generalization)
  - Summary statistics across all folds

Run:
    uv run python scripts/inspect_folds.py --task binary
    uv run python scripts/inspect_folds.py --task 9-class --window 10 --stride 10
    uv run python scripts/inspect_folds.py --task binary --generalization
    uv run python scripts/inspect_folds.py --task binary --fold 1      # single fold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thu_ep.dataset import (
    EXCLUDED_SUBJECTS,
    EXCLUDED_STIMULI,
    _build_stimulus_label_map,
)
from src.thu_ep.folds import (
    get_all_subjects,
    get_kfold_splits,
    get_stimulus_generalization_split,
    N_FOLDS,
    _EMOTION_GROUPS,
)

# Emotion names for pretty-printing
EMOTION_NAMES = [
    "Anger", "Disgust", "Fear", "Sadness", "Neutral",
    "Amusement", "Inspiration", "Joy", "Tenderness",
]

SAMPLING_RATE = 200
N_TIMEPOINTS = 6000


def stim_tag(stim_idx: int) -> str:
    """Return a short tag like 'S02(Anger)' for a stimulus index."""
    for stim_range, cls9 in _EMOTION_GROUPS:
        if stim_idx in stim_range:
            return f"S{stim_idx:02d}({EMOTION_NAMES[cls9]})"
    return f"S{stim_idx:02d}(?)"


def main():
    parser = argparse.ArgumentParser(description="Inspect fold construction for THU-EP")
    parser.add_argument("--task", choices=["binary", "9-class"], default="binary")
    parser.add_argument("--window", type=float, default=8.0, help="Window length in seconds")
    parser.add_argument("--stride", type=float, default=4.0, help="Stride in seconds")
    parser.add_argument("--generalization", action="store_true", help="Show stimulus generalization split")
    parser.add_argument("--fold", type=int, default=None, help="Show only this fold (1-indexed)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-subject stimulus details")
    args = parser.parse_args()

    task_mode = args.task
    window_size = round(args.window * SAMPLING_RATE)
    stride = round(args.stride * SAMPLING_RATE)
    n_windows = (N_TIMEPOINTS - window_size) // stride + 1

    label_map = _build_stimulus_label_map(task_mode)
    valid_stimuli = sorted(s for s, lbl in label_map.items() if lbl is not None)
    n_valid_stimuli = len(valid_stimuli)

    # Header
    print("=" * 80)
    print(f"  FOLD INSPECTION  |  task={task_mode}  |  window={args.window}s  stride={args.stride}s")
    print(f"  window_size={window_size} pts  |  stride={stride} pts  |  {n_windows} windows/stimulus")
    print("=" * 80)

    # Stimulus overview
    print(f"\n--- Stimulus -> Label Map ({task_mode}) ---")
    for stim_range, cls9 in _EMOTION_GROUPS:
        items = []
        for s in stim_range:
            lbl = label_map[s]
            if lbl is None:
                items.append(f"  S{s:02d} -> DROPPED")
            else:
                items.append(f"  S{s:02d} -> {lbl}")
        print(f"  {EMOTION_NAMES[cls9]:12s} (9cls={cls9}): " + ", ".join(items))
    print(f"  Valid stimuli: {n_valid_stimuli}  ({valid_stimuli})")

    # Excluded subjects/stimuli
    print(f"\n--- Exclusions ---")
    print(f"  Excluded subjects: {sorted(EXCLUDED_SUBJECTS)}")
    for sid, stims in EXCLUDED_STIMULI.items():
        tags = [stim_tag(s) for s in sorted(stims)]
        print(f"  Subject {sid:02d}: excluded stimuli {tags}")

    # Generalization split
    gen_train_stim: set[int] | None = None
    gen_val_stim: set[int] | None = None
    if args.generalization:
        gen_train_stim, gen_val_stim = get_stimulus_generalization_split(task_mode)
        print(f"\n--- Stimulus Generalization Split ---")
        print(f"  Train stimuli ({len(gen_train_stim)}): {sorted(gen_train_stim)}")
        for s in sorted(gen_train_stim):
            print(f"    {stim_tag(s):20s}  label={label_map[s]}")
        print(f"  Val stimuli ({len(gen_val_stim)}):  {sorted(gen_val_stim)}")
        for s in sorted(gen_val_stim):
            print(f"    {stim_tag(s):20s}  label={label_map[s]}")

        # Check label balance
        train_labels = Counter(label_map[s] for s in gen_train_stim)
        val_labels = Counter(label_map[s] for s in gen_val_stim)
        print(f"\n  Train label distribution: {dict(sorted(train_labels.items()))}")
        print(f"  Val   label distribution: {dict(sorted(val_labels.items()))}")

    # Fold splits
    all_subjects = get_all_subjects()
    folds = get_kfold_splits(all_subjects)
    print(f"\n--- Subject Pool ---")
    print(f"  Total subjects: {len(all_subjects)}")
    print(f"  IDs: {all_subjects}")

    folds_to_show = range(N_FOLDS)
    if args.fold is not None:
        folds_to_show = [args.fold - 1]

    # Accumulators for summary
    total_train_windows_all = []
    total_val_windows_all = []

    for fi in folds_to_show:
        train_idx, val_idx = folds[fi]
        train_subs = [all_subjects[i] for i in train_idx]
        val_subs = [all_subjects[i] for i in val_idx]

        print(f"\n{'-' * 80}")
        print(f"  FOLD {fi + 1}/{N_FOLDS}")
        print(f"{'-' * 80}")
        print(f"  Train subjects ({len(train_subs):2d}): {train_subs}")
        print(f"  Val   subjects ({len(val_subs):2d}):  {val_subs}")

        # Count windows per split
        def count_windows(subjects, stim_filter=None):
            """Count total windows and per-subject/stimulus breakdown."""
            total = 0
            per_subject = {}
            for sid in subjects:
                stim_count = 0
                stim_details = []
                for s in valid_stimuli:
                    # Check exclusions
                    if sid in EXCLUDED_STIMULI and s in EXCLUDED_STIMULI[sid]:
                        continue
                    # Check generalization filter
                    if stim_filter is not None and s not in stim_filter:
                        continue
                    stim_count += 1
                    stim_details.append((s, n_windows))
                wins = stim_count * n_windows
                total += wins
                per_subject[sid] = (wins, stim_count, stim_details)
            return total, per_subject

        if args.generalization:
            train_total, train_per_sub = count_windows(train_subs, gen_train_stim)
            val_total, val_per_sub = count_windows(val_subs, gen_val_stim)
        else:
            train_total, train_per_sub = count_windows(train_subs)
            val_total, val_per_sub = count_windows(val_subs)

        total_train_windows_all.append(train_total)
        total_val_windows_all.append(val_total)

        # Label distribution
        def label_dist(subjects, stim_filter=None):
            counts = Counter()
            for sid in subjects:
                for s in valid_stimuli:
                    if sid in EXCLUDED_STIMULI and s in EXCLUDED_STIMULI[sid]:
                        continue
                    if stim_filter is not None and s not in stim_filter:
                        continue
                    counts[label_map[s]] += n_windows
            return counts

        if args.generalization:
            train_ldist = label_dist(train_subs, gen_train_stim)
            val_ldist = label_dist(val_subs, gen_val_stim)
        else:
            train_ldist = label_dist(train_subs)
            val_ldist = label_dist(val_subs)

        print(f"\n  Train: {train_total:,} windows  |  Val: {val_total:,} windows")
        print(f"  Train label dist: {dict(sorted(train_ldist.items()))}")
        print(f"  Val   label dist: {dict(sorted(val_ldist.items()))}")

        if task_mode == "binary":
            tr_neg = train_ldist.get(0, 0)
            tr_pos = train_ldist.get(1, 0)
            va_neg = val_ldist.get(0, 0)
            va_pos = val_ldist.get(1, 0)
            print(f"  Train balance: neg={tr_neg} ({tr_neg/(tr_neg+tr_pos)*100:.1f}%) / pos={tr_pos} ({tr_pos/(tr_neg+tr_pos)*100:.1f}%)")
            if va_neg + va_pos > 0:
                print(f"  Val   balance: neg={va_neg} ({va_neg/(va_neg+va_pos)*100:.1f}%) / pos={va_pos} ({va_pos/(va_neg+va_pos)*100:.1f}%)")

        # Per-subject detail (verbose)
        if args.verbose:
            print(f"\n  --- Train subjects detail ---")
            for sid in train_subs:
                wins, n_stim, details = train_per_sub[sid]
                excl = EXCLUDED_STIMULI.get(sid, set())
                excl_str = f"  [excluded: {sorted(excl)}]" if excl else ""
                print(f"    Sub {sid:02d}: {n_stim} stimuli x {n_windows} windows = {wins} windows{excl_str}")

            print(f"\n  --- Val subjects detail ---")
            for sid in val_subs:
                wins, n_stim, details = val_per_sub[sid]
                excl = EXCLUDED_STIMULI.get(sid, set())
                excl_str = f"  [excluded: {sorted(excl)}]" if excl else ""
                print(f"    Sub {sid:02d}: {n_stim} stimuli x {n_windows} windows = {wins} windows{excl_str}")

        # Check for subject overlap (sanity)
        overlap = set(train_subs) & set(val_subs)
        if overlap:
            print(f"\n  WARNING: SUBJECT OVERLAP DETECTED: {sorted(overlap)}")
        else:
            print(f"\n  OK: No subject overlap between train and val")

        # Check stimulus overlap in generalization mode
        if args.generalization:
            stim_overlap = gen_train_stim & gen_val_stim
            if stim_overlap:
                print(f"  WARNING: STIMULUS OVERLAP DETECTED: {sorted(stim_overlap)}")
            else:
                print(f"  OK: No stimulus overlap between train and val")

    # Summary across all folds
    if len(folds_to_show) > 1:
        print(f"\n{'=' * 80}")
        print(f"  SUMMARY ACROSS {len(folds_to_show)} FOLDS")
        print(f"{'=' * 80}")

        # Check all subjects appear in val exactly once
        all_val_subjects = []
        for fi in range(N_FOLDS):
            _, val_idx = folds[fi]
            all_val_subjects.extend(all_subjects[i] for i in val_idx)

        val_counts = Counter(all_val_subjects)
        subjects_in_val = sorted(val_counts.keys())
        multi_val = {s: c for s, c in val_counts.items() if c != 1}

        print(f"\n  Each subject appears in validation exactly once: {'YES' if not multi_val else 'NO'}")
        if multi_val:
            print(f"    Subjects appearing >1 time: {multi_val}")
        print(f"  Subjects covered: {len(subjects_in_val)}/{len(all_subjects)}")

        print(f"\n  Windows per fold:")
        print(f"    {'Fold':>5}  {'Train':>8}  {'Val':>8}  {'Total':>8}  {'Val %':>6}")
        print(f"    {'-' * 40}")
        for fi in range(N_FOLDS):
            tr = total_train_windows_all[fi]
            va = total_val_windows_all[fi]
            pct = va / (tr + va) * 100 if (tr + va) > 0 else 0
            print(f"    {fi+1:>5}  {tr:>8,}  {va:>8,}  {tr+va:>8,}  {pct:>5.1f}%")

        avg_tr = sum(total_train_windows_all) / len(total_train_windows_all)
        avg_va = sum(total_val_windows_all) / len(total_val_windows_all)
        print(f"    {'-' * 40}")
        print(f"    {'Avg':>5}  {avg_tr:>8,.0f}  {avg_va:>8,.0f}")

        # Window math
        print(f"\n  Window math:")
        print(f"    Stimulus duration: {N_TIMEPOINTS} pts = {N_TIMEPOINTS/SAMPLING_RATE}s")
        print(f"    Window: {window_size} pts = {args.window}s")
        print(f"    Stride: {stride} pts = {args.stride}s")
        print(f"    Windows per stimulus: ({N_TIMEPOINTS} - {window_size}) / {stride} + 1 = {n_windows}")
        print(f"    Valid stimuli per task: {n_valid_stimuli}")
        normal_per_sub = n_valid_stimuli * n_windows
        print(f"    Normal subject: {n_valid_stimuli} stimuli x {n_windows} windows = {normal_per_sub} windows")

    print()


if __name__ == "__main__":
    main()
