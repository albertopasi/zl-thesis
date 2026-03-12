"""
folds.py — Shared cross-subject KFold split utilities for THU-EP.

Provides:
  - N_FOLDS: canonical number of folds (10).
  - FOLD_RANDOM_STATE: fixed random seed for reproducible splits (42).
  - get_all_subjects(): list of valid subject IDs (1-80, excluding corrupted).
  - get_kfold_splits(): 10-fold cross-subject KFold split on a subject list.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from src.thu_ep.dataset import EXCLUDED_SUBJECTS


N_FOLDS = 10
FOLD_RANDOM_STATE = 42
STIMULUS_SPLIT_SEED = 123

# Emotion groups: (stimulus index range, 9-class label).
# Used by get_stimulus_generalization_split to ensure balanced held-out stimuli.
_EMOTION_GROUPS: list[tuple[range, int]] = [
    (range(0, 3),   0),   # Anger
    (range(3, 6),   1),   # Disgust
    (range(6, 9),   2),   # Fear
    (range(9, 12),  3),   # Sadness
    (range(12, 16), 4),   # Neutral
    (range(16, 19), 5),   # Amusement
    (range(19, 22), 6),   # Inspiration
    (range(22, 25), 7),   # Joy
    (range(25, 28), 8),   # Tenderness
]


def get_all_subjects() -> list[int]:
    """Return list of valid subject IDs (1-80, excluding corrupted subjects)."""
    return [i for i in range(1, 81) if i not in EXCLUDED_SUBJECTS]


def get_stimulus_generalization_split(
    task_mode: str,
    seed: int = STIMULUS_SPLIT_SEED,
) -> tuple[set[int], set[int]]:
    """
    Split stimuli into train (~2/3) and held-out (~1/3) sets, balanced per emotion.

    For each emotion group (3 stimuli): 2 train, 1 held-out.
    For Neutral (4 stimuli): 3 train, 1 held-out.

    In binary mode, Neutral stimuli are excluded from both sets (label = None).

    Args:
        task_mode: 'binary' or '9-class'.
        seed:      Random seed for the per-group shuffle.

    Returns:
        (train_stimulus_indices, test_stimulus_indices) as sets of 0-indexed ints.
    """
    from src.thu_ep.dataset import _build_stimulus_label_map

    label_map = _build_stimulus_label_map(task_mode)
    rng = np.random.RandomState(seed)

    train_stim: set[int] = set()
    test_stim: set[int] = set()

    for stim_range, _ in _EMOTION_GROUPS:
        # Filter to stimuli that have a valid label in this task mode
        indices = [s for s in stim_range if label_map.get(s) is not None]
        if not indices:
            continue

        rng.shuffle(indices)
        n_train = round(len(indices) * 2 / 3)  # 2 for n=3, 3 for n=4
        n_train = max(1, n_train)               # safety: at least 1 train

        train_stim.update(indices[:n_train])
        test_stim.update(indices[n_train:])

    return train_stim, test_stim


def get_kfold_splits(
    subjects: list[int],
    n_folds: int = N_FOLDS,
    random_state: int = FOLD_RANDOM_STATE,
) -> list[tuple]:
    """
    Return list of (train_indices, val_indices) tuples into `subjects`.

    Args:
        subjects:     List of subject IDs to split.
        n_folds:      Number of folds (default 10).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        List of (train_idx_array, val_idx_array) tuples, one per fold.
        Indices index into the `subjects` list, not subject IDs directly.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kfold.split(subjects))
