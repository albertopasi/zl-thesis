"""
folds.py — Shared cross-subject KFold split utilities for THU-EP.

Provides:
  - N_FOLDS: canonical number of folds (10).
  - FOLD_RANDOM_STATE: fixed random seed for reproducible splits (42).
  - get_all_subjects(): list of valid subject IDs (1-80, excluding corrupted).
  - get_kfold_splits(): 10-fold cross-subject KFold split on a subject list.
"""

from __future__ import annotations

from sklearn.model_selection import KFold

from src.thu_ep.dataset import EXCLUDED_SUBJECTS


N_FOLDS = 10
FOLD_RANDOM_STATE = 42


def get_all_subjects() -> list[int]:
    """Return list of valid subject IDs (1-80, excluding corrupted subjects)."""
    return [i for i in range(1, 81) if i not in EXCLUDED_SUBJECTS]


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
