"""
dataset.py — Shared THU-EP dataset utilities.

Provides constants and classes used across all approaches (linear probing,
LoRA fine-tuning, contrastive learning, etc.):
  - EXCLUDED_SUBJECTS: corrupted subject IDs to skip.
  - EXCLUDED_STIMULI: per-subject corrupted stimulus indices.
  - _build_stimulus_label_map: maps stimulus index → class label.
  - THUEPWindowDataset: raw EEG sliding-window dataset loaded into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# Data-quality constants (corrupted recordings to exclude)

# Subject IDs (1-indexed, matching sub_XX.npy filenames) to skip entirely.
EXCLUDED_SUBJECTS: set[int] = {75}

# Per-subject stimulus indices to drop (0-indexed within the 28-stimulus array).
#   Subject 37: exclude stimuli 16, 22, 25 (1-idx) → 15, 21, 24 (0-idx)
#   Subject 46: exclude stimuli  4, 10, 18, 24, 27 (1-idx) → 3, 9, 17, 23, 26 (0-idx)
EXCLUDED_STIMULI: Dict[int, set[int]] = {
    37: {15, 21, 24},
    46: {3, 9, 17, 23, 26},
}

# Stimulus → label mappings
# Emotion categories (0-indexed stimuli ranges):
#   0-2:  Anger      → neg class 0
#   3-5:  Disgust    → neg class 0
#   6-8:  Fear       → neg class 0
#   9-11: Sadness    → neg class 0
#   12-15: Neutral   → DROPPED in binary mode; class 4 in 9-class
#   16-18: Amusement → pos class 1
#   19-21: Inspiration → pos class 1
#   22-24: Joy       → pos class 1
#   25-27: Tenderness → pos class 1

def _build_stimulus_label_map(task_mode: str) -> Dict[int, Optional[int]]:
    """
    Build a dict mapping each stimulus index (0-27) to its integer label.

    For 'binary' mode, Neutral stimuli (12-15) map to None and must be dropped.
    For '9-class' mode, all stimuli get a class label 0-8.

    Args:
        task_mode: 'binary' or '9-class'.

    Returns:
        Dict of {stimulus_idx: label | None}.
    """
    label_map: Dict[int, Optional[int]] = {}

    # Emotion group boundaries (inclusive start, exclusive end) and class labels
    groups_9class = [
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

    # Binary mapping: neg=0, pos=1, neutral=None (dropped)
    binary_remap = {0: 0, 1: 0, 2: 0, 3: 0, 4: None, 5: 1, 6: 1, 7: 1, 8: 1}

    for stim_range, class_9 in groups_9class:
        for stim_idx in stim_range:
            if task_mode == "9-class":
                label_map[stim_idx] = class_9
            else:  # binary
                label_map[stim_idx] = binary_remap[class_9]

    return label_map


# THUEPWindowDataset
class THUEPWindowDataset(Dataset):
    """
    PyTorch Dataset for THU-EP EEG data, served as sliding windows.

    All subject .npy arrays are loaded into CPU RAM at construction time to
    eliminate disk I/O during training/embedding extraction.

    Sliding-window logic
    --------------------
    Each stimulus has shape (30, 6000): 30 channels x 6000 timepoints (30 s at 200 Hz).

    With window_size=1600 (8 s) and stride=800 (4 s):
        n_windows = floor((6000 - 1600) / 800) + 1 = 6 windows per stimulus

    The flat index `self.index` stores (subject_id, stimulus_idx, window_start)
    tuples so that DataLoader.shuffle=True works out of the box.

    Args:
        subject_ids:  List of 1-indexed subject IDs to include.
        task_mode:    'binary' or '9-class'.
        data_root:    Path to the directory containing sub_XX.npy files.
        window_size:  Number of timepoints per window (default 1600 = 8 s).
        stride:       Stride between consecutive windows (default 800 = 4 s).
    """

    def __init__(
        self,
        subject_ids: List[int],
        task_mode: str,
        data_root: Path,
        window_size: int = 1600,
        stride: int = 800,
    ) -> None:
        super().__init__()

        assert task_mode in ("binary", "9-class"), (
            f"task_mode must be 'binary' or '9-class', got '{task_mode}'"
        )

        self.task_mode = task_mode
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.stride = stride

        # Build stimulus → label lookup for this task mode.
        self._label_map = _build_stimulus_label_map(task_mode)

        # Load all subject arrays into RAM
        # data_cache[subject_id] = np.ndarray of shape (28, 30, 6000)
        self.data_cache: Dict[int, np.ndarray] = {}

        for sid in subject_ids:
            if sid in EXCLUDED_SUBJECTS:
                continue  # skip corrupted subject entirely
            npy_path = self.data_root / f"sub_{sid:02d}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Subject file not found: {npy_path}")
            self.data_cache[sid] = np.load(npy_path)  # shape (28, 30, 6000)

        # Build flat index of all valid (subject, stimulus, window_start)
        # n_windows = floor((n_timepoints - window_size) / stride) + 1
        n_timepoints = 6000
        n_windows = (n_timepoints - window_size) // stride + 1

        self.index: List[Tuple[int, int, int]] = []

        for sid, data in self.data_cache.items():
            n_stimuli = data.shape[0]  # typically 28
            for stim_idx in range(n_stimuli):
                # Skip corrupted stimuli for specific subjects
                if sid in EXCLUDED_STIMULI and stim_idx in EXCLUDED_STIMULI[sid]:
                    continue

                # Skip stimuli whose label is None (neutral in binary mode)
                label = self._label_map.get(stim_idx)
                if label is None:
                    continue

                # Add one entry per window
                for w in range(n_windows):
                    window_start = w * stride
                    self.index.append((sid, stim_idx, window_start))

    # Dataset interface
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """
        Returns:
            eeg_window: float32 Tensor of shape (30, window_size) — channels × timepoints.
            label:      Integer class label.
        """
        subject_id, stim_idx, window_start = self.index[idx]

        # Slice the window from the cached array: (30, window_size)
        # data_cache[sid] shape: (28, 30, 6000)
        window = self.data_cache[subject_id][stim_idx, :, window_start : window_start + self.window_size]

        eeg_tensor = torch.from_numpy(window.astype(np.float32))  # (30, window_size)
        label = self._label_map[stim_idx]

        return eeg_tensor, label
