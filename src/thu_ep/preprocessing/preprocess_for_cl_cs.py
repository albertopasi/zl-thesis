"""Preprocessing pipeline for THU-EP data in cl-cs format.

Produces one pickle file per subject, each containing a numpy array of shape
    (28, 30, 7500)  =  (n_stimuli, n_channels, n_samples @ 250 Hz)

Steps (deliberately minimal — cl-cs does its own normalization):
    1. Load raw sub_X.mat  →  (7500, 32, 28, 6)
    2. Extract broad-band (band index 5)  →  (28, 32, 7500)
    3. Remove A1, A2 reference channels   →  (28, 30, 7500)
    4. Save as sub_XX.pkl  (pickle, float32)

Output directory: data/thu ep/cl_cs_preprocessed/
(referenced in baselines/cl-cs/load_data.py as ../../data/thu ep/cl_cs_preprocessed)

Usage:
    uv run python -m src.thu_ep.preprocessing.preprocess_for_cl_cs
    uv run python -m src.thu_ep.preprocessing.preprocess_for_cl_cs --subjects 1 2 3
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Reuse existing preprocessing step functions
from .preprocessing_steps import extract_frequency_band, remove_reference_channels
from ..config import get_config


# Helpers

def _load_mat_file(filepath: Path) -> np.ndarray:
    """Load EEG data from a MATLAB v7.3 HDF5 file.

    Returns shape (7500, 32, 28, 6) = (samples, channels, stimuli, bands).
    """
    with h5py.File(filepath, "r") as f:
        if "data" in f:
            data = np.array(f["data"])
        else:
            keys = [k for k in f.keys() if not k.startswith("#")]
            if len(keys) == 1:
                data = np.array(f[keys[0]])
            else:
                raise ValueError(
                    f"Cannot determine data key in {filepath}. "
                    f"Available keys: {list(f.keys())}"
                )
    return data


def _subject_id_from_filename(path: Path) -> int:
    """Extract integer subject ID from a filename like 'sub_3.mat' or 'sub_12.mat'."""
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse subject ID from filename: {path.name}")
    return int(match.group(1))


def preprocess_subject(mat_path: Path, channels_to_remove_indices: list,
                       band_index: int = 5) -> np.ndarray:
    """Run the minimal cl-cs preprocessing for one subject.

    Returns:
        np.ndarray of shape (28, 30, 7500), dtype float32
    """
    # Step 1: Load
    raw = _load_mat_file(mat_path)                                   # (7500, 32, 28, 6)

    # Step 2: Extract broad-band
    band = extract_frequency_band(raw, band_index=band_index)        # (28, 32, 7500)

    # Step 3: Remove A1, A2
    clean = remove_reference_channels(band, channels_to_remove_indices)  # (28, 30, 7500)

    return clean.astype(np.float32)


def save_subject_pkl(subject_id: int, data: np.ndarray, output_dir: Path) -> Path:
    """Save subject data as a pickle file named sub_XX.pkl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sub_{subject_id:02d}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess THU-EP raw data for the cl-cs baseline."
    )
    parser.add_argument(
        "--subjects", nargs="*", type=int, default=None,
        metavar="N",
        help="Subject IDs to process (1-80). Defaults to all available.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/thu ep/cl_cs_preprocessed relative to project root).",
    )
    args = parser.parse_args()

    config = get_config()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[3]   # zl-thesis/
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "thu ep" / "cl_cs_preprocessed"
    )

    raw_dir = config.raw_data_dir   # data/thu ep/EEG data/
    channels_to_remove = config.channels_to_remove_indices
    band_index = 5  # broad-band (0.5–47 Hz), same as REVE pipeline

    print(f"Raw data dir :  {raw_dir}")
    print(f"Output dir   :  {output_dir}")
    print(f"Band index   :  {band_index}  (broad-band)")
    print(f"Remove chans :  {channels_to_remove}  ({config.channels_to_remove})")

    # Discover subject files
    all_mat_files = sorted(
        raw_dir.glob("sub_*.mat"),
        key=lambda p: _subject_id_from_filename(p),
    )
    if not all_mat_files:
        print(f"[ERROR] No sub_*.mat files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    # Excluded subjects — must match REVE pipeline exclusions (see docs/excluded_data.md)
    excluded_subjects = {75}

    if args.subjects:
        requested = set(args.subjects)
        subject_files = [f for f in all_mat_files if _subject_id_from_filename(f) in requested]
        missing = requested - {_subject_id_from_filename(f) for f in subject_files}
        if missing:
            print(f"[WARN] Subject IDs not found: {sorted(missing)}")
    else:
        subject_files = all_mat_files

    # Always skip entirely-excluded subjects regardless of --subjects flag
    subject_files = [f for f in subject_files if _subject_id_from_filename(f) not in excluded_subjects]
    if excluded_subjects:
        print(f"Skipping subjects : {sorted(excluded_subjects)}  (entire recording excluded)")

    print(f"\nProcessing {len(subject_files)} subject(s)...\n")

    # Process subjects
    ok, failed = 0, []
    for mat_path in subject_files:
        subject_id = _subject_id_from_filename(mat_path)
        try:
            data = preprocess_subject(mat_path, channels_to_remove, band_index)

            expected = (28, 30, 7500)
            if data.shape != expected:
                raise ValueError(f"Unexpected shape {data.shape}, expected {expected}")

            out_path = save_subject_pkl(subject_id, data, output_dir)
            print(f"  [OK] sub {subject_id:02d}  {data.shape}  →  {out_path.name}")
            ok += 1

        except Exception as exc:
            print(f"  [FAIL] sub {subject_id:02d}  {exc}", file=sys.stderr)
            failed.append(subject_id)

    # Summary
    print(f"\nDone. {ok}/{len(subject_files)} subjects saved to {output_dir}")
    if failed:
        print(f"Failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
