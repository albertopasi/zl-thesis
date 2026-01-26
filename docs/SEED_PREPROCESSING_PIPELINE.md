# SEED Dataset Preprocessing Pipeline for REVE Foundation Model

This README outlines a technically rigorous preprocessing pipeline for the SEED dataset, specifically designed to prepare high-fidelity inputs for the **REVE foundation model encoder**. The architecture utilizes a **Manual Buffer** strategy to navigate the conflict between massive, header-corrupted raw files and the need for continuous signal conditioning and session-level statistics.

---

## 1. Dataset Context & Challenges

### SEED Dataset Nuances
The SEED dataset provides raw continuous EEG recordings in the NeuroScan `.cnt` format. Each session includes approximately 60 minutes of data, covering 15 trials of movie-induced emotional stimuli.
* **Auxiliary Noise**: In addition to 62 neural EEG channels, the raw data contains 4 non-neural channels: **M1/M2** (mastoid references) and **VEO/HEO** (eye-movement tracking).
* **The Header Bug**: Many raw SEED `.cnt` files contain corrupted headers that report impossible recording durations (often exceeding 150 hours). This makes standard "load and process" methods fail due to memory allocation errors.

### The Foundation Model Requirement (REVE)
The REVE model is a spatio-temporal transformer explicitly designed to generalize across diverse electrode arrangements. To function correctly, it requires:
1.  **3D Coordinates**: Signals must be paired with actual $(x, y, z)$ electrode positions for its 4D Positional Encoding module.
2.  **Specific Scaling**: Input data must be Z-score normalized using **session-level statistics** and clipped at a threshold of **15 standard deviations** to match its pretraining distribution.

---

## 2. Methodology: The Manual Buffer Approach

Traditional window-by-window processing is often preferred for speed, but it is technically flawed for this pipeline. Slicing raw data before filtering introduces **edge artifacts** (filter ringing) that can corrupt several seconds of each trial. Conversely, loading the entire session to filter it crashes the system due to the corrupted headers.

The **Manual Buffer** approach solves this by:
1.  **Segment Selection**: Using verified sample indices from `time.txt` to extract only the physical recording data corresponding to the movie clips (~60 minutes total). This effectively ignores the rest periods, self-assessments, and "ghost" data reported in the corrupted header.
2.  **Continuous Signal Conditioning**: Concatenating these trials into one contiguous array to perform filtering and resampling as if the session were a single stream, ensuring mathematical stability and minimizing filter "warm-up" distortions.

---

## 3. Thorough Step-by-Step Pipeline Explanation

### Step 1: Header-Independent Lazy Loading
The raw `.cnt` file is initialized without preloading the signal data into memory to bypass the corrupted header. Using the `start_point_list` and `end_point_list` from `time.txt`, I extract the 15 movie segments directly from the binary stream. While the original file contains "dead time" (pauses between movies), I only extract the active stimulus periods to optimize memory and focus on relevant neural activity.

### Step 2: Channel Management
I immediately drop the 4 auxiliary channels (`M1, M2, VEO, HEO`) to reduce the data width to 62 channels. I map these to the **SEED-specific 3D montage** using the `channel_62_pos.locs` file to ensure spatial alignment for REVE's 4D positional encoder.

### Step 3: Buffer Construction
The 15 extracted trial segments are glued together into a single contiguous NumPy array. By skipping the pauses and self-assessment periods, I create a compact "session" that contains 100% relevant stimulus data while remaining manageable in RAM.

### Step 4: Signal Conditioning (Filtering & Resampling)
* **Filter (0.5–99.5 Hz)**: I apply a band-pass filter to the **entire buffer**. This removes the DC offset (0.5 Hz) and high-frequency noise. Because the buffer is processed continuously, filter oscillations dissipate long before reaching the windows we keep.
* **Resample (200 Hz)**: The data is downsampled from 1000 Hz to 200 Hz. Resampling the buffer continuously prevents aliasing artifacts that occur when decimating short, isolated windows. REVE also requires data to be resampled to 200 Hz for downstream tasks.

### Step 5: Compute Session Statistics (todo: check)
REVE requires normalization statistics derived from the **entire recording session**, not individual trials. I calculate one global mean ($\mu$) and one standard deviation ($\sigma$) across all 62 channels in the processed buffer. This ensures that the relative emotional intensity between different trials is preserved in the final feature space.

### Step 6: Buffer Re-slicing
The continuous buffer is split back into the 15 original trials. Because we downsampled the data (1000 Hz to 200 Hz), the original sample indices are scaled by a factor of 5 to ensure the slices are accurate.

### Step 7: Final Windowing & REVE Normalization
For each trial, I keep only the **last 30 seconds** (6,000 samples at 200 Hz).
* **Motivation**: Benchmark studies (CLISA and CL-CS) show that EEG signals take time to stabilize after emotional induction. The tail end of the movie represents the most "coherent" emotional state.
* **Normalization**: Each 30s trial is Z-normalized using the **session stats** from Step 5 ($X_{norm} = (X - \mu_{session}) / \sigma_{session}$).
* **Clipping**: Any values exceeding **15 standard deviations** are clipped to ensure model stability while retaining high-amplitude neural events.

### Step 8: Structured Export
The final trials are saved as float32 NumPy files (`.npy`) in a subject/session folder structure. Alongside these files, the **3D Cartesian coordinate array** (`coords.npy`) of shape `(62, 3)` is exported. This allows the REVE encoder to reconstruct the spatial hierarchy of the SEED electrodes during inference.

---

## 4. Technical Summary for REVE Inference

| Feature | SEED Original | Pipeline Output (REVE Target) |
| :--- | :--- | :--- |
| **Sample Rate** | 1000 Hz | 200 Hz |
| **Bandpass** | 0.1 - 100 Hz (raw) | 0.5 - 99.5 Hz |
| **Duration** | ~4 minutes / trial | Last 30 seconds / trial |
| **Normalization** | Raw Voltage ($\mu V$) | Session-wise Z-score (clipped ±15σ) |
| **Channels** | 66 (raw) | 62 (Neural only) |
| **Spatial Data** | Polar (`.locs`) | 3D Cartesian $(x, y, z)$ |