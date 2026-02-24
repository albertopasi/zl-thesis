# Implementation Plan: EEG Emotion Recognition via REVE and JADE

## Phase 1: Static Subject-Level Preprocessing
Instead of creating massive datasets of overlapping windows, we will create clean, foundation-model-ready files for each of the 80 subjects.
Channel Selection: Remove the A1 and A2 channels (left and right mastoids). Because the data is referenced to the average of these linked mastoids, they no longer contain independent neural variance. You will use the remaining 30 channels.

- Band Selection: Extract only the 6th dimension (broad-band signal: 0.5 to 47 Hz) and discard the pre-computed delta, theta, alpha, beta, and gamma dimensions. REVE natively learns from continuous spatio-temporal dynamics.

- Individual Downsampling: Downsample the data from 250 Hz to exactly 200 Hz. Crucially, this must be done on each 30-second stimulus individually to prevent severe boundary artifacts from ruining the signal ends.

- Global Z-Score Normalization: Compute the mean ($\mu$) and standard deviation ($\sigma$) per channel across the entire recording session for a given subject. Apply these global statistics to Z-score normalize each individual 30-second stimulus array.

- Outlier Clipping: Clip any normalized signal values exceeding 15 standard deviations to bound extreme numerical anomalies.

- Spatial Mapping: Map the 30 standard 10-20 system channel labels to their corresponding physical 3D coordinates using REVE's official pre-registered positions.

- Storage: Save each subject as a single .pt or .npy file (shape: 28 stimuli x 30 channels x 6000 time points).

## Phase 2: Dynamic On-The-Fly Window Extraction

Do not save the sliding windows as static files. Implement dynamic extraction inside the PyTorch Dataset class.

This prevents massive storage bloat caused by overlapping data and allows you to instantly swap window sizes and strides during your empirical testing phase with a single line of code.

## Phase 3: The Empirical Window/Stride Test
Before running the final evaluations, we will empirically determine the optimal temporal resolution for the  specific JADE architecture.
The Configurations to Test: 
- 5s window / 2s stride: Matches state-of-the-art contrastive baselines (CLISA, CL-CS).
- 10s window / 5s stride: Provides the longer context REVE prefers while maintaining a safe 50% overlap for contrastive volume.
- 8s window / 4s stride: A midpoint exploration.

The Target Architecture: Run this test on the full JADE framework (not Linear Probing). Contrastive learning is uniquely vulnerable to "shortcut learning" via overlapping strides; we must test the stride on the loss function that actually uses it.
Use the strict 10-fold cross-validation setup (training on 72 subjects, testing on 8) so the contrastive loss is not starved of subject diversity. However, to save compute, only execute 3 of the 10 folds and average the results.


## Phase 4: The Classification Target Strategy

- Step 1: Binary Task: Begin all testing and hyperparameter tuning on the binary classification task (positive vs. negative emotions). Explicitly exclude the neutral baseline state due to unbalanced trial counts.
- Step 2: The 9-Class Task: Only after locking in the optimal window/stride from the binary test will we switch to the rigorous nine-class emotional classification task.

