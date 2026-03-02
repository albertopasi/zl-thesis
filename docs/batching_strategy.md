# Comprehensive Batching Strategy for the JADE Framework

This document outlines the data loading and batching pipeline for the Joint Alignment and Discriminative Embedding (JADE) architecture. Because Supervised Contrastive Learning (SupCon) relies on the relationships between samples within a batch, standard randomized sampling is scientifically and computationally inadequate. The following strategy ensures robust subject-invariant feature extraction while maximizing hardware efficiency.

---

## Part 1: The Logical Constraints

To prevent "shortcut learning" (models learn by comparing overlapping windows) and force the REVE backbone to isolate universal emotional manifolds, every batch must strictly adhere to five logical rules.

### 1. The Batch Dimension

* **Constraint:** Large batch sizes ($N = 256$ or $512$ or bigger if possible).
* **Motivation:** The SupCon loss calculates the similarity between an anchor and *all* other samples in the batch. Small batches result in empty latent spaces and unstable gradients due to a lack of positive pairs.

### 2. Class Balance (The 50/50 Rule)

* **Constraint:** For the binary affective task, exactly 50% of the batch must be Positive and 50% must be Negative (e.g., 64 Pos / 64 Neg).
* **Motivation:** Prevents the latent space from warping toward a majority class. A balanced batch guarantees the loss function always has an equal density of positive anchors to pull together and negative clusters to push away.

### 3. Subject Diversity (Inter-Subject Alignment)

* **Constraint:** The samples for a given emotion must be drawn from a wide distribution of subjects (e.g., 16 (or more?) different subjects per class per batch).
* **Motivation:** If all Positive samples come from just two subjects, the model learns individual anatomical noise. Forcing broad inter-subject representation mathematically mandates the extraction of shared, universal neural signatures.

### 4. Stimulus Diversity (Breaking Strict ISC)

* **Constraint:** Intra-subject positive pairs must come from *different* stimuli.
* **Motivation:** Traditional contrastive frameworks use Inter-Subject Correlation (ISC) by pairing subjects watching the exact same video. By explicitly pairing different videos that elicit the same broad emotion, the model is forced to ignore visual/auditory processing artifacts and isolate the core affective state.

### 5. The Overlap Veto (The Anti-Shortcut Rule)

* **Constraint:** A single batch must **never** contain overlapping or adjacent sliding windows from the exact same stimulus of the exact same subject.
* **Motivation:** If a 5s window and its adjacent overlapping window (e.g., 60% overlap) end up in the same batch, their mathematical similarity is artificially massive. The network will exploit this local temporal overlap to minimize the loss, entirely bypassing the difficult task of learning cross-subject emotional representations.

---

## Part 2: The Engineering Implementation

The pipeline leverages PyTorch's vectorized backend and memory hierarchy to achieve maximum efficiency.

### 1. Eliminating I/O Bottlenecks

* The complete preprocessed dataset (79-80 subjects $\times$ 28 stimuli $\times$ 30 channels $\times$ 6000 time points) is relatively small in 32-bit floats ($\approx$ 1.6 GB). The custom PyTorch `Dataset` loads all 80 subject tensors into CPU System RAM exactly once during `__init__`. The `DataLoader` then dynamically slices the 5-second windows directly from RAM, permanently bypassing disk I/O.

### 2. The Custom `BatchSampler` (Permutation Without Replacement)

* Pure random sampling could cause part of the data to be skipped per epoch while repeating other windows.
* **The Solution:** Implement a custom `BatchSampler`. At the start of an epoch, it generates a global roster of all valid window indices and shuffles it. It iteratively builds batches by popping indices from this roster, checking their metadata (Subject ID, Video ID), and accepting them only if they satisfy the logical constraints (e.g., the Overlap Veto). If an index violates a rule, it is deferred. This guarantees every window is seen exactly once per epoch while strictly maintaining the batch structure.

### 3. Vectorized "Compare-to-All" GPU Math

* SupCon Loss requires comparing every anchor $i$ to every other sample in the batch. Looping through a batch of 128 items (example dimension) requires $128 \times 128 = 16,384$ sequential comparisons.
* Utilize GPU matrix multiplication. Let $Z$ be the batch of projected embeddings with shape `(128, D)`, where $D$ is the embedding dimension (output dimension of the projection head). The entire grid of pairwise cosine similarities is computed instantly via the dot product of $Z$ and its transpose:
  $$S = Z \cdot Z^T$$
  The resulting matrix $S$ has a shape of `(128, 128)`, where row $i$ contains the similarity scores of sample $i$ against all other samples.

### 4. Boolean Masking for Contrastive Pairs

* We need to isolate the "Positive" pairs to pull together and the "Negative" pairs to push apart without iterating through the similarity matrix $S$.
* We construct a binary Mask matrix $M$ of shape `(128, 128)` based on the batch labels. $M_{i,j} = 1$ if sample $i$ and sample $j$ share the same emotion label, and $0$ otherwise. We multiply $S \times M$ to instantly isolate the numerator (positive pairs) for the SupCon loss, and use the inverse mask for the denominator. This completely eliminates Python control flow from the training loop, keeping the GPU at 100% utilization.
