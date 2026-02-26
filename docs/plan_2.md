# Implementation Blueprint: EEG Emotion Recognition via REVE and JADE

**Project:** A Supervised Contrastive Learning framework for Cross-Subject EEG-based Emotion Recognition via EEG Foundation Models

**Dataset:** THU-EP (Optional change: 79 Subjects, discarding Subject 75 due to massive channel corruption).

**Core Framework:** PyTorch & PyTorch Lightning

---

**Legend:** 
* 🟢 Completed step
* 🟠 Requires correctness chech / Incomplete

## 🟢Phase 1: Static Subject-Level Preprocessing (`preprocess.py`)

**Objective:** Transform raw `.mat` files into clean, REVE-ready `.npy` tensors while minimizing disk I/O and preserving neural phase dynamics.

**DSP Pipeline Steps:**
1. 🟢**Frequency Band Selection:** Isolate the 6th dimension (Broad-band: 0.5 - 47 Hz) and discard the 5 pre-computed frequency bands. REVE natively processes broad spatio-temporal dynamics.
2. 🟢**Channel Pruning:** Extract the 30 scalp electrodes, strictly removing the `A1` and `A2` (linked mastoids) reference channels.
4. 🟢**Individual Trial Downsampling:** * Using `mne.filter.resample(...)`, downsample the data from 250 Hz to 200 Hz. 
    * This must be executed on each 30-second stimulus array *individually*. Using the `polyphase` method on individual trials prevents the severe boundary artifacts that FFT-based concatenation would induce.
3. 🟢**Global Z-Score Normalization:** * Calculate the mean ($\mu$) and standard deviation ($\sigma$) per channel across the *entire* recording session (all 28 stimuli) for a given subject to ensure robust statistical estimation.
5. 🟢**Local Normalization & Clipping:** Apply the globally computed $\mu$ and $\sigma$ to the trials. Clip any signal values exceeding 15 standard deviations to bound extreme technical outliers.
6. 🟢**Storage:** Save each of the 80 (or 79) subjects as a single `.npy` file of shape `(28 stimuli X 30 channels X 6000 time points)`.

---

## Phase 2: Dynamic Data Loading (`dataset.py`)

**Objective:** Efficiently extract sliding windows on-the-fly to prevent storage bloat and allow instant hyperparameter tuning.

* **RAM Caching:** The custom PyTorch `Dataset` loads all 80 (or 79) preprocessed tensors ($\approx$ 1.6 GB) directly into System RAM during `__init__`. The hard drive is never queried during the training loop.
* **On-the-Fly Slicing:** The `__getitem__(self, index)` method dynamically slices a window (e.g., 5s = 1000 time points) based on the provided index.
* **Returns:** `(x, y, subject_id, stimulus_id)`. Returning the metadata is mandatory for the custom Batch Sampler to enforce contrastive rules.

[Note]: In no case can the preprocessed files in data/thu ep/preprocessed be modified or deleted.
---

## Phase 3: Task Modularization (Binary vs. 9-Class)

**Objective:** Architect the code to seamlessly toggle between the foundational binary task and the rigorous 9-class task without rewriting core logic.

* Stimuli order and labels: 28 stimuli in total: (Stimulus number, stimulus label)
    - 0-1-2 : Anger
    - 3-4-5 : Disgust
    - 6-7-8 : Fear
    - 9-10-11 : Sadness
    - 12-13-14-15 : Neutral
    - 16-17-18 : Amusement
    - 19-20-21 : Inspiration
    - 22-23-24 : Joy
    - 25-26-27 : Tenderness
* **Binary Task (Positive vs. Negative):**
    * **Mapping:**
        - Anger, Disgust, Fear, Sadness $\rightarrow$ `0` (Negative)
        - Amusement, Inspiration, Joy, Tenderness $\rightarrow$ `1` (Positive).
    * **Exclusion:** Neutral stimuli are strictly dropped to maintain perfect class balance.
* **9-Class Task:**
    * **Mapping:** All 9 different emotion labels are mapped to integers `0` through `8`.
* **Implementation:** Pass a `task_mode='binary'` or `'9-class'` string argument into the `Dataset` and `LightningDataModule`. The dataset automatically applies the correct label mapping dictionary and drops Neutral trials if `task_mode=='binary'`.

---

## Phase 4: The Batching Architectures (The Core Engineering)

Because Contrastive Learning and Cross-Entropy have fundamentally different mathematical assumptions, they require completely separate `DataLoader` strategies.

### A. Linear Probing / Cross-Entropy Batching
* **Strategy:** Standard Randomized Batching.
* **Logic:** The Cross-Entropy loss evaluates each window independently. Therefore, overlapping windows or intra-subject clusters do not cause "shortcut" learning.
* **Implementation:** Use PyTorch's default `DataLoader(..., shuffle=True)`. Batch size can be 32, 64, or 128.
* **Compute Hack:** Because REVE is frozen during Baseline 1, pre-compute the 512-D embeddings for all training windows once, save them to RAM, and train the linear classifier directly on the vectors to finish 100 epochs in seconds.

### B. JADE / Contrastive Learning Batching (The Grid Sampler)
* **Strategy:** Custom Permutation without Replacement (`ContrastiveBatchSampler`). 
* **Logic:** The Supervised Contrastive loss calculates similarity across the entire batch matrix ($S = Z \cdot Z^T$). To prevent spatial-temporal shortcuts and force generalized inter-subject alignment, strict rules apply.
* **The Constraints:**
    1. **Batch Size:** Large ($N = 256$ or $512$ or higher if possible).
    2. **Class Balance:** Binary requires exactly 50/50 splits. 9-class requires an $M$-per-class sampler (e.g., exactly 16 samples for all 9 classes in a batch of 144).
    3. **Subject & Stimulus Diversity:** Positive pairs must span across different subjects and different videos (e.g., Subject A Joy 1 paired with Subject B Tenderness 2).
    4. **The Overlap Veto:** A batch must *never* contain overlapping or adjacent windows from the exact same stimulus of the exact same subject.
* **Implementation:** The `ContrastiveBatchSampler` initializes a globally shuffled list of all valid window indices. It iteratively builds a batch by popping indices and validating them against a "currently in batch" dictionary. If an index violates the Overlap Veto or the Class Balance constraint, it is deferred to the next batch. 

---

## Phase 5: The Empirical Window/Stride Test

**Objective:** Scientifically prove the optimal temporal resolution for the JADE architecture without wasting compute or starving the SupCon loss.

* **Configurations to Test:** * 5s window / 2s stride (State-of-the-Art Baseline Parity)
    * 8s window / 4s stride (Midpoint)
    * 10s window / 5s stride (Optimal REVE Context + 50% Overlap)
* **The Compute-Saving Protocol:** * Run this exclusively on the **Binary Task** using the full **JADE framework**.
    * Keep the strict 10-fold cross-subject split (train on 71 subjects, test on 8) so the contrastive loss sees full subject diversity.
    * Only execute **3 of the 10 folds**. Average the validation metrics across these 3 folds to determine the winning configuration, then lock those parameters in for all final evaluations.

---

## Phase 6: 10-Fold Cross-Subject Evaluation Protocol

**Objective:** Guarantee strict, data-leakage-free proof of "plug-and-play" generalization. 

1. **The Seed:** Initialize `seed=42` globally (`torch.manual_seed(42)`). 
2. **The Split:** Use `sklearn.model_selection.KFold(n_splits=10, shuffle=True)`. This cleanly handles the 79 subjects (9 folds of 8 test subjects, 1 fold of 7 test subjects).
3. **The Hard Reset:** Between every fold, the `LightningModule` (containing the REVE backbone, LoRA weights, and classifier) must be completely destroyed and re-initialized from scratch. 
4. **Final Metric & Local Logging:** * At the end of each fold's `trainer.test()`, extract the core metrics (Accuracy, ROC-AUC, Precision-Recall AUC, F1-Score).
    * Append these fold-specific dictionaries into a master JSON file located in a dedicated `results/{experiment_name}/` directory. This ensures that even if cloud logging fails, the raw numeric data for the thesis tables is safely preserved on disk.

---

## Phase 7: PyTorch Lightning Integration

**Objective:** Modularize the training loops to avoid boilerplate code and cleanly separate the data pipeline from the mathematical models. 

1. **`THUEP_DataModule(pl.LightningDataModule)`:**
    * Handles the `KFold` splitting logic.
    * In `setup()`, it creates the `train_dataset` and `val_dataset` based on the current fold's subject lists.
    * In `train_dataloader()`, it returns either the standard `DataLoader` (for LP) or the `ContrastiveBatchSampler` (for JADE) based on a configuration flag.
2. **`JADE_Model(pl.LightningModule)`:**
    * Initializes the REVE backbone, injects LoRA adapters, and builds the non-linear projection head and linear classifier.
    * In `training_step()`, it executes the joint loss: $L_{total} = \lambda L_{SupCon} + (1-\lambda) L_{CE}$.
    * Handles the highly optimized vectorized matrix math for the SupCon loss to keep GPU utilization at 100%: 
        * First, computes the similarity matrix $S = Z \cdot Z^T$.
        * Second, dynamically generates a binary mask matrix $M$ of shape `(batch_size, batch_size)` where $M_{i,j} = 1$ if sample $i$ and sample $j$ share the same emotion label, and $0$ otherwise. 
        * Multiplies $S \times M$ to instantly isolate the numerator (positive pairs) without utilizing slow Python `for` loops, while using the inverted mask $(1 - M)$ to identify the negative pairs for the denominator.

---

## Phase 8: Experiment Tracking (Weights & Biases Integration)

**Objective:** Systematically track the 10-fold cross-subject cross-validation metrics, hardware utilization, and hyperparameter sweeps without manual data entry. 

* **Logger Setup:** Utilize `pytorch_lightning.loggers.WandbLogger` and pass it directly into the PyTorch Lightning `Trainer`.
* **Fold Grouping:** * Use the `group` parameter in W&B to cluster the 10 separate runs (folds) under a single overarching experiment name (e.g., `group="JADE_Binary_5s_Window"`). 
    * Name each specific run using the fold number (e.g., `name="Fold_1"`). This ensures the dashboard stays organized.
* **Granular Logging:** * Inside `training_step()`, independently log $L_{total}$, $L_{SupCon}$, and $L_{CE}$ so you can verify that the contrastive loss is converging appropriately relative to the classification loss.
    * Inside `validation_step()`, track the Validation Accuracy and AUC. 
* **The Hard Reset Caution:** Because the Lightning model and Trainer are destroyed and recreated for each of the 10 folds,  call `wandb.finish()` at the end of each fold's loop to close the connection before the next fold initializes a new W&B run (is this the case?).

---

## Phase 9: Local Artifact & Checkpoint Management

**Objective:** Persist model weights efficiently and generate the qualitative visual evidence required for the thesis defense (t-SNE and Confusion Matrices).

### 9.1. Visualizing the Latent Space (t-SNE)
* **The Goal:** Prove mathematically and visually that the SupCon loss successfully grouped subjects by emotion rather than by identity. 
* **Implementation:** * During the test phase of the final fold, extract the 512-dimensional projected embeddings ($Z$) and their corresponding ground-truth labels for the entire test set.
    * **2D** Pass these embeddings through `sklearn.manifold.TSNE(n_components=2)` to project them down to 2D. Save a high-resolution 2D scatter plot (`tsne_2d_fold_10.png`) colored by emotion class.. 
    * **3D** Pass the embeddings through `TSNE(n_components=3)` and generate an interactive 3D plot. This allows you to rotate the emotional manifolds live during your presentation to prove structural separation. Also create a static image of the 3d plot.

### 9.2. Error Analysis (Confusion Matrices)
* **The Goal:** Identify which specific emotions the model struggles to differentiate. 
* **Implementation:** * Aggregate the predicted classes vs. ground truth classes during the `test_step`.
    * Use `sklearn.metrics.confusion_matrix` and `seaborn.heatmap` to generate a normalized grid.
    * Save this matrix as a `.png` in the `results/` folder for each fold, and compute an average confusion matrix across all 10 folds for the final thesis manuscript.

### 9.3. PyTorch Lightning Checkpointing
* **The Callback:** Pass a `pytorch_lightning.callbacks.ModelCheckpoint` into the `Trainer`.
* **Configuration:**
    * `dirpath="checkpoints/{experiment_name}/fold_{k}/"`
    * `monitor="val_loss"` (or `val_acc`)
    * `save_top_k=1` (Saves the best performing epoch to prevent overfitting).
    * `save_last=True` (Allows you to resume training if the run crashes).
* **Storage Optimization (Crucial):** Because the REVE foundation model is massive, saving the entire network 10 times will instantly fill your hard drive. Configure your `LightningModule`'s `on_save_checkpoint` hook to **only save the LoRA adapter weights, the projection head, and the linear classifier**. The frozen REVE base weights are already stored elsewhere and do not need to be duplicated.
