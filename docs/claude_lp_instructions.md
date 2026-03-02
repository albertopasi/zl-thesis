# Context and Instructions for Claude: EEG Linear Probing Pipeline

**Role:** You are an expert PyTorch and PyTorch Lightning engineer helping build a Master's thesis on Cross-Subject EEG Emotion Recognition.

Rule: Ignore the floders that contain zl or seed in the name.

**Current State:** The EEG data has been preprocessed. We have 80 subjects. The data is saved as `.npy` files (data/thu ep/preprocessed/sub_X.npy), of shape `(28 stimuli, 30 channels, 6000 timepoints)`. The sampling rate is 200 Hz. 
*Note on corrupted Data:* Due to noise, some stimuli need to be excluded for the training and evaluation parts, the data loader must not include the following data: Subject 75 (exlude entire subject), Subject 37 exclude stimuli 16, 22, 25 (-1 if starting is 0), Subject 46 exclude stimuli 4, 10, 18, 24, 27 (-1 if starting is 0). The dataset and dataloader must handle dynamic stimuli counts gracefully.

**Immediate Goal:** Implement the "Baseline 1: Linear Probing" (LP) pipeline. We need to build the PyTorch `Dataset`, the Lightning `DataModule`, the Lightning `Module`, and the execution script. 

**Architectural Requirements:**

1. **Windowing Math:**
   * Window size: 8 seconds (1600 timepoints at 200 Hz).
   * Stride: 4 seconds (800 timepoints).
   * The `Dataset` must dynamically slice these overlapping windows from the 30-second (6000 timepoints) stimuli arrays.

2. **Task Modularization (Binary vs 9-Class):**
   * The `Dataset` and `DataModule` must accept a `task_mode` argument (either `'binary'` or `'9-class'`).
   * Stimuli mapping (0-2: Anger, 3-5: Disgust, 6-8: Fear, 9-11: Sadness, 12-15: Neutral, 16-18: Amusement, 19-21: Inspiration, 22-24: Joy, 25-27: Tenderness).
   * **Binary Mode (Current Focus):** * Positive (`1`): Amusement, Inspiration, Joy, Tenderness. 
     * Negative (`0`): Anger, Disgust, Fear, Sadness. 
     * *Crucial:* Neutral stimuli (12-15) must be completely dropped/ignored in binary mode.
   * **9-Class Mode:** Map all 9 emotions to classes 0-8.

3. **Data Loading Logic (Linear Probing):**
   * Load all required `.npy` arrays into CPU RAM in the `__init__` to prevent disk I/O bottlenecks.
   * Create a flat index or mapping of all valid sliding windows to allow PyTorch's default `DataLoader` to simply use `shuffle=True`.
   * Keep batch size manageable for 12GB GPU (e.g., 64?).

4. **Model Architecture (Frozen REVE + Classifier):**
   * We are using a frozen foundation model (REVE) as a feature extractor.
   * *Compute Hack for 12GB GPUs:* To save massive amounts of compute during the LP training loop, please write a pre-computation utility. We want to pass all training/validation windows through the frozen REVE encoder *once*, save the resulting 512-D embeddings to a tensor, and train the Lightning LP classifier directly on those 512-D vectors. 
   * The linear classifier is just a simple fully connected layer mapping `512 -> num_classes`.
   * Note: an example on how to use reve can be found in file Copy of reve tutorial eegmat.ipynb. also, from there, understand how to use the position bank to pass the electrodes positions to reve (the position to pass are specified in my config file configs/thu_ep.yml). **Crucial REVE Embedding Extraction Logic:** * To get the final 512-D embeddings, it is a two-step process (not sure about this, check the models/reve_pretrained_original/reve_base/modeling_reve.py to understand how to extract the globally pooled embedding, dont take the 3 following points as correct as Im not sure). 
     * First, pass the EEG window and the 3D electrode coordinates (loaded from the position bank specified in `configs/thu_ep.yml`) into the model's forward pass: `out_4d = reve_model(eeg, pos)`.
     * Second, pass that output directly into REVE's built-in pooling method to collapse the spatial-temporal dimensions: `final_512_emb = reve_model.attention_pooling(out_4d)`.
     * `final_512_emb` (shape `Batch, 512`) is the actual vector you must save to RAM for the compute hack and feed to the linear classifier.

5. **Cross-Validation Dry Run:**
   * Implement the `sklearn.model_selection.KFold` logic for a 10-fold cross-subject split (splitting by subject ID, not by window).
   * **Important:** Add a flag or hardcode the execution loop to *only run Fold 1 for now*. We are doing a dry run to ensure the VRAM limits hold and the metrics track properly before executing all 10 folds.

6. **Logging and Metrics:**
   * Track Accuracy, ROC-AUC, and F1-score using `torchmetrics`.
   * Set up basic Weights & Biases (`WandbLogger`) integration.
   * Save weights of the linear classifier to do inference experiments later on.

**Deliverables:**
Please generate the Python code for:
1. `dataset.py` (Dataset and DataModule)
2. `model.py` (The Lightning Module for the linear classifier and the embedding pre-computation script)
3. `train_lp.py` (The main execution script handling the single-fold dry run)
4. Every other file you think may be needed or useful

Ask for clarifying questions on what you need to do.