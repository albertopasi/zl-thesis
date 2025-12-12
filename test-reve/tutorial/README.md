# REVE EEG Classification

A Python project for training classification models on EEG data using REVE, a powerful pre-trained feature extractor.

## Overview

This project translates the [REVE tutorial Jupyter notebook](https://brain-bzh.github.io/reve/) into a modular Python project structure. It demonstrates how to:

1. Load the REVE model from Hugging Face
2. Replace the final layer with a classification head
3. Fine-tune on the EEGMAT EEG classification dataset
4. Evaluate using multiple metrics (accuracy, balanced accuracy, Cohen's kappa, F1, AUROC, AUC-PR)

## Project Structure

```
src/
├── __init__.py          # Package initialization and documentation
├── config.py            # Configuration and hyperparameters
├── model.py             # Model loading, setup, and inspection
├── data.py              # Dataset loading and preprocessing
├── training.py          # Trainer class and evaluation utilities
└── main.py              # Main training script entry point
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- Transformers
- Datasets
- scikit-learn
- tqdm

Install dependencies:
```bash
pip install torch transformers datasets scikit-learn tqdm
```

## Getting Started

### 1. Hugging Face Authentication

The REVE model is gated. You need to accept the terms and authenticate:

```bash
huggingface-cli login
```

Or in Python:
```python
from huggingface_hub import login
login()
```

### 2. Running the Training

Execute the main script from the `src` directory:

```bash
cd src
python main.py
```

## Configuration

Edit `src/config.py` to adjust:

- `BATCH_SIZE`: Training batch size (default: 64)
- `N_EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate for optimizer (default: 1e-3)
- `RANDOM_SEED`: Seed for reproducibility (default: 42)

## Features

- **Modular design**: Separate concerns into dedicated modules
- **Clean API**: Well-documented functions and classes
- **Reproducibility**: Fixed random seeds and deterministic behavior
- **GPU support**: Automatic detection and efficient training
- **Mixed precision**: FP16 training for faster computation
- **Progress tracking**: TQDM progress bars for training/evaluation
- **Comprehensive metrics**: Multiple evaluation metrics computed

## Model Architecture

The final classification head consists of:
1. Flatten layer
2. RMSNorm normalization
3. Dropout (0.1)
4. Linear layer to 2 classes

The backbone REVE model is frozen during training.

## Output

The script prints:
- Model architecture and parameter counts
- Training progress with loss values
- Validation balanced accuracy at each epoch
- Final test set metrics:
  - Accuracy
  - Balanced Accuracy
  - Cohen's Kappa
  - F1 Score
  - AUROC
  - AUC-PR

## References

- REVE Model: https://huggingface.co/collections/brain-bzh/reve
- EEGMAT Dataset: https://huggingface.co/datasets/brain-bzh/eegmat-prepro
- Original PhysioNet Dataset: https://physionet.org/content/eegmat/1.0.0/

## Notes

This implementation provides a basic training loop. Advanced techniques not included:
- Model souping
- LoRA adapters
- Channel mixup augmentation
- Position augmentation
- Two-stage fine-tuning
- Stable AdamW optimizer

Results may differ slightly from the original paper due to these simplifications.
