"""
REVE EEG Classification Project

This project demonstrates how to use the REVE model as a feature extractor
for EEG classification tasks using the EEGMAT dataset.

## Project Structure

- `config.py`: Configuration and hyperparameters
- `model.py`: Model initialization and setup
- `data.py`: Data loading and preprocessing
- `training.py`: Training and evaluation logic
- `main.py`: Main training script

## Installation

Install required dependencies:
```bash
pip install torch transformers datasets scikit-learn tqdm
```

## Hugging Face Authentication

Since the REVE model is gated, you need to authenticate first:
```bash
huggingface-cli login
```

## Usage

To run the complete training pipeline:
```bash
python main.py
```

## Notes

This implementation provides a basic training loop that:
- Freezes the REVE backbone
- Fine-tunes only the classification head
- Uses automatic mixed precision for efficiency
- Saves the best model based on validation balanced accuracy

Features NOT included (as per original notebook):
- Model souping
- LoRA wrappers
- Channel Mixup
- Position augmentation
- Two-stage fine-tuning
- Stable AdamW optimizer

Results may differ slightly from the paper due to these simplifications.
"""

__version__ = "1.0.0"
__author__ = "REVE Team"
