"""
Main training script for REVE EEG classification.

This script demonstrates how to use the REVE model as a feature extractor
for EEG classification on the EEGMAT dataset.
"""

import torch
from transformers import set_seed

from config import (
    BATCH_SIZE,
    N_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from model import (
    load_reve_model,
    setup_model,
    get_device,
    freeze_backbone,
    inspect_model,
)
from data import load_eeg_dataset
from training import Trainer, print_results


def main():
    """Main training pipeline."""
    
    # Set random seed for reproducibility
    print(f"Setting random seed to {RANDOM_SEED}")
    set_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Get device
    device = get_device()
    
    # Load REVE model and position bank
    model, pos_bank = load_reve_model()
    
    # Show pre-registered positions
    print("\nAvailable EEG positions in position bank:")
    print(pos_bank.get_all_positions())
    
    # Inspect model before modification
    print("\nModel before setup:")
    inspect_model(model)
    
    # Setup model with classification head
    model, positions = setup_model(model, pos_bank)
    
    # Freeze backbone, only train final layer
    freeze_backbone(model)
    
    # Inspect model after setup
    print("\nModel after setup:")
    inspect_model(model)
    
    # Move model to device
    model.to(device)
    
    # Load dataset
    train_loader, val_loader, test_loader = load_eeg_dataset(
        BATCH_SIZE, positions
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.final_layer.parameters(),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )
    
    # Setup loss function and trainer
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, device, criterion)
    
    # Training loop
    print(f"\nStarting training for {N_EPOCHS} epochs...")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}\n")
    
    best_val_acc = 0
    best_final_layer = None
    
    for epoch in range(N_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        print('='*50)
        
        # Train
        trainer.train_one_epoch(optimizer, train_loader)
        
        # Evaluate on validation set
        acc, b_acc, kappa, f1, auroc, auc_pr = trainer.evaluate(val_loader)
        
        # Save best model
        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_final_layer = model.final_layer.state_dict()
            print(f"✓ New best validation balanced accuracy: {b_acc:.4f}")
        
        print(f"Validation balanced accuracy: {b_acc:.4f} (best: {best_val_acc:.4f})")
        
        # Step scheduler
        scheduler.step(b_acc)
    
    # Load best model and evaluate on test set
    print(f"\n{'='*50}")
    print("TESTING")
    print('='*50)
    print("Loading best model...")
    model.final_layer.load_state_dict(best_final_layer)
    
    acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr = trainer.evaluate(
        test_loader
    )
    
    # Print results
    print_results(acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr)
    
    return model


if __name__ == "__main__":
    main()
