"""
Training and evaluation functions for EEG classification.
"""

import torch
from tqdm.auto import tqdm
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


class Trainer:
    """Handles training and evaluation of the model."""
    
    def __init__(self, model, device, criterion):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            device: torch.device to train on
            criterion: Loss function
        """
        self.model = model
        self.device = device
        self.criterion = criterion
    
    def train_one_epoch(self, optimizer, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            optimizer: Optimizer for parameter updates
            train_loader: DataLoader for training data
        """
        self.model.train()
        pbar = tqdm(train_loader, desc="Training", total=len(train_loader))
        
        for batch_data in pbar:
            data, target, pos = (
                batch_data["sample"].to(self.device, non_blocking=True),
                batch_data["label"].to(self.device, non_blocking=True),
                batch_data["pos"].to(self.device, non_blocking=True),
            )
            
            optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast(
                dtype=torch.float16,
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                output = self.model(data, pos)
            
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": loss.item()})
    
    def evaluate(self, val_loader):
        """
        Evaluate the model on a validation/test set.
        
        Args:
            val_loader: DataLoader for validation/test data
            
        Returns:
            tuple: (acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr)
        """
        self.model.eval()
        
        y_decisions = []
        y_targets = []
        y_probs = []
        score, count = 0, 0
        
        pbar = tqdm(val_loader, desc="Evaluating", total=len(val_loader))
        
        with torch.inference_mode():
            for batch_data in pbar:
                data, target, pos = (
                    batch_data["sample"].to(self.device, non_blocking=True),
                    batch_data["label"].to(self.device, non_blocking=True),
                    batch_data["pos"].to(self.device, non_blocking=True),
                )
                
                # Forward pass with automatic mixed precision
                with torch.amp.autocast(
                    dtype=torch.float16,
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    output = self.model(data, pos)
                
                decisions = torch.argmax(output, dim=1)
                score += (decisions == target).int().sum().item()
                count += target.shape[0]
                
                y_decisions.append(decisions)
                y_targets.append(target)
                y_probs.append(output)
        
        # Compute metrics
        gt = torch.cat(y_targets).cpu().numpy()
        pr = torch.cat(y_decisions).cpu().numpy()
        pr_probs = torch.cat(y_probs).cpu().numpy()
        
        acc = score / count
        balanced_acc = balanced_accuracy_score(gt, pr)
        cohen_kappa = cohen_kappa_score(gt, pr)
        f1 = f1_score(gt, pr, average="weighted")
        auroc = roc_auc_score(gt, pr_probs[:, 1])
        auc_pr = average_precision_score(gt, pr_probs[:, 1])
        
        return acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr


def print_results(acc, balanced_acc, cohen_kappa, f1, auroc, auc_pr):
    """
    Print evaluation metrics.
    
    Args:
        acc: Accuracy
        balanced_acc: Balanced accuracy
        cohen_kappa: Cohen's kappa score
        f1: F1 score
        auroc: Area under ROC curve
        auc_pr: Area under precision-recall curve
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:              {acc:.4f}")
    print(f"Balanced Accuracy:     {balanced_acc:.4f}")
    print(f"Cohen's Kappa:         {cohen_kappa:.4f}")
    print(f"F1 Score:              {f1:.4f}")
    print(f"AUROC:                 {auroc:.4f}")
    print(f"AUC-PR:                {auc_pr:.4f}")
    print("="*50 + "\n")
