"""
Comprehensive evaluation of workload classifier across multiple scenarios.
Modular design with clean separation of concerns.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, confusion_matrix, classification_report
)

from config import OUTPUT_DIR, EVAL_LEARNING_RATES, EVAL_SEEDS, EVAL_NUM_EPOCHS


# ============================================================================
# DATA LOADING
# ============================================================================

def load_subject_data(subject_id):
    """Load features and labels for a specific subject."""
    features = np.load(f"{OUTPUT_DIR}/{subject_id}_features.npy")
    labels = np.load(f"{OUTPUT_DIR}/{subject_id}_labels.npy")
    return features, labels


def load_all_data():
    """Load data from both subjects."""
    data = {}
    for subject_id in ["sub-PD089_ses-S001", "sub-PD094_ses-S001"]:
        features, labels = load_subject_data(subject_id)
        data[subject_id] = {
            "features": features,
            "labels": labels,
            "n_epochs": len(features)
        }
    return data


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def prepare_data(features, labels, test_size=0.2, random_state=42, pca_components=512):
    """
    Split data into train/test with stratification and PCA dimensionality reduction.
    
    Args:
        features: Original features
        labels: Binary labels (0 or 1)
        test_size: Fraction for test set
        random_state: Random seed
        pca_components: Number of dimensions after PCA compression
    
    Returns:
        dict with keys: X_train, X_test, y_train, y_test, label_encoder, scaler, pca
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Scale features first (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Apply PCA for dimensionality reduction
    # Be aggressive: reduce to 10x number of samples for better regularization
    # Thi helps model learn patterns instead of memorizing
    n_samples = features.shape[0]
    target_components = min(n_samples * 10, 100)  # Target 10:1 sample-to-feature ratio, max 100 dims
    n_components = min(pca_components, target_components, n_samples - 1)
    print(f"  Applying PCA: {features.shape[1]} → {n_components} dimensions...")
    pca = IncrementalPCA(n_components=n_components, batch_size=16)
    X_reduced = pca.fit_transform(X_scaled)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"    Explained variance: {explained_var:.1%} ({n_samples} samples, {n_components} dims = {n_samples/n_components:.1f}:1 ratio)")
    
    # Split: train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "pca": pca
    }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class WorkloadClassifier(nn.Module):
    """Simple neural network classifier for workload prediction (binary: task vs no-task)."""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# TRAINING
# ============================================================================

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    """Create PyTorch DataLoaders (train/test only)."""
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_and_evaluate(X_train, X_test, y_train, y_test, 
                       label_encoder, learning_rate=0.001, seed=42):
    """
    Train model and evaluate on test set.
    
    Returns:
        dict with metrics and model predictions
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} | LR: {learning_rate} | Seed: {seed}")
    
    # Create model
    input_dim = X_train.shape[1]
    model = WorkloadClassifier(input_dim=input_dim, num_classes=len(label_encoder.classes_))
    model.to(device)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)  # L2 regularization
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(EVAL_NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            train_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}/{EVAL_NUM_EPOCHS} | Loss: {train_loss:.4f}")
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    test_acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True
    )
    
    return {
        "accuracy": test_acc,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "f1_weighted": f1,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "learning_rate": learning_rate,
        "seed": seed,
        "test_preds": all_preds.tolist(),
        "test_labels": all_labels.tolist()
    }




# ============================================================================
# SCENARIOS
# ============================================================================

def scenario_combined_split(data):
    """Scenario 1: Train/test on combined data from both subjects (multiple LR & seeds)."""
    print("\nSCENARIO 1: Combined Split")
    
    # Combine data
    all_features = np.vstack([data[sid]["features"] for sid in data.keys()])
    all_labels = np.concatenate([data[sid]["labels"] for sid in data.keys()])
    
    print(f"  Total epochs: {len(all_features)}")
    
    # Prepare data
    split_data = prepare_data(all_features, all_labels)
    print(f"  Train: {len(split_data['X_train'])} | Test: {len(split_data['X_test'])}")
    
    # Run with different LRs and seeds
    all_results = {}
    for lr in EVAL_LEARNING_RATES:
        for seed in EVAL_SEEDS:
            key = f"lr{lr}_seed{seed}"
            results = train_and_evaluate(
                split_data["X_train"], split_data["X_test"],
                split_data["y_train"], split_data["y_test"],
                split_data["label_encoder"],
                learning_rate=lr,
                seed=seed
            )
            all_results[key] = results
    
    return all_results


def scenario_subject_only(subject_id, data):
    """Scenario 2/3: Train/test on single subject (multiple LR & seeds)."""
    print(f"\nSCENARIO: {subject_id} Only")
    
    features = data[subject_id]["features"]
    labels = data[subject_id]["labels"]
    
    print(f"  Total epochs: {len(features)}")
    
    # Prepare data
    split_data = prepare_data(features, labels)
    print(f"  Train: {len(split_data['X_train'])} | Test: {len(split_data['X_test'])}")
    
    # Run with different LRs and seeds
    all_results = {}
    for lr in EVAL_LEARNING_RATES:
        for seed in EVAL_SEEDS:
            key = f"lr{lr}_seed{seed}"
            results = train_and_evaluate(
                split_data["X_train"], split_data["X_test"],
                split_data["y_train"], split_data["y_test"],
                split_data["label_encoder"],
                learning_rate=lr,
                seed=seed
            )
            all_results[key] = results
    
    return all_results


def scenario_cross_subject(train_subject, test_subject, data):
    """Scenario 4/5: Train on one subject, test on another (multiple LR & seeds)."""
    print(f"\nSCENARIO: {train_subject} → {test_subject}")
    
    # Get data
    X_train_full = data[train_subject]["features"]
    y_train_full = data[train_subject]["labels"]
    X_test_full = data[test_subject]["features"]
    y_test_full = data[test_subject]["labels"]
    
    print(f"  Train subject epochs: {len(X_train_full)}")
    print(f"  Test subject epochs: {len(X_test_full)}")
    
    # Encode labels with combined vocabulary
    all_labels = np.concatenate([y_train_full, y_test_full])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    y_train_encoded = label_encoder.transform(y_train_full)
    y_test_encoded = label_encoder.transform(y_test_full)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    print(f"  Train: {len(X_train_scaled)} | Test: {len(X_test_scaled)}")
    
    # Run with different LRs and seeds
    all_results = {}
    for lr in EVAL_LEARNING_RATES:
        for seed in EVAL_SEEDS:
            key = f"lr{lr}_seed{seed}"
            results = train_and_evaluate(
                X_train_scaled, X_test_scaled,
                y_train_encoded, y_test_encoded,
                label_encoder,
                learning_rate=lr,
                seed=seed
            )
            all_results[key] = results
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all scenarios and save results."""
    print("=" * 70)
    print("COMPREHENSIVE WORKLOAD CLASSIFIER EVALUATION")
    print(f"Epochs: {EVAL_NUM_EPOCHS} | LRs: {EVAL_LEARNING_RATES} | Seeds: {EVAL_SEEDS}")
    print("=" * 70)
    
    # Load data
    print("\n[Loading data...]")
    data = load_all_data()
    for subject_id, subject_data in data.items():
        print(f"  {subject_id}: {subject_data['n_epochs']} epochs, {subject_data['features'].shape[1]} dims")
    
    # Run all scenarios
    results = {}
    
    results["combined_split"] = scenario_combined_split(data)
    results["subject_PD089_only"] = scenario_subject_only("sub-PD089_ses-S001", data)
    results["subject_PD094_only"] = scenario_subject_only("sub-PD094_ses-S001", data)
    results["cross_subject_PD089_to_PD094"] = scenario_cross_subject(
        "sub-PD089_ses-S001", "sub-PD094_ses-S001", data
    )
    results["cross_subject_PD094_to_PD089"] = scenario_cross_subject(
        "sub-PD094_ses-S001", "sub-PD089_ses-S001", data
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for scenario_name, scenario_results in results.items():
        print(f"\n{scenario_name}:")
        accs = [m["accuracy"] for m in scenario_results.values()]
        print(f"  Accuracy range: {min(accs):.4f} - {max(accs):.4f} (mean: {np.mean(accs):.4f})")
        print(f"  Best config: {max(scenario_results.items(), key=lambda x: x[1]['accuracy'])[0]} = {max(accs):.4f}")
    
    # Save results
    output_path = f"{OUTPUT_DIR}/comprehensive_results.json"
    
    # Remove non-serializable data for JSON
    results_for_json = {}
    for scenario_name, scenario_results in results.items():
        results_for_json[scenario_name] = {}
        for config_name, metrics in scenario_results.items():
            results_for_json[scenario_name][config_name] = {
                k: v for k, v in metrics.items()
                if k not in ["test_preds", "test_labels"]  # Remove lists
            }
    
    with open(output_path, "w") as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
