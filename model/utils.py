"""
PERISCAN Utility Functions
Common utility functions for data preprocessing, visualization, and model training.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from pathlib import Path


def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_directories(base_dir):
    """Create necessary directories for saving results."""
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'metrics'),
        os.path.join(base_dir, 'visualizations'),
        os.path.join(base_dir, 'data')
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return dirs_to_create


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.nan_to_num(cm_percent)
    
    plt.figure(figsize=(10, 8))
    
    # Create annotations with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm_percent[i, j]:.1f}%\n({cm[i, j]})')
        annotations.append(row)
    
    sns.heatmap(cm_percent, 
                annot=annotations, 
                fmt='', 
                cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('F1 Score Over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Over Epochs')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def calculate_metrics(y_true, y_pred, y_probs, class_names):
    """Calculate comprehensive performance metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        classification_report, roc_auc_score
    )
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names + ['Macro Avg'],
        'Precision': list(precision) + [macro_precision],
        'Recall': list(recall) + [macro_recall],
        'F1-Score': list(f1) + [macro_f1],
        'Support': list(support) + [np.sum(support)]
    })
    
    # Calculate AUC for multiclass (one-vs-rest)
    try:
        if len(class_names) > 2:
            auc_scores = roc_auc_score(y_true, y_probs, multi_class='ovr', average=None)
            macro_auc = np.mean(auc_scores)
            metrics_df['AUC'] = list(auc_scores) + [macro_auc]
    except:
        print("Warning: Could not calculate AUC scores")
    
    return {
        'accuracy': accuracy,
        'metrics_df': metrics_df,
        'classification_report': classification_report(y_true, y_pred, target_names=class_names)
    }


def save_predictions(predictions, probabilities, true_labels, sample_ids, class_names, save_path):
    """Save prediction results to CSV file."""
    results_df = pd.DataFrame({
        'sample_id': sample_ids,
        'true_label': [class_names[label] for label in true_labels],
        'predicted_label': [class_names[pred] for pred in predictions],
        'correct': predictions == true_labels
    })
    
    # Add probability columns
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = probabilities[:, i]
    
    # Add confidence (max probability)
    results_df['confidence'] = np.max(probabilities, axis=1)
    
    results_df.to_csv(save_path, index=False)
    print(f"Prediction results saved to {save_path}")
    
    return results_df


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA is not available. Using CPU.")


def check_data_quality(adata):
    """Check basic data quality metrics."""
    print("=== Data Quality Check ===")
    print(f"Total samples: {len(adata.obs['pbmc_sample'].unique())}")
    print(f"Total cells: {adata.n_obs}")
    print(f"Total genes: {adata.n_vars}")
    
    print("\nSample distribution by cancer type:")
    sample_dist = adata.obs.groupby('cancertype')['pbmc_sample'].nunique().sort_values(ascending=False)
    print(sample_dist)
    
    print("\nCell type distribution:")
    cell_dist = adata.obs['cell_type_merged'].value_counts()
    print(cell_dist)
    
    print("\nCells per sample statistics:")
    cells_per_sample = adata.obs.groupby('pbmc_sample').size()
    print(f"Mean: {cells_per_sample.mean():.0f}")
    print(f"Median: {cells_per_sample.median():.0f}")
    print(f"Min: {cells_per_sample.min()}")
    print(f"Max: {cells_per_sample.max()}")


def format_time(seconds):
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"