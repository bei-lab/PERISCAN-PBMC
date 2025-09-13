"""
PERISCAN Evaluation Module
Model evaluation, prediction generation, and performance analysis.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score
)

from utils import plot_confusion_matrix, calculate_metrics, save_predictions


class PERISCANEvaluator:
    """Evaluator for trained PERISCAN models."""
    
    def __init__(self, trainer, dataloaders, config):
        """
        Initialize evaluator.
        
        Args:
            trainer: Trained PERISCAN trainer instance
            dataloaders: Dictionary with train/val dataloaders  
            config: PERISCAN configuration
        """
        self.trainer = trainer
        self.dataloaders = dataloaders
        self.config = config
        self.device = config.device
        
        # Get class information
        self.classes = trainer.classes
        self.num_classes = trainer.num_classes
        
        print(f"Initialized evaluator for {self.num_classes} classes: {self.classes}")
    
    def load_best_model(self, checkpoint_path=None):
        """
        Load the best trained model.
        
        Args:
            checkpoint_path: Path to specific checkpoint, defaults to best model
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.save_dir, 'checkpoints', 'best_model.pt')
        
        if os.path.exists(checkpoint_path):
            self.trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded best model from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
    
    def predict_dataset(self, mode='val', return_attention=False):
        """
        Generate predictions for a dataset.
        
        Args:
            mode: Dataset mode ('train' or 'val')
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with predictions, probabilities, labels, and sample info
        """
        print(f"Generating predictions for {mode} dataset...")
        
        # Set models to evaluation mode
        self.trainer._set_eval_mode()
        
        # Storage for results
        all_predictions = []
        all_probabilities = []
        all_labels = []
        sample_ids = []
        attention_maps = [] if return_attention else None
        
        dataloader = self.dataloaders[mode]
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Predicting {mode}"):
                # Forward pass
                outputs, labels = self.trainer._forward_pass(batch)
                
                # Collect predictions
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_probabilities.extend(outputs['probs'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                sample_ids.extend(batch['sample_ids'])
                
                # Collect attention maps if requested
                if return_attention:
                    batch_attention = self._collect_batch_attention(batch)
                    attention_maps.append(batch_attention)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        true_labels = np.array(all_labels)
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'sample_ids': sample_ids,
            'classes': self.classes,
            'mode': mode
        }
        
        if return_attention:
            results['attention_maps'] = attention_maps
        
        print(f"Generated {len(predictions)} predictions for {mode} dataset")
        return results
    
    def _collect_batch_attention(self, batch):
        """Collect attention maps for a single batch (simplified version)."""
        # This would collect attention maps from all model components
        # Simplified implementation for demo purposes
        return {
            'batch_size': len(batch['sample_ids']),
            'sample_ids': batch['sample_ids']
        }
    
    def evaluate_model(self, mode='val', save_results=True):
        """
        Comprehensive model evaluation.
        
        Args:
            mode: Dataset mode to evaluate
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"\n=== Evaluating PERISCAN Model on {mode.upper()} Set ===")
        
        # Generate predictions
        results = self.predict_dataset(mode)
        
        y_true = results['true_labels']
        y_pred = results['predictions']
        y_probs = results['probabilities']
        sample_ids = results['sample_ids']
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(y_true, y_pred, y_probs, self.classes)
        
        print(f"\n{mode.upper()} Results:")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class Performance:")
        print(metrics['metrics_df'].to_string(index=False))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if save_results:
            # Save confusion matrix plot
            cm_path = os.path.join(self.config.save_dir, 'visualizations', f'{mode}_confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, self.classes, save_path=cm_path, 
                                title=f'PERISCAN {mode.upper()} Confusion Matrix')
            
            # Save detailed predictions
            pred_path = os.path.join(self.config.save_dir, 'data', f'{mode}_predictions.csv')
            save_predictions(y_pred, y_probs, y_true, sample_ids, self.classes, pred_path)
            
            # Save metrics
            metrics_path = os.path.join(self.config.save_dir, 'data', f'{mode}_metrics.csv')
            metrics['metrics_df'].to_csv(metrics_path, index=False)
            
            print(f"\nResults saved to {self.config.save_dir}")
        
        # Add confusion matrix to results
        evaluation_results = {
            'accuracy': metrics['accuracy'],
            'metrics_df': metrics['metrics_df'],
            'classification_report': metrics['classification_report'],
            'confusion_matrix': cm,
            'predictions': results
        }
        
        return evaluation_results
    
    def compare_train_val_performance(self):
        """Compare model performance on training vs validation sets."""
        print("\n=== Comparing Train vs Validation Performance ===")
        
        train_results = self.evaluate_model('train', save_results=False)
        val_results = self.evaluate_model('val', save_results=True)
        
        comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro F1', 'Weighted F1'],
            'Training': [
                train_results['accuracy'],
                train_results['metrics_df'].iloc[-1]['F1-Score'],  # Macro avg row
                np.average(train_results['metrics_df'].iloc[:-1]['F1-Score'], 
                          weights=train_results['metrics_df'].iloc[:-1]['Support'])
            ],
            'Validation': [
                val_results['accuracy'], 
                val_results['metrics_df'].iloc[-1]['F1-Score'],  # Macro avg row
                np.average(val_results['metrics_df'].iloc[:-1]['F1-Score'],
                          weights=val_results['metrics_df'].iloc[:-1]['Support'])
            ]
        })
        
        comparison['Difference'] = comparison['Validation'] - comparison['Training']
        
        print("\nTrain vs Validation Comparison:")
        print(comparison.to_string(index=False, float_format='%.4f'))
        
        # Check for overfitting
        overfitting_threshold = 0.05
        if any(comparison['Difference'] < -overfitting_threshold):
            print(f"\n⚠️  Warning: Potential overfitting detected (val performance > {overfitting_threshold:.1%} lower than train)")
        else:
            print(f"\n✓ Good generalization (difference < {overfitting_threshold:.1%})")
        
        return comparison
    
    def analyze_misclassifications(self, mode='val', top_k=10):
        """
        Analyze the most confident misclassifications.
        
        Args:
            mode: Dataset mode to analyze
            top_k: Number of top misclassifications to show
            
        Returns:
            DataFrame with misclassification analysis
        """
        print(f"\n=== Analyzing Misclassifications in {mode.upper()} Set ===")
        
        # Get predictions
        results = self.predict_dataset(mode)
        
        y_true = results['true_labels']
        y_pred = results['predictions']
        y_probs = results['probabilities']
        sample_ids = results['sample_ids']
        
        # Find misclassifications
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return pd.DataFrame()
        
        # Get confidence scores for misclassifications
        misclassified_confidences = np.max(y_probs[misclassified_indices], axis=1)
        
        # Sort by confidence (most confident mistakes first)
        sorted_indices = np.argsort(misclassified_confidences)[::-1]
        top_misclass_indices = misclassified_indices[sorted_indices[:top_k]]
        
        # Create analysis DataFrame
        analysis_data = []
        for idx in top_misclass_indices:
            true_class = self.classes[y_true[idx]]
            pred_class = self.classes[y_pred[idx]]
            confidence = misclassified_confidences[sorted_indices[np.where(misclassified_indices == idx)[0][0]]]
            
            analysis_data.append({
                'Sample ID': sample_ids[idx],
                'True Class': true_class,
                'Predicted Class': pred_class,
                'Confidence': confidence,
                'True Class Prob': y_probs[idx, y_true[idx]],
                'Pred Class Prob': y_probs[idx, y_pred[idx]]
            })
        
        misclass_df = pd.DataFrame(analysis_data)
        
        print(f"Top {len(misclass_df)} Most Confident Misclassifications:")
        print(misclass_df.to_string(index=False, float_format='%.4f'))
        
        # Save analysis
        save_path = os.path.join(self.config.save_dir, 'data', f'{mode}_misclassification_analysis.csv')
        misclass_df.to_csv(save_path, index=False)
        print(f"\nMisclassification analysis saved to {save_path}")
        
        return misclass_df
    
    def generate_performance_summary(self):
        """Generate a comprehensive performance summary."""
        print("\n=== PERISCAN Model Performance Summary ===")
        
        # Load best model
        self.load_best_model()
        
        # Evaluate both sets
        train_results = self.evaluate_model('train', save_results=False)
        val_results = self.evaluate_model('val', save_results=True)
        
        # Compare performance
        comparison = self.compare_train_val_performance()
        
        # Analyze misclassifications
        misclass_analysis = self.analyze_misclassifications('val')
        
        # Create summary
        summary = {
            'model_info': {
                'num_classes': self.num_classes,
                'classes': self.classes,
                'best_epoch': self.trainer.best_epoch + 1,
                'best_val_loss': self.trainer.best_val_loss
            },
            'train_performance': {
                'accuracy': train_results['accuracy'],
                'macro_f1': train_results['metrics_df'].iloc[-1]['F1-Score']
            },
            'val_performance': {
                'accuracy': val_results['accuracy'],
                'macro_f1': val_results['metrics_df'].iloc[-1]['F1-Score']
            },
            'generalization': {
                'accuracy_diff': comparison[comparison['Metric'] == 'Accuracy']['Difference'].iloc[0],
                'f1_diff': comparison[comparison['Metric'] == 'Macro F1']['Difference'].iloc[0]
            },
            'misclassifications': {
                'total': len(misclass_analysis),
                'avg_confidence': misclass_analysis['Confidence'].mean() if len(misclass_analysis) > 0 else 0
            }
        }
        
        print("\nSummary:")
        print(f"  Best Model: Epoch {summary['model_info']['best_epoch']} (Val Loss: {summary['model_info']['best_val_loss']:.4f})")
        print(f"  Validation Accuracy: {summary['val_performance']['accuracy']:.4f}")
        print(f"  Validation Macro F1: {summary['val_performance']['macro_f1']:.4f}")
        print(f"  Generalization Gap: {abs(summary['generalization']['accuracy_diff']):.4f}")
        
        return summary


def evaluate_trained_model(trainer, dataloaders, config, checkpoint_path=None):
    """
    Evaluate a trained PERISCAN model.
    
    Args:
        trainer: Trained PERISCAN trainer
        dataloaders: Dictionary with dataloaders
        config: PERISCAN configuration
        checkpoint_path: Optional specific checkpoint to load
        
    Returns:
        Performance summary and detailed results
    """
    # Create evaluator
    evaluator = PERISCANEvaluator(trainer, dataloaders, config)
    
    # Load best model if checkpoint provided
    if checkpoint_path:
        evaluator.load_best_model(checkpoint_path)
    else:
        evaluator.load_best_model()
    
    # Generate comprehensive performance summary
    summary = evaluator.generate_performance_summary()
    
    return evaluator, summary