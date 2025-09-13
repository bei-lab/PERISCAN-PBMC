"""
PERISCAN Training Module
Main training loop and model management for PERISCAN.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

from models.gene_encoder import create_gene_encoders
from models.cell_encoder import create_cell_encoders
from models.feature_fusion import create_feature_fusion
from models.classifier import create_classifier_and_loss
from data.dataset import get_class_weights
from utils import format_time


class PERISCANTrainer:
    """Main trainer class for PERISCAN model."""
    
    def __init__(self, adata, dataloaders, gene_lists, max_cells_dict, config):
        """
        Initialize PERISCAN trainer.
        
        Args:
            adata: AnnData object
            dataloaders: Dictionary with train/val dataloaders
            gene_lists: Dictionary with genes per cell type
            max_cells_dict: Dictionary with max cells per cell type
            config: PERISCAN configuration
        """
        self.adata = adata
        self.dataloaders = dataloaders
        self.gene_lists = gene_lists
        self.max_cells_dict = max_cells_dict
        self.config = config
        self.device = config.device
        
        # Get cell types and classes
        self.cell_types = sorted(adata.obs['cell_type_merged'].unique())
        self.classes = sorted(adata.obs['cancertype'].unique())
        self.num_classes = len(self.classes)
        
        # Initialize models
        self._init_models()
        self._init_optimizers()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.history = defaultdict(list)
        self.early_stop_counter = 0
        
        print(f"Initialized PERISCAN trainer on {self.device}")
        print(f"Cell types: {self.cell_types}")
        print(f"Classes: {self.classes}")
    
    def _init_models(self):
        """Initialize all model components."""
        print("Initializing PERISCAN models...")
        
        # Gene encoders
        self.gene_encoders = create_gene_encoders(self.gene_lists, self.config)
        
        # Cell encoders
        self.cell_encoders = create_cell_encoders(self.cell_types, self.config)
        
        # Feature fusion
        self.fusion_model = create_feature_fusion(self.config)
        
        # Classifier and loss
        class_weights = get_class_weights(self.adata, self.device)
        self.classifier, self.loss_fn = create_classifier_and_loss(
            self.num_classes, class_weights, self.config
        )
        
        # Move to device
        self.gene_encoders = {k: v.to(self.device) for k, v in self.gene_encoders.items()}
        self.cell_encoders = {k: v.to(self.device) for k, v in self.cell_encoders.items()}
        self.fusion_model = self.fusion_model.to(self.device)
        self.classifier = self.classifier.to(self.device)
    
    def _init_optimizers(self):
        """Initialize optimizers for all model components."""
        print("Initializing optimizers...")
        
        # Collect all parameters
        all_params = []
        
        # Gene encoder parameters
        for encoder in self.gene_encoders.values():
            all_params.extend(encoder.parameters())
        
        # Cell encoder parameters
        for encoder in self.cell_encoders.values():
            all_params.extend(encoder.parameters())
        
        # Fusion and classifier parameters
        all_params.extend(self.fusion_model.parameters())
        all_params.extend(self.classifier.parameters())
        
        # Single optimizer for all parameters
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.max_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
    
    def _set_train_mode(self):
        """Set all models to training mode."""
        for model in self.gene_encoders.values():
            model.train()
        for model in self.cell_encoders.values():
            model.train()
        self.fusion_model.train()
        self.classifier.train()
    
    def _set_eval_mode(self):
        """Set all models to evaluation mode."""
        for model in self.gene_encoders.values():
            model.eval()
        for model in self.cell_encoders.values():
            model.eval()
        self.fusion_model.eval()
        self.classifier.eval()
    
    def _forward_pass(self, batch):
        """
        Forward pass through all model components.
        
        Args:
            batch: Batch data from dataloader
            
        Returns:
            Model outputs
        """
        # Prepare data
        cells = {k: v.to(self.device) for k, v in batch['cells'].items()}
        masks = {k: v.to(self.device) for k, v in batch['mask'].items()}
        labels = batch['label'].to(self.device)
        
        # Gene encoding
        gene_features = {}
        for ct in self.cell_types:
            gene_out, _ = self.gene_encoders[ct](cells[ct], masks[ct])
            gene_features[ct] = gene_out
        
        # Cell encoding
        cell_features = {}
        for ct in self.cell_types:
            cell_out, _ = self.cell_encoders[ct](
                gene_features[ct], 
                gene_features[ct], 
                masks[ct]
            )
            cell_features[ct] = cell_out
        
        # Feature fusion
        fused_features, _ = self.fusion_model(cell_features, masks)
        
        # Classification
        outputs = self.classifier(fused_features)
        
        return outputs, labels
    
    def train_epoch(self):
        """Train for one epoch."""
        self._set_train_mode()
        
        epoch_metrics = defaultdict(float)
        all_preds = []
        all_labels = []
        num_batches = len(self.dataloaders['train'])
        
        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            outputs, labels = self._forward_pass(batch)
            
            # Calculate loss
            loss, loss_dict = self.loss_fn(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for model in [self.gene_encoders, self.cell_encoders, 
                                   self.fusion_model, self.classifier] 
                     for p in (model.parameters() if hasattr(model, 'parameters') 
                              else [param for m in model.values() for param in m.parameters()])],
                    self.config.gradient_clip_norm
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Collect metrics
            batch_metrics = self.loss_fn.compute_metrics(outputs, labels)
            batch_metrics.update(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_metrics["accuracy"]:.4f}'
            })
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[k] += v
            
            # Collect predictions
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch averages
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score
        epoch_metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro')
        epoch_metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self._set_eval_mode()
        
        val_metrics = defaultdict(float)
        all_preds = []
        all_labels = []
        num_batches = len(self.dataloaders['val'])
        
        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc='Validation')
            
            for batch in pbar:
                # Forward pass
                outputs, labels = self._forward_pass(batch)
                
                # Calculate loss and metrics
                loss, loss_dict = self.loss_fn(outputs, labels)
                batch_metrics = self.loss_fn.compute_metrics(outputs, labels)
                batch_metrics.update(loss_dict)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{batch_metrics["accuracy"]:.4f}'
                })
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    if isinstance(v, (int, float)):
                        val_metrics[k] += v
                
                # Collect predictions
                all_preds.extend(outputs['predictions'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch averages
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score
        val_metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro')
        val_metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted')
        
        return val_metrics
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.max_epochs} epochs...")
        start_time = time.time()
        
        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validate epoch
                val_metrics = self.validate_epoch()
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Save metrics
                for k, v in train_metrics.items():
                    self.history[f'train_{k}'].append(v)
                for k, v in val_metrics.items():
                    self.history[f'val_{k}'].append(v)
                self.history['learning_rate'].append(current_lr)
                
                # Print epoch results
                print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}:")
                print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}, "
                      f"F1: {train_metrics['macro_f1']:.4f}")
                print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['macro_f1']:.4f}")
                print(f"  LR: {current_lr:.2e}")
                
                # Check for best model
                val_loss = val_metrics['total_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stop_counter = 0
                    self.save_checkpoint('best_model.pt')
                    print(f"  ✓ New best model saved! (Loss: {val_loss:.4f})")
                else:
                    self.early_stop_counter += 1
                
                # Regular checkpoint save
                if (epoch + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
                
                # Early stopping
                if self.early_stop_counter >= self.config.early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        # Training complete
        duration = time.time() - start_time
        print(f"\nTraining completed in {format_time(duration)}")
        print(f"Best model at epoch {self.best_epoch + 1} with val loss: {self.best_val_loss:.4f}")
        
        # Save final results
        self._save_training_results()
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'gene_encoders': {k: v.state_dict() for k, v in self.gene_encoders.items()},
            'cell_encoders': {k: v.state_dict() for k, v in self.cell_encoders.items()},
            'fusion_model': self.fusion_model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history),
            'config': self.config.__dict__
        }
        
        save_path = os.path.join(self.config.save_dir, 'checkpoints', filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model states
        for k, v in checkpoint['gene_encoders'].items():
            self.gene_encoders[k].load_state_dict(v)
        for k, v in checkpoint['cell_encoders'].items():
            self.cell_encoders[k].load_state_dict(v)
        self.fusion_model.load_state_dict(checkpoint['fusion_model'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = defaultdict(list, checkpoint['history'])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def _save_training_results(self):
        """Save final training results and history."""
        results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_metrics': {
                k: v[-1] for k, v in self.history.items() 
                if k.startswith('train_') and len(v) > 0
            },
            'final_val_metrics': {
                k: v[-1] for k, v in self.history.items() 
                if k.startswith('val_') and len(v) > 0
            },
            'history': dict(self.history),
            'config': self.config.__dict__
        }
        
        # Save training history
        history_path = os.path.join(self.config.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to {history_path}")


def train_periscan_model(adata, dataloaders, gene_lists, max_cells_dict, config):
    """
    Train PERISCAN model.
    
    Args:
        adata: AnnData object
        dataloaders: Train/val dataloaders
        gene_lists: Gene lists per cell type
        max_cells_dict: Max cells per cell type
        config: PERISCAN configuration
        
    Returns:
        Trained model and training history
    """
    # Create trainer
    trainer = PERISCANTrainer(
        adata=adata,
        dataloaders=dataloaders,
        gene_lists=gene_lists,
        max_cells_dict=max_cells_dict,
        config=config
    )
    
    # Train model
    history = trainer.train()
    
    return trainer, history