"""
PERISCAN Classifier Module
Final classification head and loss function for cancer type prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


class PERISCANClassifier(nn.Module):
    """Final classifier for PERISCAN model."""
    
    def __init__(self, input_dim=256, hidden_dim=256, num_classes=3, dropout=0.3, temperature=1.0):
        """
        Initialize PERISCAN classifier.
        
        Args:
            input_dim: Input feature dimension from fusion module
            hidden_dim: Hidden dimension for processing
            num_classes: Number of output classes
            dropout: Dropout rate
            temperature: Temperature for softmax (for calibration)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Feature enhancement network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass of classifier.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with logits, probabilities, predictions, and features
        """
        # Handle NaN inputs
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature processing
        features_mid = self.feature_net[:4](x)  # First layer features
        features_mid = torch.nan_to_num(features_mid, nan=0.0)
        
        features = self.feature_net(x)  # Complete features
        features = torch.nan_to_num(features, nan=0.0)
        
        # Clamp features to prevent extreme values
        features = torch.clamp(features, -10.0, 10.0)
        
        # Classification
        hidden = self.classifier[:-1](features)  # Hidden layer before final classification
        hidden = torch.nan_to_num(hidden, nan=0.0)
        
        logits = self.classifier(features)  # Final logits
        logits = torch.nan_to_num(logits, nan=0.0)
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Handle any remaining NaNs in probabilities
        probs = torch.nan_to_num(probs, nan=0.0)
        row_sums = probs.sum(dim=1, keepdim=True)
        valid_rows = row_sums > 0
        probs = torch.where(
            valid_rows,
            probs / torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums)),
            torch.ones_like(probs) / self.num_classes  # Uniform distribution for invalid rows
        )
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': torch.argmax(logits, dim=1),
            'features': {
                'input_features': x,
                'mid_features': features_mid,
                'final_features': features,
                'classifier_hidden': hidden
            }
        }


class PERISCANLoss(nn.Module):
    """Loss function for PERISCAN with class balancing and label smoothing."""
    
    def __init__(self, num_classes, class_weights=None, label_smoothing=0.1, device='cpu'):
        """
        Initialize PERISCAN loss function.
        
        Args:
            num_classes: Number of classes
            class_weights: Optional class weights for imbalanced datasets
            label_smoothing: Label smoothing factor
            device: Device to run on
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.device = device
        
        # Set class weights
        if class_weights is not None:
            self.class_weights = class_weights.to(device)
        else:
            self.class_weights = torch.ones(num_classes).to(device)
        
        print(f"Initialized PERISCAN loss with {num_classes} classes")
        print(f"Class weights: {self.class_weights}")
    
    def forward(self, outputs, labels):
        """
        Calculate loss and metrics.
        
        Args:
            outputs: Dictionary from classifier
            labels: True labels [batch_size]
            
        Returns:
            Loss value and loss dictionary
        """
        batch_size = labels.size(0)
        
        # Label smoothing
        soft_labels = torch.zeros(batch_size, self.num_classes).to(self.device)
        soft_labels.fill_(self.label_smoothing / (self.num_classes - 1))
        soft_labels.scatter_(1, labels.unsqueeze(1), 1 - self.label_smoothing)
        
        # Cross-entropy loss with label smoothing
        log_probs = F.log_softmax(outputs['logits'], dim=1)
        per_sample_loss = -(soft_labels * log_probs).sum(dim=1)
        
        # Apply class weights
        weighted_loss = per_sample_loss * self.class_weights[labels]
        loss = weighted_loss.mean()
        
        return loss, {
            'total_loss': loss.item(),
            'per_class_loss': self._compute_per_class_loss(per_sample_loss, labels)
        }
    
    def _compute_per_class_loss(self, per_sample_loss, labels):
        """Calculate per-class average loss."""
        per_class_loss = defaultdict(list)
        
        for loss_val, label in zip(per_sample_loss.detach().cpu().numpy(), 
                                  labels.cpu().numpy()):
            per_class_loss[int(label)].append(loss_val)
        
        return {k: np.mean(v) if v else 0.0 for k, v in per_class_loss.items()}
    
    def compute_metrics(self, outputs, labels):
        """
        Compute evaluation metrics.
        
        Args:
            outputs: Dictionary from classifier
            labels: True labels
            
        Returns:
            Dictionary of metrics
        """
        predictions = outputs['predictions']
        probs = outputs['probs']
        
        # Basic metrics
        accuracy = (predictions == labels).float().mean().item()
        
        # Top-k accuracy
        metrics = {
            'accuracy': accuracy,
            'top2_acc': self._compute_topk_accuracy(probs, labels, k=2),
            'top3_acc': self._compute_topk_accuracy(probs, labels, k=3) if self.num_classes > 2 else accuracy
        }
        
        return metrics
    
    def _compute_topk_accuracy(self, probs, labels, k):
        """Calculate top-k accuracy."""
        if k >= self.num_classes:
            return 1.0
        
        _, top_k = torch.topk(probs, k, dim=1)
        correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
        correct_k = correct.sum().float()
        return correct_k.item() / labels.size(0)
    
    def update_class_weights(self, new_weights):
        """Update class weights during training."""
        self.class_weights = torch.FloatTensor(new_weights).to(self.device)


def create_classifier_and_loss(num_classes, class_weights, config):
    """
    Create classifier and loss function.
    
    Args:
        num_classes: Number of classes
        class_weights: Class weights tensor
        config: PERISCAN configuration
        
    Returns:
        Tuple of (classifier, loss_function)
    """
    # Create classifier
    classifier = PERISCANClassifier(
        input_dim=config.fusion_output_dim,
        hidden_dim=config.fusion_hidden_dim,
        num_classes=num_classes,
        dropout=config.fusion_dropout
    )
    
    # Create loss function
    loss_fn = PERISCANLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        label_smoothing=0.1,
        device=config.device
    )
    
    print(f"Created classifier: {config.fusion_output_dim} -> {num_classes} classes")
    
    return classifier, loss_fn