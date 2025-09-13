"""
PERISCAN Configuration Module
Contains model hyperparameters and training configurations.
"""

import os
import json
import torch


class PERISCANConfig:
    """Configuration class for PERISCAN model training and inference."""
    
    def __init__(self, **kwargs):
        # Data parameters
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        self.train_split = 0.8  # 80% for training, 20% for validation

        # Gene Encoder parameters
        self.gene_hidden_dim = 256 
        self.gene_num_heads = 4
        self.gene_dropout = 0.4

        # Cell Encoder parameters
        self.cell_hidden_dim = 256
        self.cell_output_dim = 128
        self.cell_num_heads = 4
        self.cell_dropout = 0.3

        # Feature Fusion parameters
        self.fusion_hidden_dim = 128
        self.fusion_output_dim = 128
        self.fusion_num_heads = 4
        self.fusion_dropout = 0.3

        # Training parameters
        self.learning_rate = 3e-4
        self.weight_decay = 5e-3
        self.max_epochs = 100
        self.early_stop_patience = 20
        self.gradient_clip_norm = 0.7

        # Gene selection parameters
        self.min_genes_per_cell_type = 50
        self.max_genes_per_cell_type = 200
        self.gene_selection_seed = 42

        # Save parameters
        self.save_dir = './results'
        self.save_frequency = 10
        
        # Update with custom parameters
        self.__dict__.update(kwargs)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Save config
        self._save_config()
    
    def _save_config(self):
        """Save configuration to JSON file."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        config_path = os.path.join(self.save_dir, 'config.json')
        
        # Create serializable config dict
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not isinstance(v, torch.device)}
        config_dict['device'] = str(self.device)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def __repr__(self):
        """Print configuration information."""
        attrs = [f"{key}={value}" for key, value in self.__dict__.items() 
                if not key.startswith('_')]
        return f"PERISCANConfig(\n  " + "\n  ".join(attrs) + "\n)"


def get_default_config():
    """Get default PERISCAN configuration."""
    return PERISCANConfig()


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return PERISCANConfig(**config_dict)