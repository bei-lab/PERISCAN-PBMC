"""
PERISCAN Data Package
Data preprocessing and dataset utilities for single-cell cancer detection.
"""

from .preprocessing import (
    preprocess_adata,
    filter_low_quality_samples,
    create_cancer_labels,
    split_train_validation,
    select_random_genes,
    get_cell_type_params,
    check_data_integrity
)

from .dataset import (
    PERISCANDataset,
    periscan_collate_fn,
    create_dataloaders,
    get_class_weights,
    check_dataloader
)

__all__ = [
    # Preprocessing functions
    'preprocess_adata',
    'filter_low_quality_samples', 
    'create_cancer_labels',
    'split_train_validation',
    'select_random_genes',
    'get_cell_type_params',
    'check_data_integrity',
    
    # Dataset functions
    'PERISCANDataset',
    'periscan_collate_fn',
    'create_dataloaders',
    'get_class_weights',
    'check_dataloader'
]