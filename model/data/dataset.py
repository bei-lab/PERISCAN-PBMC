"""
PERISCAN Dataset Module
PyTorch Dataset and DataLoader utilities for single-cell data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy
from collections import defaultdict


class PERISCANDataset(Dataset):
    """PyTorch Dataset for PERISCAN single-cell data."""
    
    def __init__(self, adata, mode='train', max_cells_dict=None, gene_lists=None):
        """
        Initialize PERISCAN dataset.
        
        Args:
            adata: AnnData object
            mode: 'train' or 'val'
            max_cells_dict: Dictionary with max cells per cell type
            gene_lists: Dictionary with selected genes per cell type
        """
        # Filter data by mode
        self.adata = adata[adata.obs['dataset'] == mode].copy()
        self.mode = mode
        self.max_cells_dict = max_cells_dict
        self.gene_lists = gene_lists
        
        # Get unique samples and labels
        self.samples = self.adata.obs['pbmc_sample'].unique()
        self.labels = self.adata.obs.groupby('pbmc_sample')['cancertype'].first()
        
        # Create label to index mapping
        unique_cancers = sorted(self.labels.unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_cancers)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Get cell types
        self.cell_types = sorted(self.adata.obs['cell_type_merged'].unique())
        
        # Create gene indices mapping
        self.gene_indices = {}
        for cell_type in self.cell_types:
            self.gene_indices[cell_type] = [
                list(self.adata.var_names).index(gene) 
                for gene in self.gene_lists[cell_type]
                if gene in self.adata.var_names
            ]
        
        self._print_stats()
    
    def _print_stats(self):
        """Print dataset statistics."""
        print(f"\n{self.mode.upper()} Dataset Statistics:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Total cells: {self.adata.n_obs}")
        
        print("  Cancer type distribution:")
        for cancer_type in sorted(self.labels.unique()):
            count = sum(self.labels == cancer_type)
            print(f"    {cancer_type}: {count} samples")
        
        print("  Cell type distribution:")
        cell_dist = self.adata.obs['cell_type_merged'].value_counts()
        for cell_type in self.cell_types:
            print(f"    {cell_type}: {cell_dist.get(cell_type, 0)} cells")
        
        print("  Selected genes per cell type:")
        for cell_type in self.cell_types:
            print(f"    {cell_type}: {len(self.gene_indices[cell_type])} genes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with all cell types."""
        sample_id = self.samples[idx]
        
        # Initialize data containers
        cell_data_dict = {}
        cell_mask_dict = {}
        
        # Process each cell type
        for cell_type in self.cell_types:
            # Get cells of this type for this sample
            mask = (self.adata.obs['pbmc_sample'] == sample_id) & \
                   (self.adata.obs['cell_type_merged'] == cell_type)
            cells = self.adata[mask]
            
            # Get parameters
            max_cells = self.max_cells_dict[cell_type]
            gene_indices = self.gene_indices[cell_type]
            n_genes = len(gene_indices)
            
            # Initialize tensors
            expression = torch.zeros((max_cells, n_genes))
            mask_tensor = torch.zeros(max_cells, dtype=torch.bool)
            
            # Fill with actual data if cells exist
            n_cells = len(cells)
            if n_cells > 0:
                # Extract gene expression data
                if scipy.sparse.issparse(cells.X):
                    cell_data = cells.X[:, gene_indices].toarray()
                else:
                    cell_data = cells.X[:, gene_indices]
                
                cell_data = torch.FloatTensor(cell_data)
                
                # Subsample if too many cells
                if n_cells > max_cells:
                    indices = torch.randperm(n_cells)[:max_cells]
                    cell_data = cell_data[indices]
                    n_cells = max_cells
                
                # Fill tensors
                expression[:n_cells] = cell_data
                mask_tensor[:n_cells] = True
            
            cell_data_dict[cell_type] = expression
            cell_mask_dict[cell_type] = mask_tensor
        
        return {
            'cell_data': cell_data_dict,
            'cell_mask': cell_mask_dict,
            'label': self.label_to_idx[self.labels[sample_id]],
            'sample_id': sample_id
        }


def periscan_collate_fn(batch, max_cells_dict):
    """
    Custom collate function for PERISCAN data.
    
    Args:
        batch: List of samples from dataset
        max_cells_dict: Dictionary with max cells per cell type
    
    Returns:
        Batched data dictionary
    """
    batch_size = len(batch)
    cell_types = list(batch[0]['cell_data'].keys())
    
    # Initialize result dictionary
    result = {
        'cells': {},
        'mask': {},
        'label': torch.tensor([x['label'] for x in batch], dtype=torch.long),
        'sample_ids': [x['sample_id'] for x in batch]
    }
    
    # Process each cell type
    for cell_type in cell_types:
        max_cells = max_cells_dict[cell_type]
        gene_dim = batch[0]['cell_data'][cell_type].shape[1]
        
        # Initialize batch tensors
        cells = torch.zeros(batch_size, max_cells, gene_dim)
        mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
        
        # Fill batch tensors
        for i, item in enumerate(batch):
            cells[i] = item['cell_data'][cell_type]
            mask[i] = item['cell_mask'][cell_type]
        
        result['cells'][cell_type] = cells
        result['mask'][cell_type] = mask
    
    return result


def create_dataloaders(adata, gene_lists, max_cells_dict, config):
    """
    Create train and validation DataLoaders.
    
    Args:
        adata: AnnData object
        gene_lists: Dictionary with genes per cell type
        max_cells_dict: Dictionary with max cells per cell type
        config: PERISCAN configuration
    
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    print("Creating DataLoaders...")
    
    datasets = {}
    dataloaders = {}
    
    for mode in ['train', 'val']:
        # Create dataset
        datasets[mode] = PERISCANDataset(
            adata=adata,
            mode=mode,
            max_cells_dict=max_cells_dict,
            gene_lists=gene_lists
        )
        
        # Create dataloader
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=config.batch_size,
            shuffle=(mode == 'train'),
            num_workers=config.num_workers,
            collate_fn=lambda x: periscan_collate_fn(x, max_cells_dict),
            pin_memory=config.pin_memory,
            drop_last=False
        )
    
    return dataloaders, datasets


def get_class_weights(adata, device='cpu'):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        adata: AnnData object
        device: Device to place weights on
    
    Returns:
        Tensor with class weights
    """
    # Count samples per class
    class_counts = adata.obs.groupby('cancertype')['pbmc_sample'].nunique()
    
    # Calculate weights (inverse frequency)
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    weights = []
    
    for cancer_type in sorted(class_counts.index):
        weight = total_samples / (num_classes * class_counts[cancer_type])
        weights.append(weight)
    
    class_weights = torch.FloatTensor(weights).to(device)
    
    print("Class weights calculated:")
    for i, cancer_type in enumerate(sorted(class_counts.index)):
        print(f"  {cancer_type}: {class_weights[i]:.4f}")
    
    return class_weights


def check_dataloader(dataloader, max_batches=2):
    """
    Check DataLoader functionality and data shapes.
    
    Args:
        dataloader: DataLoader to check
        max_batches: Maximum number of batches to check
    """
    print(f"\nChecking {dataloader.dataset.mode} DataLoader...")
    print(f"  Total batches: {len(dataloader)}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
            
        print(f"\n  Batch {batch_idx + 1}:")
        print(f"    Batch size: {len(batch['sample_ids'])}")
        print(f"    Labels: {batch['label']}")
        
        for cell_type in batch['cells']:
            cells_shape = batch['cells'][cell_type].shape
            mask_shape = batch['mask'][cell_type].shape
            valid_cells = batch['mask'][cell_type].sum(dim=1)
            
            print(f"    {cell_type}:")
            print(f"      Cells shape: {cells_shape}")
            print(f"      Mask shape: {mask_shape}")
            print(f"      Valid cells per sample: {valid_cells.tolist()}")
    
    print("✓ DataLoader check completed")