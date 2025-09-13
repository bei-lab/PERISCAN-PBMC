"""
PERISCAN Data Preprocessing Module
Functions for data filtering, gene selection, and train/validation splitting.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def filter_low_quality_samples(adata, min_cells_per_type=10):
    """
    Filter out samples with insufficient cells for any cell type.
    
    Args:
        adata: AnnData object
        min_cells_per_type: Minimum number of cells required per cell type per sample
    
    Returns:
        Filtered AnnData object
    """
    print("Filtering low quality samples...")
    
    # Get cell type counts per sample
    cell_counts = pd.crosstab(adata.obs['pbmc_sample'], adata.obs['cell_type_merged'])
    
    # Find samples with insufficient cells
    samples_to_remove = []
    for sample in cell_counts.index:
        if (cell_counts.loc[sample] < min_cells_per_type).any():
            samples_to_remove.append(sample)
    
    if samples_to_remove:
        print(f"Removing {len(samples_to_remove)} samples with insufficient cells")
        print(f"Samples removed: {samples_to_remove}")
    
    # Filter adata
    samples_to_keep = ~adata.obs['pbmc_sample'].isin(samples_to_remove)
    adata_filtered = adata[samples_to_keep].copy()
    
    print(f"Samples after filtering: {len(adata_filtered.obs['pbmc_sample'].unique())}")
    return adata_filtered


def create_cancer_labels(adata):
    """
    Create simplified cancer type labels (CANCER, AD, HC).
    
    Args:
        adata: AnnData object
    
    Returns:
        AnnData object with updated cancer labels
    """
    print("Creating cancer type labels...")
    
    # Keep original labels
    adata.obs['original_cancertype'] = adata.obs['cancertype'].copy()
    
    # Create mapping
    cancer_types = ['CESC', 'COAD', 'HNSC', 'LIHC', 'LUCA', 'NPC', 'STAD', 'PAAD', 'ESCA', 'BLCA']
    mapping_dict = {'HC': 'HC', 'AD': 'AD'}
    
    for cancer_type in cancer_types:
        mapping_dict[cancer_type] = 'CANCER'
    
    # Apply mapping
    adata.obs['cancertype'] = adata.obs['cancertype'].map(mapping_dict)
    
    # Print distribution
    print("Cancer type distribution after mapping:")
    print(adata.obs.groupby('cancertype')['pbmc_sample'].nunique())
    
    return adata


def split_train_validation(adata, train_ratio=0.8, random_state=42):
    """
    Split data into training and validation sets.
    
    Args:
        adata: AnnData object
        train_ratio: Ratio of data for training
        random_state: Random seed
    
    Returns:
        AnnData object with dataset split labels
    """
    print(f"Splitting data into train ({train_ratio:.0%}) and validation ({1-train_ratio:.0%}) sets...")
    
    # Get sample info
    sample_info = adata.obs[['pbmc_sample', 'cancertype']].drop_duplicates()
    
    # Split by cancer type to maintain balanced distribution
    train_samples = []
    val_samples = []
    
    for cancer_type in sample_info['cancertype'].unique():
        cancer_samples = sample_info[sample_info['cancertype'] == cancer_type]['pbmc_sample'].values
        
        if len(cancer_samples) == 1:
            # If only one sample, put it in training set
            train_samples.extend(cancer_samples)
        else:
            # Split samples for this cancer type
            train_cancer, val_cancer = train_test_split(
                cancer_samples, 
                train_size=train_ratio, 
                random_state=random_state,
                stratify=None  # No stratification needed within cancer type
            )
            train_samples.extend(train_cancer)
            val_samples.extend(val_cancer)
    
    # Create dataset labels
    adata.obs['dataset'] = 'train'
    adata.obs.loc[adata.obs['pbmc_sample'].isin(val_samples), 'dataset'] = 'val'
    
    # Print split statistics
    print("\nDataset split summary:")
    split_stats = pd.crosstab(
        adata.obs.groupby('pbmc_sample')['cancertype'].first(),
        adata.obs.groupby('pbmc_sample')['dataset'].first()
    )
    print(split_stats)
    
    print(f"\nTotal samples - Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    return adata


def select_random_genes(adata, cell_types, min_genes=50, max_genes=200, random_state=42):
    """
    Randomly select genes for each cell type from available genes.
    
    Args:
        adata: AnnData object
        cell_types: List of cell types to select genes for
        min_genes: Minimum number of genes per cell type
        max_genes: Maximum number of genes per cell type  
        random_state: Random seed
    
    Returns:
        Dictionary with selected genes for each cell type
    """
    print("Selecting random genes for each cell type...")
    
    np.random.seed(random_state)
    gene_lists = {}
    
    # Get available genes (expressed in at least 20% of cells)
    expressed_genes = []
    for gene in adata.var_names:
        gene_data = adata[:, gene].X
        if hasattr(gene_data, 'toarray'):
            gene_data = gene_data.toarray().flatten()
        else:
            gene_data = gene_data.flatten()
        
        # Check if gene is expressed in at least 20% of cells
        expression_rate = np.mean(gene_data > 0)
        if expression_rate >= 0.2:
            expressed_genes.append(gene)
    
    print(f"Found {len(expressed_genes)} genes expressed in ≥20% of cells")
    
    for cell_type in cell_types:
        # Random number of genes between min and max
        n_genes = np.random.randint(min_genes, max_genes + 1)
        
        # Randomly select genes
        if len(expressed_genes) >= n_genes:
            selected_genes = np.random.choice(expressed_genes, size=n_genes, replace=False)
            gene_lists[cell_type] = list(selected_genes)
        else:
            # If not enough genes, use all available genes
            gene_lists[cell_type] = expressed_genes.copy()
        
        print(f"{cell_type}: {len(gene_lists[cell_type])} genes selected")
    
    return gene_lists


def get_cell_type_params(adata, cell_types):
    """
    Calculate max cells parameters for each cell type.
    
    Args:
        adata: AnnData object
        cell_types: List of cell types
    
    Returns:
        Dictionary with max cells for each cell type
    """
    max_cells_dict = {}
    
    for cell_type in cell_types:
        # Calculate median cell count for this cell type across samples
        cell_counts = adata[adata.obs['cell_type_merged'] == cell_type].obs['pbmc_sample'].value_counts()
        median_count = cell_counts.median()
        
        # Round down to nearest hundred
        max_cells = int(median_count // 100 * 100)
        max_cells = max(max_cells, 100)  # Minimum 100 cells
        max_cells_dict[cell_type] = max_cells
        
        print(f"{cell_type}: max_cells = {max_cells} (median = {median_count:.0f})")
    
    return max_cells_dict


def preprocess_adata(adata_path, config):
    """
    Main preprocessing pipeline for PERISCAN.
    
    Args:
        adata_path: Path to h5ad file
        config: PERISCAN configuration object
    
    Returns:
        Tuple of (processed_adata, gene_lists, max_cells_dict)
    """
    print("=== Starting PERISCAN Data Preprocessing ===")
    
    # Load data
    print(f"Loading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes, {len(adata.obs['pbmc_sample'].unique())} samples")
    
    # Filter low quality samples
    adata = filter_low_quality_samples(adata, min_cells_per_type=10)
    
    # Create cancer type labels
    adata = create_cancer_labels(adata)
    
    # Split train/validation
    adata = split_train_validation(adata, train_ratio=config.train_split, random_state=42)
    
    # Get cell types
    cell_types = sorted(adata.obs['cell_type_merged'].unique())
    print(f"Cell types found: {cell_types}")
    
    # Select random genes
    gene_lists = select_random_genes(
        adata, 
        cell_types,
        min_genes=config.min_genes_per_cell_type,
        max_genes=config.max_genes_per_cell_type,
        random_state=config.gene_selection_seed
    )
    
    # Calculate max cells parameters
    max_cells_dict = get_cell_type_params(adata, cell_types)
    
    print("\n=== Preprocessing Complete ===")
    print(f"Final dataset: {adata.n_obs} cells, {len(adata.obs['pbmc_sample'].unique())} samples")
    print(f"Cell types: {len(cell_types)}")
    print(f"Genes selected per cell type: {[len(genes) for genes in gene_lists.values()]}")
    
    return adata, gene_lists, max_cells_dict


def check_data_integrity(adata, gene_lists):
    """
    Check data integrity after preprocessing.
    
    Args:
        adata: Processed AnnData object
        gene_lists: Dictionary of selected genes per cell type
    """
    print("\n=== Data Integrity Check ===")
    
    # Check if all selected genes exist in adata
    all_genes = set(adata.var_names)
    for cell_type, genes in gene_lists.items():
        missing_genes = set(genes) - all_genes
        if missing_genes:
            print(f"WARNING: {cell_type} has {len(missing_genes)} missing genes")
        else:
            print(f"✓ {cell_type}: All {len(genes)} genes found in data")
    
    # Check cell counts per sample per cell type
    print("\nCell count statistics per cell type:")
    for cell_type in gene_lists.keys():
        cell_counts = adata[adata.obs['cell_type_merged'] == cell_type].obs['pbmc_sample'].value_counts()
        print(f"{cell_type}: min={cell_counts.min()}, median={cell_counts.median():.0f}, max={cell_counts.max()}")
    
    # Check train/val split
    train_samples = len(adata.obs[adata.obs['dataset'] == 'train']['pbmc_sample'].unique())
    val_samples = len(adata.obs[adata.obs['dataset'] == 'val']['pbmc_sample'].unique())
    print(f"\nDataset split: {train_samples} train samples, {val_samples} val samples")
    
    print("✓ Data integrity check completed")