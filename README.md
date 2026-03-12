# PERISCAN: Peripheral immune dynamics at single-cell resolution enable blood-based detection and classification of cancer

## Overview

PERISCAN is an attention-based deep learning framework that leverages peripheral immune dynamics for cancer detection and classification using single-cell RNA sequencing data from peripheral blood mononuclear cells (PBMCs). This approach exploits the systemic nature of cancer to detect disease through peripheral immune responses, offering a non-invasive alternative to traditional diagnostic methods.

### Key Features

- **High Sensitivity Cancer Detection**: Achieves >99% sensitivity for distinguishing cancer patients from healthy controls
- **Multi-Cancer Classification**: Performs tissue-of-origin prediction across 10 different cancer types with 75%+ accuracy  
- **Early Stage Detection**: Maintains high performance even for Stage I cancers where traditional biomarkers often fail
- **Mechanistic Insights**: Provides interpretable attention weights revealing immune reprogramming patterns
- **Clinical Applicability**: Requires only a standard blood draw, making it suitable for routine screening

### Scientific Innovation

The model addresses the fundamental challenge that cancer is a systemic disease affecting immune homeostasis throughout the body. By analyzing peripheral immune cell transcriptomes at single-cell resolution, PERISCAN captures:

1. **Immune Cell Reprogramming**: Changes in gene expression within individual immune cells
2. **Cellular Communication**: Altered interactions between different immune cell populations  
3. **System-wide Coordination**: Disrupted immune system organization in cancer patients

### Model Architecture

PERISCAN employs a hierarchical architecture processing information at three levels:

1. **Gene Encoder**: Transforms gene expression vectors within individual cells into unified embedding representations via cell-type-specific linear projections, layer normalization, and nonlinear activation
2. **Cell Encoder**: Aggregates cell-level embeddings within each immune cell type using attention-weighted pooling, producing a population-level representation
3. **Feature Fusion**: Concatenates cell-type embeddings into a sample-level representation and passes it through fully connected layers to generate class probabilities

## Reproducible Execution

A fully configured and executable environment is available on Code Ocean, including pre-processed example data, trained model weights, and the inference notebook:

🔗 https://codeocean.com/capsule/1340213/

## Data Requirements

Your input data should be an AnnData (`.h5ad`) file with the following structure:

**Required observation metadata (`adata.obs`)**:
- `sample_id`: Unique sample identifiers
- `annotation`: Cell type annotations

**Expected cell types**:
- `CD14high_Monocyte`: Classical monocytes
- `CD4_Naive_CCR7`: Naive CD4+ T cells  
- `CD56lowCD16high_NK`: Cytotoxic NK cells
- `CD8_Temra_FGFBP2`: Terminal effector memory CD8+ T cells
- `NaiveB-TCL1A`: Naive B cells

**Data preprocessing**:
- Gene expression should be log-normalized
- Minimum 3 cells per cell type per sample

## Data Availability


## Citation

If you use PERISCAN in your research, please cite:


---

**Repository**: https://github.com/bei-lab/PERISCAN-PBMC  
**License**: MIT  
**Contact**: Jin-Xin Bei — beijx@sysucc.org.cn
