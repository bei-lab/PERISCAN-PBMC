# PERISCAN: Peripheral immune dynamics at single-cell resolution enable blood-based detection and classification of cancer

## Overview

PERISCAN is a transformer-based deep learning framework that leverages peripheral immune dynamics for cancer detection and classification using single-cell RNA sequencing data from peripheral blood mononuclear cells (PBMCs). This approach exploits the systemic nature of cancer to detect disease through peripheral immune responses, offering a non-invasive alternative to traditional diagnostic methods.

### Key Features

- **High Sensitivity Cancer Detection**: Achieves >99% sensitivity for distinguishing cancer patients from healthy controls
- **Multi-Cancer Classification**: Performs tissue-of-origin prediction across 10 different cancer types with 85%+ accuracy  
- **Early Stage Detection**: Maintains high performance even for Stage I cancers where traditional biomarkers often fail
- **Mechanistic Insights**: Provides interpretable attention maps revealing immune reprogramming patterns
- **Clinical Applicability**: Requires only a standard blood draw, making it suitable for routine screening

### Scientific Innovation

The model addresses the fundamental challenge that cancer is a systemic disease affecting immune homeostasis throughout the body. By analyzing peripheral immune cell transcriptomes at single-cell resolution, PERISCAN captures:

1. **Immune Cell Reprogramming**: Changes in gene expression within individual immune cells
2. **Cellular Communication**: Altered interactions between different immune cell populations  
3. **System-wide Coordination**: Disrupted immune system organization in cancer patients

### Model Architecture

PERISCAN employs a hierarchical transformer architecture processing information at three levels:

1. **Gene Encoder**: Processes gene expression patterns within individual cells using multi-head self-attention
2. **Cell Encoder**: Aggregates gene-level features and models inter-cellular relationships within cell types
3. **Feature Fusion**: Integrates information across different immune cell types to capture system-wide immune responses

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU
- Minimum 40GB RAM (80GB recommended for large datasets)

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/bei-lab/PERISCAN-PBMC.git
cd PERISCAN
```

2. **Create and activate virtual environment**:
```bash
# Using conda (recommended)
conda create -n periscan python=3.9
conda activate periscan


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```

### GPU Setup

If you have a CUDA-capable GPU:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install GPU-optimized PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Quick Start

**Basic cancer detection**:
```bash
python main.py --data_path your_pbmc_data.h5ad --output_dir ./results --gpu --batch_size 64
```

**Training with custom parameters**:
```bash
python main.py \
    --data_path your_pbmc_data.h5ad \
    --output_dir ./results \
    --max_epochs 200 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --min_genes 100 \
    --max_genes 300 \
    --gpu
```

### Data Requirements

Your input data should be an AnnData (.h5ad) file with the following structure:

**Required observation metadata (`adata.obs`)**:
- `pbmc_sample`: Unique sample identifiers
- `cancertype`: Cancer type labels (e.g., 'LUCA', 'PAAD', 'BLCA', 'HC', 'AD')
- `cell_type_merged`: Cell type annotations

**Expected cell types**:
- `CD14high_Monocyte`: Classical monocytes
- `CD4_Naive_CCR7`: Naive CD4+ T cells  
- `CD56lowCD16high_NK`: Cytotoxic NK cells
- `CD8_Temra_FGFBP2`: Terminal effector memory CD8+ T cells
- `NaiveB-TCL1A`: Naive B cells

**Data preprocessing**:
- Gene expression should be log-normalized
- Minimum 3 cells per cell type per sample

### Training Modes

**Full pipeline** (default):
```bash
python main.py --data_path data.h5ad --output_dir ./results
```

**Training only**:
```bash
python main.py --data_path data.h5ad --output_dir ./results --train_only
```

**Evaluation only**:
```bash
python main.py --data_path data.h5ad --eval_only ./results/checkpoints/best_model.pt
```

### Output Structure

After successful execution, the output directory contains:

```
results/
├── config.json                    # Model configuration
├── training_history.json          # Training metrics
├── checkpoints/
│   ├── best_model.pt              # Best performing model
│   └── checkpoint_epoch_*.pt      # Regular checkpoints
├── data/
│   ├── val_predictions.csv        # Detailed predictions
│   ├── val_metrics.csv            # Performance metrics
│   └── misclassification_analysis.csv
├── visualizations/
│   ├── confusion_matrix.png       # Classification results
│   └── training_history.png       # Training curves
```

### Performance Interpretation

**Key metrics provided**:
- **Sensitivity/Recall**: Ability to correctly identify cancer cases
- **Specificity**: Ability to correctly identify non-cancer cases  
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve for multi-class classification
- **Confidence Scores**: Model certainty for each prediction


### Advanced Configuration

Create custom configurations programmatically:

```python
from config import PERISCANConfig

### Troubleshooting

**Common issues and solutions**:

1. **Memory errors**:
   - Reduce batch size: `--batch_size 16`
   - Use CPU mode: remove `--gpu`
   - Reduce gene selection range

2. **Poor performance**:
   - Check class balance in your dataset
   - Verify cell type annotations
   - Increase model capacity: `--gene_hidden_dim 512`

3. **Data format issues**:
   - Ensure all required metadata columns exist
   - Check cell type naming consistency
   - Verify minimum cell counts per sample

**Getting help**:
- Check existing issues on GitHub
- Refer to the detailed README in the repository
- Contact the development team

## Citation

This work is currently under peer review. Once published, please cite:

```
[to be added]
```

For now, if you use this code in your research, please reference this GitHub repository and consider citing related work on single-cell immune profiling and transformer architectures for biological data.

---

**Repository**: https://github.com/bei-lab/PERISCAN-PBMC  
**License**: MIT  
**Maintainers**: [To be added]  
**Contact**: [Contact information to be added]
