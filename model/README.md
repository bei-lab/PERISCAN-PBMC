# PERISCAN: Peripheral Immune Dynamics for Cancer Detection

A transformer-based deep learning framework for cancer detection and classification using single-cell RNA sequencing data from peripheral blood mononuclear cells (PBMCs).

## Overview

PERISCAN leverages peripheral immune dynamics at single-cell resolution to enable:
- **Cancer Detection**: Distinguish cancer patients from healthy controls and autoimmune disease patients
- **Cancer Classification**: Identify tissue-of-origin for different cancer types
- **Early Detection**: Particularly effective for early-stage cancers

The model employs a hierarchical transformer architecture that processes gene expression at multiple levels:
1. **Gene-level**: Within-cell gene expression patterns
2. **Cell-level**: Inter-cellular relationships within cell types  
3. **Population-level**: Cross-cell-type immune coordination


## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bei-lab/PERISCAN-PBMC.git
cd PERISCAN
```

2. Create a virtual environment:
```bash
python -m venv periscan_env
source periscan_env/bin/activate  # On Windows: periscan_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run PERISCAN on your single-cell data:

```bash
python main.py --data_path your_data.h5ad --output_dir ./results
```

### Command Line Options

```bash
python main.py \
    --data_path your_data.h5ad \
    --output_dir ./results \
    --batch_size 32 \
    --max_epochs 100 \
    --learning_rate 3e-4 \
    --gpu
```

### Key Parameters

- `--data_path`: Path to input h5ad file with single-cell data
- `--output_dir`: Directory to save results and checkpoints  
- `--batch_size`: Training batch size (default: 32)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--min_genes`/`--max_genes`: Gene selection range per cell type (default: 50-200)
- `--gpu`: Use GPU acceleration if available
- `--train_only`: Only train model, skip evaluation
- `--eval_only checkpoint.pt`: Only evaluate using existing checkpoint

## Data Format

Your input data should be an AnnData object (.h5ad file) with:

### Required columns in `obs`:
- `pbmc_sample`: Sample identifiers
- `cancertype`: Cancer type labels (e.g., 'LUCA', 'PAAD', 'HC', 'AD')
- `cell_type_merged`: Cell type annotations (e.g., 'CD4_Naive_CCR7', 'CD14high_Monocyte')

### Expected cell types:
- `CD14high_Monocyte`
- `CD4_Naive_CCR7`  
- `CD56lowCD16high_NK`
- `CD8_Temra_FGFBP2`
- `NaiveB-TCL1A`

### Preprocessing
The pipeline automatically:
- Filters low-quality samples
- Randomly selects genes per cell type
- Splits data into train/validation sets
- Creates class labels (CANCER, AD, HC)

## Project Structure

```
PERISCAN/
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── main.py                   # Main execution script
├── train.py                  # Training pipeline
├── evaluate.py               # Model evaluation
├── data/
│   ├── preprocessing.py      # Data preprocessing
│   └── dataset.py           # PyTorch dataset
├── models/
│   ├── gene_encoder.py      # Gene-level transformer
│   ├── cell_encoder.py      # Cell-level aggregation
│   ├── feature_fusion.py    # Cross-cell-type fusion
│   └── classifier.py        # Classification head
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Model Architecture

### Gene Encoder
- Multi-head self-attention for gene-gene interactions
- Processes expression patterns within individual cells
- Outputs gene-level representations

### Cell Encoder  
- Aggregates gene features to cell-level representations
- Models inter-cellular relationships within cell types
- Handles variable cell numbers with masking

### Feature Fusion
- Cross-attention across different cell types
- Integrates immune system-wide responses
- Generates sample-level representations

### Classifier
- Final prediction head with class balancing
- Supports both binary and multi-class classification
- Includes temperature scaling for calibration

## Output Files

After training and evaluation, the following files are generated:

### Checkpoints
- `checkpoints/best_model.pt`: Best performing model
- `checkpoints/checkpoint_epoch_X.pt`: Regular checkpoints

### Results
- `training_history.json`: Training metrics over time
- `config.json`: Model configuration
- `val_predictions.csv`: Detailed predictions
- `val_metrics.csv`: Performance metrics per class

### Visualizations
- `confusion_matrix.png`: Confusion matrix heatmap
- `training_history.png`: Loss and accuracy curves

## Performance Evaluation

The model provides comprehensive evaluation including:
- **Accuracy**: Overall classification accuracy
- **Per-class Metrics**: Precision, recall, F1-score for each class
- **Confusion Matrix**: Detailed classification breakdown
- **Confidence Analysis**: Prediction confidence scores
- **Misclassification Analysis**: Analysis of incorrect predictions

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```python
from config import PERISCANConfig

config = PERISCANConfig(
    batch_size=64,
    max_epochs=200,
    learning_rate=1e-4,
    gene_hidden_dim=512,
    early_stop_patience=30
)
```

### Training Only Mode

Train model without evaluation:

```bash
python main.py --data_path data.h5ad --train_only --max_epochs 50
```

### Evaluation Only Mode

Evaluate existing checkpoint:

```bash
python main.py --data_path data.h5ad --eval_only checkpoints/best_model.pt
```

### GPU Training

Enable GPU acceleration:

```bash
python main.py --data_path data.h5ad --gpu --batch_size 64
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 8`
   - Use CPU: Remove `--gpu` flag

2. **Low Performance**
   - Check data quality and class balance
   - Increase model capacity: `--gene_hidden_dim 256`
   - Adjust gene selection range: `--min_genes 100 --max_genes 300`

3. **Missing Cell Types**
   - Ensure all required cell types are present in data
   - Check cell type annotations match expected names

4. **Memory Issues**
   - Reduce number of workers: `--num_workers 2`
   - Use smaller gene selection range

### Data Requirements

- Minimum 3 cells per cell type per sample
- At least 20 samples for training
- Gene expression should be log-normalized
- Cell type annotations must be consistent

## Citation

If you use PERISCAN in your research, please cite:


## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with:
   - Error message (if any)
   - System information
   - Steps to reproduce
   - Sample data

## Acknowledgments

- Built on PyTorch and Scanpy frameworks
- Inspired by transformer architectures for biological data
- Thanks to the single-cell genomics community for data standards and best practices