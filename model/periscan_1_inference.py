"""
PERISCAN-I Inference Module
---------------------------
Attention-based deep learning framework for three-class disease-state
classification (Cancer / AD / HC) from PBMC single-cell RNA-seq data.

Input  : AnnData object with obs fields 'sample_id' and 'annotation'
Output : Per-sample prediction probabilities and class assignments
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Fixed parameters (derived from discovery cohort, Fold 1)
# ---------------------------------------------------------------------------
CELL_TYPES = [
    'CD14high_Monocyte',
    'CD4_Naive_CCR7',
    'CD56lowCD16high_NK',
    'CD8_Temra_FGFBP2',
    'NaiveB-TCL1A',
]

# Maximum cell count per cell type (median-based, discovery cohort)
MAX_CELLS_DICT = {
    'CD14high_Monocyte':  200,
    'CD4_Naive_CCR7':     600,
    'CD56lowCD16high_NK': 500,
    'CD8_Temra_FGFBP2':   300,
    'NaiveB-TCL1A':       100,
}

# Model hyperparameters
EMBED_DIM   = 64
HIDDEN_DIM  = 64
NUM_CLASSES = 3   # Cancer, AD, HC
DROPOUT     = 0.3

# Class index mapping
IDX_TO_LABEL = {0: 'AD', 1: 'CANCER', 2: 'HC'}


# ---------------------------------------------------------------------------
# Model: Attention Pooling
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Cell-type-specific attention pooling module.

    Projects each cell's gene expression vector into an embedding space,
    computes scalar attention scores via a two-layer MLP with tanh activation,
    and returns a weighted-sum population-level embedding.
    """

    def __init__(self, gene_dim: int, embed_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        # Gene encoder: linear projection + layer norm + activation + dropout
        self.gene_projection = nn.Sequential(
            nn.Linear(gene_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention scoring network
        self.attention_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters
        ----------
        x    : (batch, max_cells, gene_dim)
        mask : (batch, max_cells) boolean mask; True = valid cell

        Returns
        -------
        pooled           : (batch, embed_dim)
        attention_weights: (batch, max_cells, 1)
        """
        cell_embeddings = self.gene_projection(x)
        attention_scores = self.attention_score(cell_embeddings)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~mask.unsqueeze(-1), -1e9
            )

        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = (attention_weights * cell_embeddings).sum(dim=1)

        return pooled, attention_weights


# ---------------------------------------------------------------------------
# Model: PERISCAN-I
# ---------------------------------------------------------------------------
class PeriscanSimplified(nn.Module):
    """
    PERISCAN-I: three-class disease-state classifier (Cancer / AD / HC).

    Architecture
    ------------
    1. Gene encoder     – cell-type-specific linear projection (AttentionPooling.gene_encoder)
    2. Cell encoder     – attention-weighted aggregation       (AttentionPooling.attention_scorer)
    3. Feature fusion   – concatenation of five cell-type embeddings
    4. Classification   – fully connected layers with softmax output
    """

    def __init__(
        self,
        gene_dims: dict,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cell_types = CELL_TYPES
        self.embed_dim  = embed_dim

        # Cell-type-specific attention pooling layers (gene encoder + cell encoder)
        self.pooling_layers = nn.ModuleDict({
            ct: AttentionPooling(gene_dims[ct], embed_dim, dropout)
            for ct in self.cell_types
        })

        # Feature fusion + classification module
        fused_dim = len(self.cell_types) * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cells: dict, mask: dict):
        """
        Parameters
        ----------
        cells : dict[cell_type -> (batch, max_cells, gene_dim)]
        mask  : dict[cell_type -> (batch, max_cells) bool]

        Returns
        -------
        dict with keys: logits, cell_embeddings, attention_weights
        """
        cell_embeddings   = {}
        attention_weights = {}

        for ct in self.cell_types:
            pooled, attn = self.pooling_layers[ct](cells[ct], mask[ct])
            cell_embeddings[ct]   = pooled
            attention_weights[ct] = attn

        # Feature fusion: concatenate all cell-type embeddings
        fused_embedding = torch.cat(
            [cell_embeddings[ct] for ct in self.cell_types], dim=1
        )
        logits = self.classifier(fused_embedding)

        return {
            'logits':            logits,
            'cell_embeddings':   cell_embeddings,
            'attention_weights': attention_weights,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PBMCInferenceDataset(Dataset):
    """
    Dataset for PERISCAN-I inference (no labels required).

    Expects AnnData with:
        obs['sample_id']  : sample identifier
        obs['annotation'] : cell-type label (must match CELL_TYPES)
    """

    def __init__(self, adata, gene_lists: dict):
        self.adata   = adata.copy()
        self.samples = sorted(self.adata.obs['sample_id'].unique())

        # Pre-compute gene index positions for fast slicing
        var_names = list(adata.var_names)
        self.gene_indices = {
            ct: [var_names.index(g) for g in gene_lists[ct]]
            for ct in CELL_TYPES
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id      = self.samples[idx]
        cell_data_dict = {}
        cell_mask_dict = {}

        for ct in CELL_TYPES:
            # Select cells belonging to this sample and cell type
            selection = (
                (self.adata.obs['sample_id'] == sample_id) &
                (self.adata.obs['annotation'] == ct)
            )
            cells     = self.adata[selection]
            max_cells = MAX_CELLS_DICT[ct]
            n_genes   = len(self.gene_indices[ct])

            # Initialise padded tensors
            expression  = torch.zeros((max_cells, n_genes))
            mask_tensor = torch.zeros(max_cells, dtype=torch.bool)

            n_cells = len(cells)
            if n_cells > 0:
                # Extract expression matrix for selected genes
                if scipy.sparse.issparse(cells.X):
                    expr = cells.X[:, self.gene_indices[ct]].toarray()
                else:
                    expr = np.array(cells.X[:, self.gene_indices[ct]])
                expr = torch.FloatTensor(expr)

                # Random subsampling if cell count exceeds maximum
                if n_cells > max_cells:
                    generator = torch.Generator()
                    generator.manual_seed(SEED + hash(sample_id) % 100000)
                    indices = torch.randperm(n_cells, generator=generator)[:max_cells]
                    expr    = expr[indices]
                    n_cells = max_cells

                expression[:n_cells]  = expr
                mask_tensor[:n_cells] = True

            cell_data_dict[ct] = expression
            cell_mask_dict[ct] = mask_tensor

        return {
            'cell_data': cell_data_dict,
            'cell_mask': cell_mask_dict,
            'sample_id': sample_id,
        }


def collate_fn(batch: list) -> dict:
    """Collate a list of samples into batched tensors."""
    batch_size = len(batch)
    result = {
        'cells':      {},
        'mask':       {},
        'sample_ids': [x['sample_id'] for x in batch],
    }

    for ct in CELL_TYPES:
        max_cells = MAX_CELLS_DICT[ct]
        gene_dim  = batch[0]['cell_data'][ct].shape[1]

        cells = torch.zeros(batch_size, max_cells, gene_dim)
        mask  = torch.zeros(batch_size, max_cells, dtype=torch.bool)

        for i, item in enumerate(batch):
            cells[i] = item['cell_data'][ct]
            mask[i]  = item['cell_mask'][ct]

        result['cells'][ct] = cells
        result['mask'][ct]  = mask

    return result


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def run_inference(
    adata,
    gene_lists: dict,
    model_path: str,
    results_dir: str = '/results',
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Run PERISCAN-I inference on a PBMC AnnData object.

    Parameters
    ----------
    adata       : AnnData
                  Must contain obs['sample_id'] and obs['annotation'].
    gene_lists  : dict
                  Cell-type-specific feature gene sets loaded from
                  Periscan_1_genes.pkl.
    model_path  : str
                  Path to trained model checkpoint (.pt).
    results_dir : str
                  Directory for saving prediction output.
    batch_size  : int

    Returns
    -------
    results_df : pd.DataFrame
                 Columns: sample_id, prediction, prob_AD,
                          prob_CANCER, prob_HC, confidence
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialise model
    gene_dims = {ct: len(gene_lists[ct]) for ct in CELL_TYPES}
    model = PeriscanSimplified(
        gene_dims   = gene_dims,
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ).to(device)

    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Build DataLoader
    dataset = PBMCInferenceDataset(adata, gene_lists)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )

    # Run inference
    all_results = []
    with torch.no_grad():
        for batch in loader:
            cells = {ct: batch['cells'][ct].to(device) for ct in CELL_TYPES}
            mask  = {ct: batch['mask'][ct].to(device)  for ct in CELL_TYPES}

            outputs = model(cells, mask)
            probs   = F.softmax(outputs['logits'], dim=1).cpu().numpy()
            preds   = probs.argmax(axis=1)

            for i, sample_id in enumerate(batch['sample_ids']):
                all_results.append({
                    'sample_id':   sample_id,
                    'prediction':  IDX_TO_LABEL[preds[i]],
                    'prob_AD':     round(float(probs[i, 0]), 4),
                    'prob_CANCER': round(float(probs[i, 1]), 4),
                    'prob_HC':     round(float(probs[i, 2]), 4),
                    'confidence':  round(float(probs[i].max()), 4),
                })

    results_df = pd.DataFrame(all_results)

    # Save predictions
    out_path = os.path.join(results_dir, 'periscan_1_predictions.csv')
    results_df.to_csv(out_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("PERISCAN-I Inference Results")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nPrediction distribution:")
    print(results_df['prediction'].value_counts().to_string())
    print(f"\nResults saved to: {out_path}")
    print("=" * 60)

    return results_df
