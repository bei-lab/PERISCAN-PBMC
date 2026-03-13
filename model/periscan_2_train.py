"""
PERISCAN-II Training Script
----------------------------
Attention-based deep learning framework for ten-class cancer
tissue-of-origin classification from PBMC single-cell RNA-seq data.

NOTE: This script is provided for reproducibility and transparency.
Full training requires access to the discovery cohort dataset and is not intended to be executed
within this capsule environment. For inference on the provided
validation cohort, please run the accompanying notebook using
periscan_2_inference.py.

Training was performed on a multi-GPU high-performance computing cluster.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import math
import pickle
import warnings
import random
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import scanpy as sc

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
CELL_TYPES = [
    'CD14high_Monocyte',
    'CD4_Naive_CCR7',
    'CD56lowCD16high_NK',
    'CD8_Temra_FGFBP2',
    'NaiveB-TCL1A',
]

CANCER_TYPES = sorted([
    'BLCA', 'CESC', 'COAD', 'ESCA', 'HNSC',
    'LIHC', 'LUCA', 'NPC',  'PAAD', 'STAD',
])

# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------
def preprocess_adata(adata):
    """
    Prepare AnnData for PERISCAN-II training.

    Retains only cancer samples; disease label is the cancer subtype
    stored in obs['cancertype'].

    Parameters
    ----------
    adata : AnnData
        Discovery cohort AnnData. Expected obs fields:
        'pbmc_sample', 'cancertype', 'cell_type_merged'.

    Returns
    -------
    adata : AnnData containing cancer samples only
    """
    adata = adata[adata.obs['cancertype'].isin(CANCER_TYPES)].copy()
    return adata


def compute_max_cells(adata):
    """
    Compute per-cell-type cell count cap as the median count
    rounded down to the nearest 100.

    Parameters
    ----------
    adata : AnnData

    Returns
    -------
    max_cells_dict : dict[str, int]
    """
    max_cells_dict = {}
    for ct in CELL_TYPES:
        counts = (
            adata[adata.obs['cell_type_merged'] == ct]
            .obs['pbmc_sample']
            .value_counts()
        )
        max_cells_dict[ct] = int(counts.median() // 100 * 100)
    return max_cells_dict


# ---------------------------------------------------------------------------
# Cross-validation split
# ---------------------------------------------------------------------------
def split_data_balanced_cv(adata, n_folds=5, random_state=42):
    """
    Stratified k-fold cross-validation split at the sample level.

    Each cancer subtype is split independently to ensure balanced
    representation across folds. Results are stored as 'dataset_1'
    ... 'dataset_k' columns in adata.obs, where each column marks
    samples as 'train' or 'val' for that fold.

    Parameters
    ----------
    adata        : AnnData
    n_folds      : int
    random_state : int

    Returns
    -------
    adata : AnnData with added dataset_* columns
    """
    sample_info  = adata.obs[['pbmc_sample', 'cancertype']].drop_duplicates()
    fold_samples = {f'fold_{i+1}': [] for i in range(n_folds)}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for cancer in CANCER_TYPES:
        samples = sample_info[
            sample_info['cancertype'] == cancer
        ]['pbmc_sample'].values
        if len(samples) == 0:
            continue
        if len(samples) >= n_folds:
            for i, (_, val_idx) in enumerate(kf.split(samples)):
                fold_samples[f'fold_{i+1}'].extend(samples[val_idx])
        else:
            np.random.seed(random_state)
            np.random.shuffle(samples)
            for i in range(min(n_folds, len(samples))):
                fold_samples[f'fold_{i+1}'].append(samples[i])

    for i in range(n_folds):
        col = f'dataset_{i+1}'
        adata.obs[col] = 'train'
        adata.obs.loc[
            adata.obs['pbmc_sample'].isin(fold_samples[f'fold_{i+1}']), col
        ] = 'val'

    return adata


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PBMCTrainDataset(Dataset):
    """
    Dataset for PERISCAN-II training and validation.

    Loads cancer cells from a single fold ('train' or 'val') of the
    discovery cohort and returns padded cell-by-gene tensors with
    binary masks. Labels are cancer subtype indices (0–9, alphabetical).
    """

    def __init__(self, adata, mode: str, max_cells_dict: dict, gene_lists: dict):
        self.adata          = adata[adata.obs['dataset'] == mode].copy()
        self.max_cells_dict = max_cells_dict

        self.samples = self.adata.obs['pbmc_sample'].unique()
        self.labels  = self.adata.obs.groupby('pbmc_sample')['cancertype'].first()

        self.label_to_idx = {lbl: i for i, lbl in enumerate(CANCER_TYPES)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

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
            selection = (
                (self.adata.obs['pbmc_sample'] == sample_id) &
                (self.adata.obs['cell_type_merged'] == ct)
            )
            cells     = self.adata[selection]
            max_cells = self.max_cells_dict[ct]
            n_genes   = len(self.gene_indices[ct])

            expression  = torch.zeros((max_cells, n_genes))
            mask_tensor = torch.zeros(max_cells, dtype=torch.bool)

            n_cells = len(cells)
            if n_cells > 0:
                if scipy.sparse.issparse(cells.X):
                    expr = cells.X[:, self.gene_indices[ct]].toarray()
                else:
                    expr = np.array(cells.X[:, self.gene_indices[ct]])
                expr = torch.FloatTensor(expr)

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
            'label':     self.label_to_idx[self.labels[sample_id]],
            'sample_id': sample_id,
        }


def collate_fn(batch: list, max_cells_dict: dict) -> dict:
    """Collate a list of samples into batched tensors."""
    batch_size = len(batch)
    result = {
        'cells':      {},
        'mask':       {},
        'label':      torch.tensor([x['label'] for x in batch], dtype=torch.long),
        'sample_ids': [x['sample_id'] for x in batch],
    }

    for ct in CELL_TYPES:
        max_cells = max_cells_dict[ct]
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

        self.gene_projection = nn.Sequential(
            nn.Linear(gene_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        cell_embeddings  = self.gene_projection(x)
        attention_scores = self.attention_score(cell_embeddings)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~mask.unsqueeze(-1), -1e9
            )

        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = (attention_weights * cell_embeddings).sum(dim=1)

        return pooled, attention_weights


# ---------------------------------------------------------------------------
# Model: PERISCAN-II
# ---------------------------------------------------------------------------
class PeriscanSimplified(nn.Module):
    """
    PERISCAN-II: ten-class cancer tissue-of-origin classifier.

    Architecture
    ------------
    1. Gene encoder     - cell-type-specific linear projection
    2. Cell encoder     - attention-weighted aggregation
    3. Feature fusion   - concatenation of five cell-type embeddings
    4. Classification   - fully connected layers with softmax output
    """

    def __init__(
        self,
        gene_dims: dict,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cell_types = CELL_TYPES
        self.embed_dim  = embed_dim

        self.pooling_layers = nn.ModuleDict({
            ct: AttentionPooling(gene_dims[ct], embed_dim, dropout)
            for ct in self.cell_types
        })

        fused_dim = len(self.cell_types) * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cells: dict, mask: dict):
        cell_embeddings   = {}
        attention_weights = {}

        for ct in self.cell_types:
            pooled, attn = self.pooling_layers[ct](cells[ct], mask[ct])
            cell_embeddings[ct]   = pooled
            attention_weights[ct] = attn

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
# Training and validation utilities
# ---------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device, idx_to_label):
    """Run one training epoch; return loss, accuracy, and per-class metrics."""
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(dataloader, desc='Training', leave=False):
        cells  = {ct: batch['cells'][ct].to(device) for ct in model.cell_types}
        mask   = {ct: batch['mask'][ct].to(device)  for ct in model.cell_types}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(cells, mask)
        loss    = criterion(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs['logits'].argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss    = total_loss / len(dataloader)
    overall_acc = accuracy_score(all_labels, all_preds)
    cm          = confusion_matrix(all_labels, all_preds,
                                   labels=list(range(len(idx_to_label))))
    class_acc   = {
        idx_to_label[i]: cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        for i in range(len(idx_to_label))
    }

    return avg_loss, overall_acc, class_acc


def validate(model, dataloader, criterion, device, idx_to_label,
             return_details=False):
    """Run validation; optionally return full prediction details."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    all_sample_ids, all_probs         = [], []
    all_attn  = {ct: [] for ct in model.cell_types}
    all_embed = {ct: [] for ct in model.cell_types}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            cells  = {ct: batch['cells'][ct].to(device) for ct in model.cell_types}
            mask   = {ct: batch['mask'][ct].to(device)  for ct in model.cell_types}
            labels = batch['label'].to(device)

            outputs = model(cells, mask)
            loss    = criterion(outputs['logits'], labels)

            total_loss += loss.item()
            all_preds.extend(outputs['logits'].argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sample_ids.extend(batch['sample_ids'])
            all_probs.append(F.softmax(outputs['logits'], dim=1).cpu())

            if return_details:
                for ct in model.cell_types:
                    all_attn[ct].append(outputs['attention_weights'][ct].cpu())
                    all_embed[ct].append(outputs['cell_embeddings'][ct].cpu())

    avg_loss    = total_loss / len(dataloader)
    overall_acc = accuracy_score(all_labels, all_preds)
    cm          = confusion_matrix(all_labels, all_preds,
                                   labels=list(range(len(idx_to_label))))
    class_acc   = {
        idx_to_label[i]: cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        for i in range(len(idx_to_label))
    }

    if not return_details:
        return avg_loss, overall_acc, class_acc

    all_probs = torch.cat(all_probs, dim=0).numpy()
    for ct in model.cell_types:
        all_attn[ct]  = torch.cat(all_attn[ct],  dim=0)
        all_embed[ct] = torch.cat(all_embed[ct], dim=0)

    return avg_loss, overall_acc, class_acc, {
        'predictions':      np.array(all_preds),
        'labels':           np.array(all_labels),
        'sample_ids':       all_sample_ids,
        'probabilities':    all_probs,
        'attention_weights': all_attn,
        'embeddings':        all_embed,
        'confusion_matrix':  cm,
    }


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(
    adata,
    gene_lists:      dict,
    max_cells_dict:  dict,
    fold:            int,
    batch_size:      int   = 128,
    num_workers:     int   = 8,
    embed_dim:       int   = 128,
    hidden_dim:      int   = 128,
    dropout:         float = 0.5,
    num_epochs:      int   = 800,
    learning_rate:   float = 1e-4,
    weight_decay:    float = 0.01,
    use_scheduler:   bool  = False,
    warmup_epochs:   int   = 10,
    manual_class_weights: dict = None,
    save_dir:        str   = './results',
    save_prefix:     str   = 'periscan_2',
):
    """
    Train PERISCAN-II for one cross-validation fold.

    Parameters
    ----------
    adata            : AnnData, cancer-only discovery cohort with
                       dataset_* fold columns added by split_data_balanced_cv.
    gene_lists       : dict, cell-type-specific feature gene sets.
    max_cells_dict   : dict, per-cell-type cell count caps.
    fold             : int, which fold to use as validation set.
    manual_class_weights : dict, optional per-cancer loss weights
                       (cancer_type -> float). Uniform weights used if None.
    ...              : see function signature for all hyperparameters.

    Returns
    -------
    model   : trained PeriscanSimplified
    results : dict with predictions, metrics, and training history
    history : dict with per-epoch loss and accuracy logs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(save_dir, exist_ok=True)

    # Assign fold column
    adata.obs['dataset'] = adata.obs[f'dataset_{fold}']

    # Build datasets and loaders
    train_dataset = PBMCTrainDataset(adata, 'train', max_cells_dict, gene_lists)
    val_dataset   = PBMCTrainDataset(adata, 'val',   max_cells_dict, gene_lists)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, max_cells_dict),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, max_cells_dict),
    )

    # Class weights
    if manual_class_weights is not None:
        weights = torch.ones(len(CANCER_TYPES))
        for cancer, w in manual_class_weights.items():
            weights[train_dataset.label_to_idx[cancer]] = w
    else:
        weights = torch.ones(len(CANCER_TYPES))
    class_weights = weights.to(device)

    # Initialise model
    gene_dims = {ct: len(gene_lists[ct]) for ct in CELL_TYPES}
    model = PeriscanSimplified(
        gene_dims   = gene_dims,
        embed_dim   = embed_dim,
        hidden_dim  = hidden_dim,
        num_classes = len(CANCER_TYPES),
        dropout     = dropout,
    ).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Warmup + cosine annealing scheduler
    if use_scheduler:
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + math.cos(
                math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            ))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=patience)
    best_val_acc   = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_class_acc': [],
        'val_loss':   [], 'val_acc':   [], 'val_class_acc':   [],
    }

    print(f'\nFold {fold} — training started')
    print('=' * 70)

    for epoch in range(num_epochs):
        train_loss, train_acc, train_class_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            train_dataset.idx_to_label,
        )
        val_loss, val_acc, val_class_acc = validate(
            model, val_loader, criterion, device, val_dataset.idx_to_label,
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_class_acc'].append(train_class_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_class_acc'].append(val_class_acc)

        print(f'Epoch {epoch+1:3d}/{num_epochs} | '
              f'Train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'Val loss {val_loss:.4f} acc {val_acc:.4f}')

        if scheduler:
            scheduler.step()

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':              val_acc,
                'val_loss':             val_loss,
                'class_weights':        class_weights.cpu(),
            }, f'{save_dir}/{save_prefix}_fold{fold}_best.pt')

        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':              val_acc,
                'val_loss':             val_loss,
                'train_acc':            train_acc,
                'train_loss':           train_loss,
                'class_weights':        class_weights.cpu(),
            }, f'{save_dir}/{save_prefix}_fold{fold}_epoch{epoch+1}.pt')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Final evaluation with best checkpoint
    checkpoint = torch.load(f'{save_dir}/{save_prefix}_fold{fold}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    _, val_acc, val_class_acc, details = validate(
        model, val_loader, criterion, device,
        val_dataset.idx_to_label, return_details=True,
    )

    report = classification_report(
        details['labels'], details['predictions'],
        target_names=CANCER_TYPES,
        output_dict=True, zero_division=0,
    )

    results = {
        'fold':                  fold,
        'best_val_acc':          best_val_acc,
        'best_val_loss':         checkpoint['val_loss'],
        'history':               history,
        'predictions':           details['predictions'],
        'labels':                details['labels'],
        'sample_ids':            details['sample_ids'],
        'probabilities':         details['probabilities'],
        'confusion_matrix':      details['confusion_matrix'],
        'classification_report': report,
        'class_weights':         class_weights.cpu().numpy(),
    }

    with open(f'{save_dir}/{save_prefix}_fold{fold}_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    torch.save({
        'attention_weights': details['attention_weights'],
        'embeddings':        details['embeddings'],
        'sample_ids':        details['sample_ids'],
        'labels':            details['labels'],
        'predictions':       details['predictions'],
    }, f'{save_dir}/{save_prefix}_fold{fold}_analysis.pt')

    print(f'\nFold {fold} complete | Best Val Acc: {best_val_acc:.4f}')
    return model, results, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Load discovery cohort (cancer samples only)
    adata = sc.read_h5ad('./data/discovery_cohort_cancer.h5ad')

    # Load feature gene sets (selected on Fold 1 training data)
    with open('./data/periscan_2_genes.pkl', 'rb') as f:
        gene_lists = pickle.load(f)

    # Preprocess: retain cancer samples only
    adata = preprocess_adata(adata)

    # Compute cell count caps
    max_cells_dict = compute_max_cells(adata)

    # 5-fold cross-validation split
    adata = split_data_balanced_cv(adata, n_folds=5, random_state=42)

    # Train on each fold
    # Hyperparameters below reflect the final published configuration
    for fold in range(1, 6):
        run_training(
            adata          = adata,
            gene_lists     = gene_lists,
            max_cells_dict = max_cells_dict,
            fold           = fold,
            save_dir       = f'./results/fold{fold}',
            save_prefix    = 'periscan_2',
        )
