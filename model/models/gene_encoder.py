"""
PERISCAN Gene Encoder Module
Processes gene expression within individual cells using transformer attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PERISCANGeneEncoder(nn.Module):
    """Gene-level encoder with multi-head attention for processing gene expression within cells."""
    
    def __init__(self, gene_dim, hidden_dim=256, num_heads=4, dropout=0.1):
        """
        Initialize Gene Encoder.
        
        Args:
            gene_dim: Number of input genes
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gene-gene attention
        self.gene_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.gene_norm = nn.LayerNorm(hidden_dim)
        
        # Cell-gene attention
        self.cell_gene_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cell_norm = nn.LayerNorm(hidden_dim)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Residual projection
        self.residual_projection = nn.Linear(gene_dim, hidden_dim) if gene_dim != hidden_dim else nn.Identity()
        
        # Analysis projection for gene importance scores
        self.analysis_projection = nn.Sequential(
            nn.Linear(hidden_dim, gene_dim),
            nn.LayerNorm(gene_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None, return_gene_scores=False):
        """
        Forward pass of gene encoder.
        
        Args:
            x: Gene expression tensor [batch_size, max_cells, gene_dim]
            mask: Cell mask [batch_size, max_cells]
            return_gene_scores: Whether to return gene importance scores
            
        Returns:
            Encoded features and attention maps
        """
        batch_size, max_cells, _ = x.shape
        
        # Handle NaN inputs
        x = torch.nan_to_num(x, nan=0.0)
        
        # Prepare attention mask
        key_padding_mask = ~mask if mask is not None else None
        
        # Input projection
        features = self.input_projection(x)
        residual = self.residual_projection(x)
        
        # Gene-gene attention
        gene_features, gene_attention = self.gene_attention(
            query=features,
            key=features,
            value=features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        gene_features = self.gene_norm(gene_features + residual)
        
        # Cell-gene attention
        cell_features, cell_gene_attention = self.cell_gene_attention(
            query=gene_features,
            key=gene_features,
            value=gene_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        cell_features = self.cell_norm(cell_features + gene_features)
        
        # Feature fusion
        gene_features = torch.clamp(gene_features, -10.0, 10.0)
        cell_features = torch.clamp(cell_features, -10.0, 10.0)
        
        combined_features = torch.cat([gene_features, cell_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Output projection
        outputs = self.output_projection(fused_features)
        
        # Handle any remaining NaNs
        outputs = torch.nan_to_num(outputs, nan=0.0)
        
        # Prepare attention maps
        attention_maps = {
            'gene_gene_attention': gene_attention,
            'gene_cell_attention': cell_gene_attention
        }
        
        # Apply mask to attention maps if available
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attention_maps['gene_gene_attention'] = gene_attention * mask_expanded
            attention_maps['gene_cell_attention'] = cell_gene_attention * mask_expanded
        
        # Calculate gene importance scores if requested
        if return_gene_scores:
            with torch.no_grad():
                gene_scores = self.analysis_projection(fused_features)
                gene_scores = F.softmax(gene_scores, dim=-1)
                
                if mask is not None:
                    gene_scores = gene_scores * mask.unsqueeze(-1)
                
                gene_scores = torch.nan_to_num(gene_scores, nan=0.0)
                attention_maps['cell_gene_scores'] = gene_scores
        
        return outputs, {
            'attention_maps': attention_maps,
            'intermediate_features': {
                'gene_features': gene_features,
                'cell_features': cell_features,
                'fused_features': fused_features
            }
        }


def create_gene_encoders(gene_lists, config):
    """
    Create gene encoders for all cell types.
    
    Args:
        gene_lists: Dictionary with genes per cell type
        config: PERISCAN configuration
        
    Returns:
        Dictionary of gene encoders
    """
    gene_encoders = {}
    
    for cell_type, genes in gene_lists.items():
        gene_encoders[cell_type] = PERISCANGeneEncoder(
            gene_dim=len(genes),
            hidden_dim=config.gene_hidden_dim,
            num_heads=config.gene_num_heads,
            dropout=config.gene_dropout
        )
        
        print(f"Created gene encoder for {cell_type}: {len(genes)} genes -> {config.gene_hidden_dim} dim")
    
    return gene_encoders