"""
PERISCAN Cell Encoder Module
Aggregates gene-level features to generate cell-level representations.
"""

import torch
import torch.nn as nn


class PERISCANCellEncoder(nn.Module):
    """Cell-level encoder that processes gene features to create cell representations."""
    
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128, num_heads=4, dropout=0.1):
        """
        Initialize Cell Encoder.
        
        Args:
            input_dim: Input feature dimension from gene encoder
            hidden_dim: Hidden dimension size
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cell-cell attention
        self.cell_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cell_norm = nn.LayerNorm(hidden_dim)
        
        # Gene feature integration
        self.gene_integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output layer
        self._create_output_layer(dropout)
        
        self._init_weights()
    
    def _create_output_layer(self, dropout):
        """Create output layer with appropriate architecture based on dimension reduction."""
        if self.output_dim < self.hidden_dim and self.hidden_dim / self.output_dim > 4:
            # Two-step dimension reduction for large reductions
            middle_dim = (self.hidden_dim + self.output_dim) // 2
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, middle_dim),
                nn.LayerNorm(middle_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(middle_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.Tanh()
            )
        else:
            # Single-step dimension reduction
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.Tanh()
            )
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, gene_features=None, mask=None):
        """
        Forward pass of cell encoder.
        
        Args:
            x: Input features from gene encoder [batch_size, max_cells, input_dim]
            gene_features: Optional gene-level features [batch_size, max_cells, hidden_dim]
            mask: Cell mask [batch_size, max_cells]
            
        Returns:
            Encoded cell features and attention maps
        """
        batch_size, max_cells, _ = x.shape
        
        # Handle NaN inputs
        x = torch.nan_to_num(x, nan=0.0)
        if gene_features is not None:
            gene_features = torch.nan_to_num(gene_features, nan=0.0)
        
        # Prepare attention mask
        key_padding_mask = ~mask if mask is not None else None
        
        # Input projection
        features = self.input_projection(x)
        features = torch.nan_to_num(features, nan=0.0)
        
        # Cell-cell attention
        cell_features, cell_attention = self.cell_attention(
            query=features,
            key=features,
            value=features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        cell_features = torch.nan_to_num(cell_features, nan=0.0)
        cell_features = self.cell_norm(cell_features + features)  # Residual connection
        cell_features = torch.nan_to_num(cell_features, nan=0.0)
        
        # Integrate gene features if provided
        if gene_features is not None:
            # Clamp values to prevent extreme values
            cell_features = torch.clamp(cell_features, -10.0, 10.0)
            gene_features = torch.clamp(gene_features, -10.0, 10.0)
            
            combined = torch.cat([cell_features, gene_features], dim=-1)
            features = self.gene_integration(combined)
            features = torch.nan_to_num(features, nan=0.0)
        else:
            features = cell_features
        
        # Output projection
        outputs = self.output_layer(features)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        
        # Apply mask to attention maps if available
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            cell_attention = cell_attention * mask_expanded
        
        return outputs, {
            'attention_maps': {
                'cell_cell_attention': cell_attention
            },
            'intermediate_features': {
                'cell_features': cell_features,
                'final_features': outputs
            }
        }


def create_cell_encoders(cell_types, config):
    """
    Create cell encoders for all cell types.
    
    Args:
        cell_types: List of cell type names
        config: PERISCAN configuration
        
    Returns:
        Dictionary of cell encoders
    """
    cell_encoders = {}
    
    for cell_type in cell_types:
        cell_encoders[cell_type] = PERISCANCellEncoder(
            input_dim=config.gene_hidden_dim,
            hidden_dim=config.cell_hidden_dim,
            output_dim=config.cell_output_dim,
            num_heads=config.cell_num_heads,
            dropout=config.cell_dropout
        )
        
        print(f"Created cell encoder for {cell_type}: {config.gene_hidden_dim} -> {config.cell_output_dim} dim")
    
    return cell_encoders