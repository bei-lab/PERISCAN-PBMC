"""
PERISCAN Feature Fusion Module
Integrates features across different cell types using cross-attention mechanisms.
"""

import torch
import torch.nn as nn


class PERISCANFeatureFusion(nn.Module):
    """Feature fusion module that integrates representations across cell types."""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=256, num_heads=4, dropout=0.1):
        """
        Initialize Feature Fusion module.
        
        Args:
            input_dim: Input dimension from cell encoders
            hidden_dim: Hidden dimension for processing
            output_dim: Final output dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Define cell types (should match your data)
        self.cell_types = ['CD14high_Monocyte', 'CD4_Naive_CCR7', 'CD56lowCD16high_NK', 
                          'CD8_Temra_FGFBP2', 'NaiveB-TCL1A']
        
        # Cell type-specific feature enhancement
        self.type_specific = nn.ModuleDict({
            cell_type: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for cell_type in self.cell_types
        })
        
        # Cross-cell-type attention
        self.type_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self._create_output_layer(dropout)
        
        self._init_weights()
    
    def _create_output_layer(self, dropout):
        """Create output layer with appropriate architecture."""
        if self.output_dim < self.hidden_dim and self.hidden_dim / self.output_dim > 2:
            # Two-step dimension reduction
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
    
    def forward(self, features, mask=None):
        """
        Forward pass of feature fusion.
        
        Args:
            features: Dictionary of cell type features {cell_type: [batch_size, max_cells, input_dim]}
            mask: Dictionary of cell type masks {cell_type: [batch_size, max_cells]}
            
        Returns:
            Fused features and attention maps
        """
        batch_size = next(iter(features.values())).shape[0]
        
        # Handle NaN inputs
        for ct in features:
            features[ct] = torch.nan_to_num(features[ct], nan=0.0)
        
        # Cell type-specific feature enhancement
        enhanced = {}
        for cell_type, feat in features.items():
            enhanced_feat = self.type_specific[cell_type](feat)
            enhanced_feat = torch.nan_to_num(enhanced_feat, nan=0.0)
            enhanced[cell_type] = enhanced_feat
        
        # Aggregate features for each cell type (with masking)
        type_features = []
        for ct in self.cell_types:
            if mask is not None and ct in mask:
                # Masked average
                valid_mask = mask[ct]  # [batch_size, max_cells]
                sum_mask = valid_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
                safe_mask = torch.where(sum_mask > 0, sum_mask, torch.ones_like(sum_mask))
                
                masked_features = enhanced[ct] * valid_mask.unsqueeze(-1)
                avg_feature = masked_features.sum(dim=1) / safe_mask
                avg_feature = torch.where(torch.isnan(avg_feature), 
                                        torch.zeros_like(avg_feature), avg_feature)
            else:
                # Simple average
                avg_feature = enhanced[ct].mean(dim=1)
                avg_feature = torch.where(torch.isnan(avg_feature), 
                                        torch.zeros_like(avg_feature), avg_feature)
            
            type_features.append(avg_feature)
        
        # Stack features for cross-attention
        stacked = torch.stack(type_features, dim=1)  # [batch_size, num_cell_types, hidden_dim]
        stacked = torch.nan_to_num(stacked, nan=0.0)
        
        # Prepare cell-type level mask
        if mask is not None:
            type_mask = torch.stack([
                mask[ct].any(dim=1) for ct in self.cell_types
            ], dim=1)  # [batch_size, num_cell_types]
            key_padding_mask = ~type_mask
        else:
            key_padding_mask = None
        
        # Cross-cell-type attention
        type_features, type_attention = self.type_attention(
            query=stacked,
            key=stacked,
            value=stacked,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        type_features = torch.nan_to_num(type_features, nan=0.0)
        type_features = self.attention_norm(type_features + stacked)
        type_features = torch.nan_to_num(type_features, nan=0.0)
        
        # Feed-forward network
        ffn_output = self.ffn(type_features)
        ffn_output = torch.nan_to_num(ffn_output, nan=0.0)
        type_features = self.ffn_norm(type_features + ffn_output)
        type_features = torch.nan_to_num(type_features, nan=0.0)
        
        # Global feature aggregation
        if mask is not None and key_padding_mask is not None:
            valid_types = (~key_padding_mask).float()
            sum_valid = valid_types.sum(dim=1, keepdim=True)
            safe_sum = torch.where(sum_valid > 0, sum_valid, torch.ones_like(sum_valid))
            masked_features = type_features * valid_types.unsqueeze(-1)
            global_features = masked_features.sum(dim=1) / safe_sum
        else:
            global_features = type_features.mean(dim=1)
        
        global_features = torch.nan_to_num(global_features, nan=0.0)
        global_features = torch.clamp(global_features, -10.0, 10.0)
        
        # Output processing
        outputs = self.output_layer(global_features)
        outputs = torch.nan_to_num(outputs, nan=0.0)
        
        aux_outputs = {
            'attention_maps': {
                'celltype_attention': type_attention,
            },
            'intermediate_features': {
                'enhanced_features': enhanced,
                'type_features': type_features,
                'global_features': global_features
            }
        }
        
        return outputs, aux_outputs


def create_feature_fusion(config):
    """
    Create feature fusion module.
    
    Args:
        config: PERISCAN configuration
        
    Returns:
        Feature fusion module
    """
    fusion_model = PERISCANFeatureFusion(
        input_dim=config.cell_output_dim,
        hidden_dim=config.fusion_hidden_dim,
        output_dim=config.fusion_output_dim,
        num_heads=config.fusion_num_heads,
        dropout=config.fusion_dropout
    )
    
    print(f"Created feature fusion: {config.cell_output_dim} -> {config.fusion_output_dim} dim")
    return fusion_model