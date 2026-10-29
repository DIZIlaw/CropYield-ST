"""
CropYield-ST: Spatio-Temporal Graph Neural Network (ST-GNN)
Integrates temporal recurrent trajectory modeling (LSTM/GRU) with
spatial graph convolutions over neighboring field parcels.
"""
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.models.temporal_encoder import TemporalEncoder
from backend.app.models.gcn import GCNEncoder


class CropYieldSTGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Parcel-Level Crop Yield Forecasting.
    
    Pipeline:
      1. Temporal Encoder: Transforms multi-sensor time-series [N, T, F] -> Temporal embeddings [N, d_temp]
      2. Spatial GCN: Convolves temporal embeddings over the parcel graph [N, N] -> Spatial embeddings [N, d_spatial]
      3. Multi-modal Fusion: Combines temporal dynamics and neighborhood spatial context
      4. Regression Head: Predicts hyperlocal yield (tonnes/ha) per parcel
    """
    def __init__(
        self,
        in_features: int,
        temporal_dim: int = 64,
        spatial_dim: int = 64,
        num_rnn_layers: int = 2,
        num_gcn_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        
        # 1. Temporal feature extractor
        self.temporal_encoder = TemporalEncoder(
            in_features=in_features,
            hidden_dim=temporal_dim,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            bidirectional=True,
            dropout=dropout
        )
        
        # 2. Spatial Graph Convolutional Network
        self.spatial_gcn = GCNEncoder(
            in_features=temporal_dim,
            hidden_dim=spatial_dim,
            out_features=spatial_dim,
            num_layers=num_gcn_layers,
            dropout=dropout
        )
        
        # 3. Fusion gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim, temporal_dim),
            nn.Sigmoid()
        )
        
        # 4. Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(temporal_dim + spatial_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N_parcels, T_timesteps, F_features]
        adj_norm: [N_parcels, N_parcels] normalized adjacency matrix
        
        Returns:
          yield_preds: [N_parcels, 1] predicted yield in tonnes/ha
          h_temporal: [N_parcels, temporal_dim]
          h_spatial: [N_parcels, spatial_dim]
        """
        # Step 1: Temporal encoding
        h_temp = self.temporal_encoder(x) # [N, temporal_dim]
        
        # Step 2: Spatial graph message passing
        h_spat = self.spatial_gcn(h_temp, adj_norm) # [N, spatial_dim]
        
        # Step 3: Multi-modal fusion with adaptive gating
        gate = self.fusion_gate(torch.cat([h_temp, h_spat], dim=-1))
        gated_temp = gate * h_temp + (1.0 - gate) * h_spat
        fused = torch.cat([gated_temp, h_spat], dim=-1) # [N, temporal_dim + spatial_dim]
        
        # Step 4: Yield regression
        yield_preds = self.regressor(fused) # [N, 1]
        
        return yield_preds, h_temp, h_spat
