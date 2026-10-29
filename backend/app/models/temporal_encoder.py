"""
CropYield-ST: Temporal Encoder Module
Encodes multi-spectral & SAR temporal trajectories using LSTM or GRU.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """
    Temporal recurrent encoder (LSTM or GRU) for agricultural parcel growth trajectories.
    
    Input shape:
      [N_parcels, T_timesteps, F_features]
    Output shape:
      [N_parcels, embedding_dim]
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Linear projection layer before recurrent unit
        self.input_projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=rnn_dropout
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=rnn_dropout
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose 'lstm' or 'gru'.")
            
        out_dim = hidden_dim * self.num_directions
        self.out_projection = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, T, F]
        returns: [N, hidden_dim]
        """
        # 1. Project input features
        h = self.input_projection(x) # [N, T, hidden_dim]
        
        # 2. Recurrent pass
        out, _ = self.rnn(h) # [N, T, hidden_dim * num_directions]
        
        # 3. Temporal pooling (average pool across time steps + last step)
        mean_pool = torch.mean(out, dim=1) # [N, hidden_dim * num_directions]
        last_step = out[:, -1, :] # [N, hidden_dim * num_directions]
        combined = 0.6 * mean_pool + 0.4 * last_step
        
        # 4. Final embedding projection
        emb = self.out_projection(combined) # [N, hidden_dim]
        return emb
