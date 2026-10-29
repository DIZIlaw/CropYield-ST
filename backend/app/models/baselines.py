"""
CropYield-ST: Baseline Machine Learning & Deep Learning Models
Implements benchmark models for rigorous comparative evaluation:
  1. Pure Temporal RNN (LSTM/GRU without spatial graph)
  2. Multi-Modal Fusion MLP (No spatial graph)
  3. Classical ML (XGBoost / Random Forest on seasonal aggregate features)
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from backend.app.models.temporal_encoder import TemporalEncoder


class PureTemporalRNN(nn.Module):
    """
    Temporal Baseline: Models crop phenology trajectories using LSTM/GRU
    WITHOUT any spatial graph convolution.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = TemporalEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            bidirectional=True,
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(x)
        out = self.regressor(h)
        return out, h, None


class ClassicalMLBaseline:
    """
    Classical Machine Learning Baseline (Random Forest or XGBoost)
    trained on aggregate seasonal summary statistics (mean, max, std, min).
    """
    def __init__(self, model_type: str = "xgboost", n_estimators: int = 100, random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        if self.model_type == "xgboost" and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=0.08,
                random_state=random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=random_state
            )
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalMLBaseline":
        self.model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
