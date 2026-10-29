"""
CropYield-ST: Graph Convolutional Network (GCN) Spatial Module
Implements spectral graph convolution over parcel adjacency matrices (Kipf & Welling formulation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    Standard GCN Layer:
      Z = A_norm * X * W + b
    where A_norm is the symmetrically normalized adjacency matrix with self-loops:
      D^-0.5 * (A + I) * D^-0.5
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    def reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix [N, in_features]
        adj_norm: Symmetrically normalized adjacency matrix [N, N]
        returns: [N, out_features]
        """
        # Linear transform: X * W -> [N, out_features]
        support = torch.matmul(x, self.weight)
        # Message passing: A_norm * support -> [N, out_features]
        output = torch.matmul(adj_norm, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNEncoder(nn.Module):
    """
    Multi-layer Graph Convolutional Network with residual connections,
    LayerNorm, activation, and dropout.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_features: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Layer 1
        self.layers.append(GraphConvolutionLayer(in_features, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Intermediate and final layers
        for _ in range(num_layers - 1):
            self.layers.append(GraphConvolutionLayer(hidden_dim, out_features))
            self.norms.append(nn.LayerNorm(out_features))
            hidden_dim = out_features
            
        self.residual_proj = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_features]
        adj_norm: [N, N]
        """
        residual = self.residual_proj(x)
        h = x
        for i in range(self.num_layers):
            h = self.layers[i](h, adj_norm)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        # Add residual connection from input
        return h + residual
