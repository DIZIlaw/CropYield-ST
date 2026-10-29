"""
Unit tests for ST-GNN and baseline deep learning architectures.
"""
import torch
import pytest

from backend.app.models.temporal_encoder import TemporalEncoder
from backend.app.models.gcn import GraphConvolutionLayer, GCNEncoder
from backend.app.models.st_gnn import CropYieldSTGNN
from backend.app.models.baselines import PureTemporalRNN, ClassicalMLBaseline


def test_temporal_encoder_shape():
    """Verify temporal recurrent encoder output shape [N, hidden_dim]."""
    N, T, F = 16, 12, 8
    hidden_dim = 32
    x = torch.randn(N, T, F)
    
    encoder = TemporalEncoder(in_features=F, hidden_dim=hidden_dim, num_layers=2, rnn_type="gru")
    out = encoder(x)
    
    assert out.shape == (N, hidden_dim)
    assert not torch.isnan(out).any()


def test_gcn_layer_message_passing():
    """Verify GCN message passing transforms representation using normalized adjacency."""
    N, in_f, out_f = 6, 16, 32
    x = torch.randn(N, in_f)
    adj_norm = torch.eye(N) # Identity adjacency
    
    gcn = GraphConvolutionLayer(in_features=in_f, out_features=out_f)
    out1 = gcn(x, adj_norm)
    assert out1.shape == (N, out_f)
    
    # Fully connected adjacency
    adj_dense = torch.ones(N, N) / N
    out2 = gcn(x, adj_dense)
    assert out2.shape == (N, out_f)
    
    # Graph structure should change representation
    assert not torch.allclose(out1, out2)


def test_st_gnn_forward_and_backward():
    """Verify integrated CropYieldSTGNN forward pass and loss backpropagation."""
    N, T, F = 10, 8, 6
    x = torch.randn(N, T, F)
    adj = torch.eye(N)
    y_true = torch.randn(N, 1)
    
    model = CropYieldSTGNN(
        in_features=F,
        temporal_dim=32,
        spatial_dim=32,
        num_rnn_layers=1,
        num_gcn_layers=1,
        dropout=0.1
    )
    
    preds, h_temp, h_spat = model(x, adj)
    assert preds.shape == (N, 1)
    assert h_temp.shape == (N, 32)
    assert h_spat.shape == (N, 32)
    
    # Test backward pass
    loss = torch.nn.functional.mse_loss(preds, y_true)
    loss.backward()
    
    # Verify gradients computed
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_pure_temporal_rnn_forward():
    """Verify baseline PureTemporalRNN forward pass."""
    N, T, F = 8, 10, 5
    x = torch.randn(N, T, F)
    
    baseline = PureTemporalRNN(in_features=F, hidden_dim=32)
    preds, h, _ = baseline(x)
    assert preds.shape == (N, 1)
    assert h.shape == (N, 32)
