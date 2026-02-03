import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


# =========================================================
# 1) Dynamic Adjacency via Multi-Head Attention
# =========================================================
class Attention_guide_graph(nn.Module):
    """
    Generate a dynamic adjacency matrix using multi-head self-attention.

    Input:
        x: (B, N, F)  - node features
    Output:
        adj_norm: (B, N, N) - symmetric normalized adjacency matrix
    """

    def __init__(self, num_nodes: int, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, F)
        return: (B, N, N)
        """
        B, N, Fdim = x.shape
        assert N == self.num_nodes, f"Expected N={self.num_nodes}, got {N}"

        # Q/K/V: (B, N, H, Dh)
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim)

        # Attention scores: (B, H, N, N)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (self.head_dim ** 0.5)

        # NOTE: keep your original scaling behavior (even though it's extreme)
        scores = scores * 1e9

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)

        # Average over heads -> adjacency
        adj = attn_weights.mean(dim=1)  # (B, N, N)

        # Symmetrize (undirected graph)
        adj = (adj + adj.transpose(1, 2)) / 2.0

        # Normalize: D^{-1/2} A D^{-1/2}
        degree = adj.sum(dim=2).pow(-0.5)  # (B, N)
        adj_norm = degree.unsqueeze(2) * adj * degree.unsqueeze(1)

        return adj_norm


# =========================================================
# 2) Chebyshev Graph Convolution
# =========================================================
class GCN(nn.Module):
    """
    Chebyshev Graph Convolution (ChebNet-style).

    H' = σ( Σ_{k=0}^{K-1} θ_k T_k(L~) H )

    Inputs:
        H:      (B, N, F_in)
        A_norm: (B, N, N)

    Output:
        H_out:  (B, N, F_out)
    """

    def __init__(self, in_features: int, out_features: int, K: int = 3):
        super().__init__()
        self.K = K

        self.theta = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features, out_features))
            for _ in range(K)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.theta:
            nn.init.xavier_uniform_(param)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        B, N, Fin = H.shape

        # Laplacian: L = I - A_norm
        I = torch.eye(N, device=A_norm.device).unsqueeze(0).expand(B, -1, -1)
        L = I - A_norm

        # Scaled Laplacian approximation (λ_max≈2) => L_tilde = L - I
        L_tilde = L - I

        # Chebyshev polynomials T_k(L_tilde)
        T_k = [I, L_tilde]  # T0=I, T1=L_tilde
        for k in range(2, self.K):
            T_k.append(2 * torch.matmul(L_tilde, T_k[k - 1]) - T_k[k - 2])

        # Aggregate
        out = torch.zeros(B, N, self.theta[0].shape[1], device=H.device, dtype=H.dtype)
        for k in range(self.K):
            transformed = torch.matmul(T_k[k], H)  # (B, N, Fin)
            out = out + torch.einsum("bnf,fg->bng", transformed, self.theta[k])

        return F.relu(out)


# =========================================================
# 3) EEG Graph Network (Attention Adjacency + ChebConv Layers)
# =========================================================
class STGC(nn.Module):
    """
    EEG graph backbone.

    Input:
        x: (B, N, F)
    Output:
        V:   (B, layer, N, F) - stacked layer outputs
        adj: (B, N, N)        - dynamic normalized adjacency
    """

    def __init__(self, layer: int = 4, num_nodes: int = 32, input_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.attention = Attention_guide_graph(num_nodes, input_dim, num_heads)

        self.gcn = nn.ModuleList([
            GCN(input_dim, input_dim, K=3)
            for _ in range(layer)
        ])

    def forward(self, x: torch.Tensor):
        adj_norm = self.attention(x)  # (B, N, N)

        layer_outputs = []
        for conv in self.gcn:
            x = conv(x, adj_norm)  # (B, N, F)
            layer_outputs.append(x.unsqueeze(1))  # (B, 1, N, F)

        V = torch.cat(layer_outputs, dim=1)  # (B, layer, N, F)
        print(V.shape)
        return V, adj_norm