import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt import PoincareBall
from scipy.signal import firwin
from STGC import STGC

# =========================
# Environment (Optional)
# =========================
os.environ["TRITON_F32_DEFAULT"] = "ieee"
os.environ["TRITON_PRINT_INFO"] = "0"
os.environ["TRITON_SILENT"] = "1"


# =========================
# Utility Modules
# =========================
class Squeeze(nn.Module):
    """Squeeze tensor along a given dimension."""
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return self.pe[:, :x.size(1)]


class Attention(nn.Module):
    """Multi-head self-attention with LayerNorm output."""
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5

        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        # x: (B, T, D)
        batch_size, seq_len, _ = x.shape

        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)  # (B, T, D)

        return self.to_out(out)

# =========================
# Hyperbolic Modules
# =========================
class HyperbolicMapper(nn.Module):
    """Map Euclidean features to Poincaré ball via expmap0."""
    def __init__(self, input_dim: int, c: float = 1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)

        self.linear = nn.Linear(input_dim, input_dim)
        self.linear.weight = geoopt.ManifoldParameter(self.linear.weight, manifold=self.manifold)
        self.linear.bias = None

    def forward(self, x_euclidean):
        x_transformed = self.linear(x_euclidean)
        return self.manifold.expmap0(x_transformed)


class HyperbolicHierarchicalFusion(nn.Module):
    """Fuse two hyperbolic embeddings using learnable gating in tangent space."""
    def __init__(self, dim: int, c: float = 1.0):
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.attn_time = nn.Linear(2 * dim, 1)

    def forward(self, z_space, z_time):
        log_s = self.manifold.logmap0(z_space)
        log_t = self.manifold.logmap0(z_time)

        fused = torch.cat([log_s, log_t], dim=-1)  # (B, 2*dim)
        beta = torch.sigmoid(self.attn_time(fused))  # (B, 1)

        return self.manifold.expmap0(beta * log_s + (1.0 - beta) * log_t)


class HyperbolicModel(nn.Module):
    """Hyperbolic mapping + hierarchical fusion."""
    def __init__(self, manifold, input_dim: int = 128, num_classes: int = 2, c: float = 1.0):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_mapper = HyperbolicMapper(input_dim, c=c)
        self.fusion = HyperbolicHierarchicalFusion(dim=input_dim, c=c)

    def forward(self, g_feature, t_feature):
        z1 = self.hyperbolic_mapper(g_feature)
        z2 = self.hyperbolic_mapper(t_feature)
        return self.fusion(z1, z2)


# =========================
# Main Network
# =========================
class HE_MLPO(nn.Module):
    """
    EEG Feature Extractor + Graph Network + Hyperbolic Fusion.
    Input:  x (B, C, T)
    Output: feature, proto0, proto1, manifold, adj
    """
    def __init__(self, d_model: int, in_dim: int, sample_rate: int, layer: int, c: float = 1.0):
        super().__init__()

        # -------- Temporal multi-scale conv --------
        self.T_block1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.T_block2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.T_block3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.first_fusion = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, groups=3),
            nn.BatchNorm2d(3),
            nn.GELU(),
            nn.Conv2d(3, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(0.3),
            Squeeze(1),
        )

        # -------- Temporal attention + pooling --------
        self.position_embed = PositionalEmbedding(d_model)
        self.attention = Attention(d_model, num_heads=4, dropout=0.2)

        # NOTE: keep original behavior (even if AdaptiveAvgPool2d on 3D looks odd)
        self.second_fusion = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((32, 4)),
        )

        # -------- Graph network --------
        self.stgc = STGC(layer=layer, num_nodes=d_model, input_dim=in_dim, num_heads=8)

        self.fusion = nn.Sequential(
            nn.Conv2d(layer, layer, kernel_size=3, groups=layer),
            nn.BatchNorm2d(layer),
            nn.GELU(),
            nn.Conv2d(layer, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.GELU(),
            Squeeze(1),
            nn.AdaptiveAvgPool2d((32, 4)),
        )

        # -------- Hyperbolic fusion --------
        self.manifold = geoopt.PoincareBall(c=c)
        self.Hyperbolic = HyperbolicModel(self.manifold, input_dim=128, num_classes=2, c=c)

        # -------- Hyperbolic prototypes --------
        self.proto0 = geoopt.ManifoldParameter(
            data=self.manifold.expmap0(torch.zeros(1, 128) + 0.01),
            manifold=self.manifold,
        )
        self.proto1 = geoopt.ManifoldParameter(
            data=self.manifold.expmap0(torch.zeros(1, 128) - 0.01),
            manifold=self.manifold,
        )

    def forward(self, x):
        """
        x: (B, C, T)
        """
        batch, channel, seq_len = x.size()
        x = x.unsqueeze(1)  # (B, 1, C, T)

        # --- Multi-scale temporal conv ---
        x1 = self.T_block1(x)
        x2 = self.T_block2(x)
        x3 = self.T_block3(x)

        multi_scale = torch.cat((x1, x2, x3), dim=1)  # (B, 3, C, T)
        multi_scale = self.first_fusion(multi_scale)  # (B, C, T)

        # --- Temporal attention ---
        pos_in = multi_scale.transpose(1, 2)  # (B, T, C)
        pos_in = pos_in + self.position_embed(pos_in)
        t_attention = self.attention(pos_in).transpose(1, 2)  # (B, C, T)

        t_feature = self.second_fusion(t_attention).flatten(1, 2)
        print(t_feature.shape)
        # --- Graph feature ---
        g_feature, adj = self.stgc(x.squeeze(1))  # (B, layer, ?, ?), adj
        g_feature = self.fusion(g_feature).flatten(1, 2)
        print(g_feature.shape)
        # --- Hyperbolic fusion ---
        feature = self.Hyperbolic(g_feature, t_feature)

        return feature, self.proto0, self.proto1, self.manifold, adj


# =========================
# Parameter Count
# =========================
def get_parameter_number(model: nn.Module):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


if __name__ == "__main__":
    model = HE_MLPO(d_model=64, in_dim=128, sample_rate=128, layer=4)
    params = get_parameter_number(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 示例
    # model = YourModel(...)
    x = torch.randn((1, 64, 128), device=device)
    model = HE_MLPO(64, 128, sample_rate=128, layer=4).to(device)
    model(x)
    # print(model)
    print("Total:", params["Total"], "Trainable:", params["Trainable"])

