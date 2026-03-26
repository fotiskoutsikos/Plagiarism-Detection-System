import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Optional, Tuple

# LAYERS
class MeanPool(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            x = x.transpose(1, 2)  # (B, T, C)
            mask = mask.unsqueeze(-1).float()
            x = x * mask
            return x.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return x.mean(dim=2)

class GeMPool(nn.Module):
    def __init__(self, ncha=1, init=3, eps=1e-6):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.softplus = nn.Softplus()
        pinit = math.log(math.exp(init - 1) - 1)
        self.p = nn.Parameter(pinit * torch.ones(1, ncha, 1))
        self.eps = eps

    def forward(self, h):
        h = self.flatten(h)
        pow = 1 + self.softplus(self.p)
        h = h.clamp(min=self.eps).pow(pow)
        h = h.mean(-1).pow(1 / pow.squeeze(-1))
        return h

# POSITIONAL ENCODING
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# MAIN MODEL
class Model(nn.Module):
    def __init__(self, conf: DictConfig, use_avg_pooling: bool = False, embedding_type: str = "last_hidden_states"):
        super().__init__()
        self.working_dim = conf.hidden_dim
        self.use_avg_pooling = use_avg_pooling
        
        # Input projection
        self.pre_transformer_proj = nn.Linear(conf.input_channels, self.working_dim)
        
        # Positional Encoding & Transformer
        self.positional_encoding = SinusoidalPositionalEncoding(
            hidden_dim=self.working_dim, 
            max_length=5000, 
            dropout=conf.dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.working_dim,
            nhead=conf.num_heads,
            dim_feedforward=conf.ff_dim,
            dropout=conf.dropout,
            batch_first=True,
            norm_first=True  # Pre-LN architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=conf.num_transformer_blocks)
        
        # Pooling & Projection
        self.pool = GeMPool(ncha=self.working_dim)
        self.proj = nn.Linear(self.working_dim, conf.zdim, bias=False)

    def embed(self, h: torch.Tensor) -> Tuple[torch.Tensor, None]:
        if h.ndim == 2: h = h.unsqueeze(1)
        
        x = self.pre_transformer_proj(h)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        
        # Pooling
        x = x.transpose(1, 2)
        x = self.pool(x)
        
        z = self.proj(x)
        return z, None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z, _ = self.embed(h)
        return z