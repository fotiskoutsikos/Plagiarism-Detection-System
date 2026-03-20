import math
from typing import Optional, Tuple, Union
import torch
from omegaconf import DictConfig

import torch.nn as nn
import torch.nn.functional as F


# INLINE POOLING LAYERS
class MeanPool(nn.Module):
    """Simple mean pooling layer."""
    def forward(self, h):
        return h.mean(dim=-1)


class GeMPool(nn.Module):
    """Generalized Mean Pooling layer with learnable power parameter."""
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
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# MAIN MODEL
class Model(nn.Module):
    """Transformer-based model for audio-lyrics matching (inference-only)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.working_dim = config.working_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.activation = config.activation
        self.pooling_type = config.pooling_type

        # Build model components
        self.conv_layers = self._build_conv_layers()
        self.positional_encoding = self._build_positional_encoding()
        self.transformer = self._build_transformer()
        self.pooling_layer = self._build_pooling_layer()
        self.avg_mlp = self._build_avg_mlp()

    def _build_conv_layers(self) -> nn.Module:
        """Build convolutional layers. Returns Identity for conv_blocks: 0."""
        return nn.Identity()

    def _build_positional_encoding(self) -> SinusoidalPositionalEncoding:
        """Build positional encoding."""
        return SinusoidalPositionalEncoding(
            d_model=self.working_dim,
            max_len=self.config.get("max_seq_length", 5000),
        )

    def _build_transformer(self) -> nn.TransformerEncoder:
        """Build transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.working_dim,
            nhead=self.num_heads,
            dim_feedforward=self.config.get("dim_feedforward", 2048),
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def _build_pooling_layer(self) -> nn.Module:
        """Build pooling layer."""
        if self.pooling_type == "gem":
            return GeMPool(ncha=self.working_dim)
        else:
            return MeanPool()

    def _build_avg_mlp(self) -> nn.Module:
        """Build MLP head for averaging embeddings."""
        return nn.Sequential(
            nn.Linear(self.working_dim, self.working_dim),
            nn.ReLU(),
            nn.Linear(self.working_dim, self.working_dim),
        )

    def get_shingle_params(self) -> Tuple[int, int]:
        """Return shingle size and hop length."""
        return self.config.shingle_size, self.config.shingle_hop

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input tensor (handle shape if needed)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Generate embeddings from input tensor."""
        # Pass through conv layers (Identity in this case)
        x = self.conv_layers(x)

        # Project to working dimension if needed
        if x.shape[-1] != self.working_dim:
            x = F.linear(x, torch.randn(self.working_dim, x.shape[-1]))

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply transformer
        x = self.transformer(x)

        # Pool temporal dimension
        x = self.pooling_layer(x)

        # Apply MLP head
        x = self.avg_mlp(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: prepare input and generate embeddings."""
        x = self.prepare(x)
        embeddings = self.embed(x)
        return embeddings