import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch.nn.attention import sdpa_kernel


from typing import List, Optional


# --------------------- Attention Pooling Layer ---------------------

# Adapted from:
# https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py#L156
# Implements an attention pooling mechanism to aggregate frame-level features (e.g., from Whisper or similar models) into a single embedding.
# A special [CLS] token is appended to the input sequence to serve as the query for the attention mechanism.


class Mlp(nn.Module):
    """
    A simple two-layer feed-forward MLP block commonly used in transformer-style architectures.

    Structure:
        Linear (in_features → hidden_features)
        → GELU activation
        → Linear (hidden_features → out_features)
        → Dropout

    This block is used after self-attention layers to project embeddings into a higher-dimensional
    space, apply a non-linear transformation, and then reduce them back to the desired output dimension.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        """
        Initialize the MLP block.

        Args:
            in_features (int): Input feature dimension.
            hidden_features (Optional[int], default=None): Dimension of the hidden layer.
                If None, defaults to `in_features`.
            out_features (Optional[int], default=None): Output feature dimension.
                If None, defaults to `in_features`.
            drop (float, default=0.0): Dropout probability applied after the final linear projection.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    RoPE is a technique introduced in "RoFormer: Enhanced Transformer with
    Rotary Position Embedding" (Su et al., 2021). It applies a rotation
    to query/key embeddings in self-attention to encode relative positions.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the RoPE module.

        Args:
            embedding_dims (int): Dimensionality of the input embeddings. Must be even.
            min_timescale (int, default=1): Lower bound for the sinusoidal frequencies.
            max_timescale (int, default=10000): Upper bound for the sinusoidal frequencies.
            dtype (torch.dtype, default=torch.float32): Data type used for internal computations.
        """
        super().__init__()

        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dimension must be even for RoPE.")

        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.compute_dtype = dtype

        # Precompute the scaling factors ("timescales") for sinusoidal frequencies.
        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)

        self.register_buffer("timescale", timescale, persistent=False)

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings to the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape (..., embedding_dims).
            position (torch.Tensor): Position indices of shape (...,)

        Returns:
            torch.Tensor: Tensor of the same shape as `inputs`, with RoPE applied.
        """
        position = position.unsqueeze(-1).unsqueeze(-1)

        sinusoid_inp = position / self.timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)

        first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)

        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin

        return torch.cat(
            (first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)),
            dim=-1,
        )


class Learned_Aggregation_Layer(nn.Module):
    """
    Learned Aggregation Layer with Attention Pooling.

    This layer uses a self-attention mechanism where only the [CLS] token
    acts as the query. The CLS token learns to aggregate information from
    all other tokens in the sequence (acting as a weighted summary).

    Structure:
        Input sequence (B, T, C)
        → Linear projections to Q (CLS), K (all tokens), V (all tokens)
        → Apply Rotary Embeddings to K for positional encoding
        → Scaled dot-product attention
        → Projection and dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Initialize the attention pooling layer.

        Args:
            dim (int): Input embedding dimension.
            num_heads (int, default=1): Number of attention heads.
            qkv_bias (bool, default=False): Whether to add bias terms in QKV projections.
            qk_scale (float, optional): Override for the QK scaling factor.
                If None, defaults to 1/sqrt(head_dim).
            attn_drop (float, default=0.0): Dropout applied to attention weights.
            proj_drop (float, default=0.0): Dropout applied to the output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Linear projections for query, key, value
        self.ql = nn.Linear(dim, dim, bias=qkv_bias)
        self.kl = nn.Linear(dim, dim, bias=qkv_bias)
        self.vl = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_prob = proj_drop

        # Rotary embeddings for positional encoding of keys
        self.rotary_emb = RotaryEmbedding(embedding_dims=head_dim)

        # Whether to use PyTorch's fused scaled_dot_product_attention (faster on CUDA)
        self.use_sdpa = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                B = batch size,
                T = sequence length,
                C = embedding dimension.

        Returns:
            torch.Tensor: Aggregated tensor of shape (B, 1, C),
            representing the [CLS] token after attention pooling.
        """
        B, T, C = x.shape
        H = self.num_heads
        E = C // H

        # --- Prepare queries, keys, values ---
        # CLS token (first token) acts as query
        q = x[:, 0].unsqueeze(1)  # (B, 1, C)
        k = x  # (B, T, C)
        v = x  # (B, T, C)

        # Project CLS → Q
        q = self.ql(q).view(B, 1, H, E)  # (B, 1, H, E)
        q = rearrange(q, "b t h e -> b h t e")

        # Project sequence → K and apply rotary embeddings
        k_pos = repeat(torch.arange(T), "l -> b l", b=B).to(x.device)
        k = self.kl(k).view(B, T, H, E)
        k = self.rotary_emb(k, position=k_pos)
        k = rearrange(k, "b t h e -> b h t e")

        # Project sequence → V
        v = self.vl(v).view(B, T, H, E)
        v = rearrange(v, "b t h e -> b h t e")

        # --- Attention computation ---
        if self.use_sdpa:
            # Use PyTorch's efficient fused attention kernel (CUDA only)
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
        else:
            # Manual attention implementation
            attn = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, 1, T)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # (B, H, 1, E)

        # --- Output projection ---
        x = rearrange(x, "b h t e -> b t (h e)")  # (B, 1, C)
        x = self.proj(x)  # (B, 1, C)
        x = self.proj_drop(x)
        return x


class Layer_scale_init_Block_only_token(nn.Module):
    """
    Transformer-style block that updates only the [CLS] token representation.

    The [CLS] token attends to all tokens in the sequence using an attention
    pooling block, then passes through an MLP. Both branches use residual
    connections scaled by learnable parameters (`gamma_1`, `gamma_2`).

    Structure:
        Input: token embeddings (x) and CLS embedding (x_cls)

        u = concat([CLS], x)
        x_cls = x_cls + gamma_1 * Attention(norm1(u))
        x_cls = x_cls + gamma_2 * MLP(norm2(x_cls))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        Attention_block=Learned_Aggregation_Layer,
        Mlp_block=Mlp,
        init_values: float = 1e-4,
    ):
        """
        Initialize the CLS-only Transformer block.

        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads in the attention block.
            mlp_ratio (float, default=2.0): Expansion ratio for the hidden
                dimension in the MLP block (hidden_dim = dim * mlp_ratio).
            qkv_bias (bool, default=False): Whether to use bias in QKV projections.
            qk_scale (float, optional): Override scaling factor for QK dot product.
                If None, defaults to 1/sqrt(head_dim).
            drop (float, default=0.0): Dropout probability for MLP and projections.
            attn_drop (float, default=0.0): Dropout probability for attention weights.
            Attention_block (nn.Module, default=Learned_Aggregation_Layer):
                The attention pooling block to apply on the CLS token.
            Mlp_block (nn.Module, default=Mlp): The feed-forward block applied
                after attention.
            init_values (float, default=1e-4): Initial scale for the residual
                scaling parameters (gamma_1, gamma_2).
        """
        super().__init__()

        # Normalization layers for attention and MLP branches
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Attention pooling (CLS attends to all tokens)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Feed-forward MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Learnable scaling parameters for residual connections
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, x_cls: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CLS-only Transformer block.

        Args:
            x (torch.Tensor): Sequence of token embeddings (B, T, C).
            x_cls (torch.Tensor): CLS token embedding (B, 1, C).

        Returns:
            torch.Tensor: Updated CLS token embedding (B, 1, C).
        """
        # Concatenate CLS token with all other tokens for attention
        u = torch.cat((x_cls, x), dim=1)  # (B, 1+T, C)

        # Attention branch: CLS attends to the sequence
        x_cls = x_cls + self.gamma_1 * self.attn(self.norm1(u))

        # MLP branch: transform CLS embedding
        x_cls = x_cls + self.gamma_2 * self.mlp(self.norm2(x_cls))

        return x_cls


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer.

    This module aggregates information from a sequence of embeddings into a single
    vector representation by introducing a learnable [CLS] token that attends to all
    other tokens. The [CLS] token is updated using a transformer-style block
    (attention + MLP), then normalized and returned as the pooled embedding.

    Structure:
        Input: sequence embeddings (B, T, C)
        → prepend a learnable CLS token
        → transformer block (Layer_scale_init_Block_only_token)
        → normalization
        → return updated CLS token (B, C)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: int = 2,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_scale: float = 1e-4,
        rescale: float = 0.02,
    ):
        """
        Initialize the AttentionPooling layer.

        Args:
            dim (int): Embedding dimension.
            num_heads (int, default=1): Number of attention heads.
            mlp_ratio (int, default=2): Expansion ratio for the hidden dimension in the MLP.
            qkv_bias (bool, default=False): Whether to use bias in the QKV projections.
            qk_scale (float, optional): Override for QK scaling factor. If None, defaults to 1/sqrt(head_dim).
            drop (float, default=0.0): Dropout applied in MLP and projections.
            attn_drop (float, default=0.0): Dropout applied to attention weights.
            init_scale (float, default=1e-4): Initial scale for residual layer-scaling parameters.
            rescale (float, default=0.02): Standard deviation used for truncated normal initialization
                of weights and CLS token.
        """
        super().__init__()

        # Core attention block: updates CLS token based on input sequence
        self.attn = Layer_scale_init_Block_only_token(
            dim=int(dim),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            Attention_block=Learned_Aggregation_Layer,
            Mlp_block=Mlp,
            init_values=init_scale,
        )

        # Final normalization applied after attention + MLP
        self.norm_layer = nn.LayerNorm(dim)

        # Learnable CLS token (acts as a pooling query)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(dim)))

        # Initialize CLS token with truncated normal distribution
        self.rescale = rescale
        trunc_normal_(self.cls_token, std=self.rescale)

        # Apply custom weight initialization to linear/LayerNorm modules
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Custom weight initialization for Linear and LayerNorm layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.rescale)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequence of embeddings of shape (B, T, C),
                where B = batch size, T = sequence length, C = embedding dimension.

        Returns:
            torch.Tensor: Pooled [CLS] embedding of shape (B, C).
        """
        B = x.shape[0]

        # Expand CLS token for the batch
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, C)

        # (Optional) Could initialize CLS token with mean of input features
        # cls_token = cls_token + x.mean(dim=1, keepdim=True)

        # Update CLS token using attention over the input sequence
        cls_token = self.attn(x, cls_token)

        # Concatenate updated CLS with the original sequence
        x = torch.cat((cls_token, x), dim=1)  # (B, 1+T, C)

        # Apply normalization
        x = self.norm_layer(x)

        # Return only the CLS token (aggregated sequence representation)
        return x[:, 0]  # (B, C)
