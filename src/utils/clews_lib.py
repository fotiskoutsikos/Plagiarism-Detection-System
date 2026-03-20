import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nnAudio import features

# TENSOR OPS 
def force_length(x: torch.Tensor, length: int, dim: int = -1, pad_mode: str = "constant", allow_longer: bool = False) -> torch.Tensor:
    """
    Ensure x has at least `length` along dim. If shorter, pad; if longer:
    - if allow_longer=False => crop
    - if allow_longer=True => keep as-is
    """
    orig_len = x.size(dim)
    if orig_len == length:
        return x

    if orig_len < length:
        pad_amt = length - orig_len
        if pad_mode == "repeat":
            # tile end section to pad
            if orig_len == 0:
                pad_tensor = x.new_zeros(*x.shape[:dim], pad_amt, *x.shape[dim + 1 :])
            else:
                repeat_tile = x.narrow(dim, orig_len - 1, 1).expand(*x.shape[:dim], pad_amt, *x.shape[dim + 1 :])
                pad_tensor = repeat_tile
        else:
            pad_tensor = x.new_zeros(*x.shape[:dim], pad_amt, *x.shape[dim + 1 :])
        return torch.cat([x, pad_tensor], dim=dim)

    if orig_len > length:
        if allow_longer:
            return x
        return x.narrow(dim, 0, length)

    return x


def get_frames(x: torch.Tensor, frame_length: int, hop_length: int, pad_mode: str = "constant") -> torch.Tensor:
    """
    Slice signal into overlapping frames.
    x: [B, T] or [T]
    returns: [B, n_frames, frame_length]
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)

    batch, total = x.shape
    if total < frame_length:
        x = force_length(x, frame_length, dim=1, pad_mode=pad_mode, allow_longer=False)
        total = x.shape[1]

    n_frames = 1 + max(0, math.floor((total - frame_length) / hop_length))
    if n_frames <= 0:
        # at least one frame
        x = force_length(x, frame_length, dim=1, pad_mode=pad_mode, allow_longer=False)
        n_frames = 1

    frames = x.unfold(1, frame_length, hop_length)  # [B, n_frames, frame_length]
    if frames.shape[1] != n_frames:
        # fallback to manual frame counting and pad
        desired = n_frames
        if frames.shape[1] < desired:
            pad_frames = desired - frames.shape[1]
            last = frames[:, -1:].expand(batch, pad_frames, frame_length)
            frames = torch.cat([frames, last], dim=1)
    return frames.contiguous()


# LAYERS 
class CQTPrepare(nn.Module):
    """Simple CQT preparation block. Applies power and scaling."""
    def __init__(self, pow: float = 1.0):
        super().__init__()
        self.pow = pow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected non-negative magnitude tensors
        x = x.abs() + 1e-8
        if self.pow != 1.0:
            x = x.pow(self.pow)
        return x


class PadConv2d(nn.Module):
    """Conv2d with explicit pad mode on the spatial dims."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, pad_mode="reflect"):
        super().__init__()
        self.pad_size = padding
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        if self.pad_size > 0:
            x = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode=self.pad_mode)
        return self.conv(x)


class InstanceBatchNorm2d(nn.Module):
    """Blend of InstanceNorm2d and BatchNorm2d for robust normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.inst = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=False)
        self.batch = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        out_inst = self.inst(x)
        out_batch = self.batch(x)
        return self.alpha * out_inst + (1.0 - self.alpha) * out_batch


class SqueezeExcitation2d(nn.Module):
    """Squeeze-and-Excitation 2D."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.shape
        y = x.mean(dim=[2, 3])
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class MyIBNResBlock(nn.Module):
    """Residual block with IBN and optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = PadConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = InstanceBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PadConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = InstanceBatchNorm2d(out_channels)
        self.se = SqueezeExcitation2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                InstanceBatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class GeMPool(nn.Module):
    """Generalized mean pooling."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return F.adaptive_avg_pool2d((x.clamp(min=self.eps).pow(self.p)), 1).pow(1.0 / self.p).view(x.size(0), x.size(1))


# MAIN MODEL 
class Model(torch.nn.Module):
    """Standalone CLEWS feature extractor (inference-only)."""
    def __init__(self, conf, sr=16000):
        super().__init__()
        self.sr = sr

        # Shingling configuration
        self.shingling_len = conf.shingling.len
        self.shingling_hop = conf.shingling.hop
        self.minlen = self.shingling_len

        # Constant-Q transform frontend
        self.cqt = nn.Sequential(
            features.CQT1992v2(
                sr=self.sr,
                hop_length=int(conf.cqt.hoplen * sr),
                n_bins=int(conf.cqt.noctaves * conf.cqt.nbinsoct),
                bins_per_octave=int(conf.cqt.nbinsoct),
                filter_scale=float(conf.cqt.fscale),
                trainable=False,
                verbose=False,
            ),
            nn.AvgPool1d(int(conf.cqt.pool), stride=int(conf.cqt.pool)),
        )

        # Frontend conv block
        ncha0, ncha = conf.frontend.channels
        self.frontend = nn.Sequential(
            CQTPrepare(pow=float(conf.frontend.cqtpow)),
            nn.Conv2d(1, ncha0, kernel_size=(12, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(ncha0),
            nn.ReLU(inplace=True),
            nn.Conv2d(ncha0, ncha, kernel_size=(12, 3), stride=2, bias=False),
        )

        # Backbone residual blocks
        blocks = []
        in_channels = ncha
        for nb, nc, st in zip(conf.backbone.blocks, conf.backbone.channels, conf.backbone.down):
            blocks.append(MyIBNResBlock(in_channels, nc, stride=st))
            for _ in range(nb - 1):
                blocks.append(MyIBNResBlock(nc, nc, stride=1))
            in_channels = nc
        self.backbone = nn.Sequential(*blocks)

        # Pooling and projection
        self.pool = GeMPool()
        self.proj = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, int(conf.zdim), bias=False),
        )

    def get_shingle_params(self):
        """Returns currently configured shingle length and hop in seconds."""
        return self.shingling_len, self.shingling_hop

    def prepare(self, h: torch.Tensor, shingle_len=None, shingle_hop=None) -> torch.Tensor:
        """Convert raw waveform [B, T] into CQT tensor [B, S, C, T]."""
        assert h.ndim == 2, "Input must be [B, T]"
        slen = self.shingling_len if shingle_len is None else shingle_len
        shop = self.shingling_hop if shingle_hop is None else shingle_hop

        h = get_frames(h, int(self.sr * slen), int(self.sr * shop), pad_mode="repeat")
        h = force_length(h, int(self.sr * self.minlen), dim=-1, pad_mode="repeat", allow_longer=True)

        s = h.size(1)
        h = rearrange(h, "b s t -> (b s) t")
        h = self.cqt(h)
        h = rearrange(h, "(b s) c t -> b s c t", s=s)
        return h

    def embed(self, h: torch.Tensor) -> torch.Tensor:
        """Take CQT tensor [B, S, C, T] and produce [B, S, zdim] embeddings."""
        assert h.ndim == 4, "CQT input must be [B, S, C, T]"
        s = h.size(1)
        h = rearrange(h, "b s c t -> (b s) 1 c t")
        h = self.frontend(h)
        h = self.backbone(h)
        h = self.pool(h)
        z = self.proj(h)
        z = rearrange(z, "(b s) c -> b s c", s=s)
        return z

    def forward(self, h: torch.Tensor, shingle_len=None, shingle_hop=None):
        """Forward inference path: prepare -> embed."""
        with torch.inference_mode():
            h = self.prepare(h, shingle_len=shingle_len, shingle_hop=shingle_hop)
        z = self.embed(h)
        return z