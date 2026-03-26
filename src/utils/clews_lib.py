import math
import torch
from einops import rearrange
from nnAudio import features


# TENSOR OPS
def force_length(x, length, dim=-1, pad_mode="repeat", cut_mode="start", allow_longer=False):
    assert pad_mode in ("repeat", "zeros", "crazy")
    assert cut_mode in ("start", "end", "random")
    
    if x.size(dim) == length or (x.size(dim) > length and allow_longer):
        return x
        
    aux = x.clone()
    while aux.size(dim) < length:
        if pad_mode == "repeat":
            aux = torch.cat([aux, x], dim=dim)
        elif pad_mode == "zeros":
            aux = torch.cat([aux, torch.zeros_like(x)], dim=dim)
        elif pad_mode == "crazy":
            r = torch.randint(0, 4, (1,)).item()
            if r == 0:
                aux = torch.cat([aux, x], dim=dim)
            elif r == 1:
                aux = torch.cat([x, aux], dim=dim)
            elif r == 2:
                aux = torch.cat([aux, torch.zeros_like(x)], dim=dim)
            elif r == 3:
                aux = torch.cat([torch.zeros_like(x), aux], dim=dim)
                
    if not allow_longer and aux.size(-1) > length:
        if dim != -1:
            aux = aux.transpose(dim, -1)
        if cut_mode == "start":
            aux = aux[..., :length]
        elif cut_mode == "end":
            aux = aux[..., -length:]
        elif cut_mode == "random":
            r = torch.randint(0, aux.size(-1) - length + 1, (1,)).item()
            aux = aux[..., r : r + length]
        if dim != -1:
            aux = aux.transpose(-1, dim)
            
    return aux

def get_frames(x, length, step, dim=-1, pad_end=True, pad_mode="zeros", cut_mode="start"):
    if pad_end:
        newlength = max(int(math.ceil((x.size(dim) - length) / step)), 0) * step + length
        x = force_length(x, newlength, dim=dim, pad_mode=pad_mode, cut_mode=cut_mode, allow_longer=False)
    return x.unfold(dim, length, step)


# LAYERS
class CQTPrepare(torch.nn.Module):
    def __init__(self, pow=0.5, norm="max2d", noise=True, affine=True, eps=1e-6):
        super().__init__()
        assert norm in ("max1d", "max2d", "mean2d")
        self.pow = pow
        self.norm = norm
        self.noise = noise
        self.affine = affine
        if self.affine:
            self.gain = torch.nn.Parameter(torch.ones(1))
            self.bias = torch.nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, h):
        h = h.clamp(min=0).pow(self.pow)
        h = self.normalize(h)
        if self.noise and self.training:
            h = h + self.eps * torch.rand_like(h)
            h = self.normalize(h)
        if self.affine:
            h = self.gain * h + self.bias
        return h

    def normalize(self, h):
        h = h - h.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
        if self.norm == "max2d":
            h = h / (h.max(2, keepdim=True)[0].max(3, keepdim=True)[0] + self.eps)
        elif self.norm == "max1d":
            h = h / (h.max(2, keepdim=True)[0] + self.eps)
        elif self.norm == "mean2d":
            h = h / (h.mean((2, 3), keepdim=True) + self.eps)
        return h

class PadConv2d(torch.nn.Module):
    def __init__(self, nin, nout, kern, stride=1, bias=True):
        super().__init__()
        assert kern % 2 == 1
        pad = kern // 2
        self.conv = torch.nn.Conv2d(nin, nout, kern, stride=stride, padding=pad, bias=bias)

    def forward(self, h):
        return self.conv(h)

class InstanceBatchNorm2d(torch.nn.Module):
    def __init__(self, ncha, affine=True):
        super().__init__()
        assert ncha % 2 == 0
        self.bn = torch.nn.BatchNorm2d(ncha // 2, affine=affine)
        self.inst = torch.nn.InstanceNorm2d(ncha // 2, affine=affine)

    def forward(self, h):
        h1, h2 = torch.chunk(h, 2, dim=1)
        h1 = self.bn(h1)
        h2 = self.inst(h2)
        h = torch.cat([h1, h2], dim=1)
        return h

class SqueezeExcitation2d(torch.nn.Module):
    def __init__(self, ncha, r=2):
        super().__init__()
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        nmid = max(1, int(ncha / r))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(ncha, nmid, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nmid, ncha, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, h):
        s = self.pooling(h).transpose(1, -1)
        s = self.mlp(s).transpose(-1, 1)
        return h * s

class MyIBNResBlock(torch.nn.Module):
    def __init__(self, ncin, ncout, factor=0.5, kern=3, stride=1, ibn="pre", se="none"):
        super().__init__()
        ncmid = max(1, int(max(ncin, ncout) * factor))
        ncmid += ncmid % 2
        tmp = []
        if ibn == "pre":
            tmp += [InstanceBatchNorm2d(ncin)]
        else:
            tmp += [torch.nn.BatchNorm2d(ncin)]
        if se == "pre":
            tmp += [SqueezeExcitation2d(ncin)]
            
        tmp += [
            torch.nn.ReLU(inplace=True),
            PadConv2d(ncin, ncmid, kern, stride=stride, bias=False),
        ]
        
        if ibn == "post":
            tmp += [InstanceBatchNorm2d(ncmid)]
        else:
            tmp += [torch.nn.BatchNorm2d(ncmid)]
            
        tmp += [
            torch.nn.ReLU(inplace=True),
            PadConv2d(ncmid, ncout, kern, bias=False),
        ]
        
        if se == "post":
            tmp += [SqueezeExcitation2d(ncout)]
            
        self.convs = torch.nn.Sequential(*tmp)
        
        if ncin != ncout or stride != 1:
            self.skip = torch.nn.Sequential(
                torch.nn.BatchNorm2d(ncin),
                torch.nn.ReLU(inplace=True),
                PadConv2d(ncin, ncout, kern, stride=stride, bias=False),
            )
        else:
            self.skip = torch.nn.Identity()
            
        self.gain = torch.nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return self.gain * self.convs(h) + self.skip(h)

class GeMPool(torch.nn.Module):
    def __init__(self, ncha=1, init=3, eps=1e-6):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=-1)
        self.softplus = torch.nn.Softplus()
        pinit = math.log(math.exp(init - 1) - 1)
        self.p = torch.nn.Parameter(pinit * torch.ones(1, ncha, 1))
        self.eps = eps

    def forward(self, h):
        h = self.flatten(h)
        pow = 1 + self.softplus(self.p)
        h = h.clamp(min=self.eps).pow(pow)
        h = h.mean(-1).pow(1 / pow.squeeze(-1))
        return h


# MAIN MODEL
class Model(torch.nn.Module):
    def __init__(self, conf, sr=16000, eps=1e-6):
        super().__init__()
        self.sr = sr
        self.eps = eps
        
        # Shingling
        self.shingling_len = conf.shingling.len
        self.shingling_hop = conf.shingling.hop
        self.minlen = self.shingling_len 
        
        # CQT
        self.cqt = torch.nn.Sequential(
            features.CQT1992v2(
                sr=self.sr,
                hop_length=int(conf.cqt.hoplen * sr),
                n_bins=int(conf.cqt.noctaves * conf.cqt.nbinsoct),
                bins_per_octave=int(conf.cqt.nbinsoct),
                filter_scale=float(conf.cqt.fscale),
                trainable=False,
                verbose=False,
            ),
            torch.nn.AvgPool1d(int(conf.cqt.pool), stride=int(conf.cqt.pool)),
        )
        
        # Model - Frontend
        ncha0, ncha = conf.frontend.channels
        self.frontend = torch.nn.Sequential(
            CQTPrepare(pow=float(conf.frontend.cqtpow)),
            torch.nn.Conv2d(1, ncha0, (12, 3), stride=(1, 2), bias=False),
            torch.nn.BatchNorm2d(ncha0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ncha0, ncha, (12, 3), stride=2, bias=False),
        )
        
        # Model - Backbone
        aux = []
        for nb, nc, st in zip(conf.backbone.blocks, conf.backbone.channels, conf.backbone.down):
            aux += [MyIBNResBlock(ncha, nc, stride=st)]
            for _ in range(nb - 1):
                aux += [MyIBNResBlock(nc, nc)]
            ncha = nc
        self.backbone = torch.nn.Sequential(*aux)
        
        # Pooling & projection
        self.pool = GeMPool()
        self.proj = torch.nn.Sequential(
            torch.nn.BatchNorm1d(ncha),
            torch.nn.Linear(ncha, int(conf.zdim), bias=False),
        )

    def prepare(self, h, shingle_len=None, shingle_hop=None):
        assert h.ndim == 2
        slen = self.shingling_len if shingle_len is None else shingle_len
        shop = self.shingling_hop if shingle_hop is None else shingle_hop
        
        h = get_frames(h, int(self.sr * slen), int(self.sr * shop), pad_mode="repeat")
        h = force_length(h, int(self.sr * self.minlen), dim=-1, pad_mode="repeat", allow_longer=True)
        
        s = h.size(1)
        h = rearrange(h, "b s t -> (b s) t")
        h = self.cqt(h)
        h = rearrange(h, "(b s) c t -> b s c t", s=s)
        return h

    def embed(self, h):
        assert h.ndim == 4
        s = h.size(1)
        h = rearrange(h, "b s c t -> (b s) 1 c t")
        h = self.frontend(h)
        h = self.backbone(h)
        h = self.pool(h)
        z = self.proj(h)
        z = rearrange(z, "(b s) c -> b s c", s=s)
        return z, None 

    def forward(self, h, shingle_len=None, shingle_hop=None):
        with torch.inference_mode():
            h = self.prepare(h, shingle_len=shingle_len, shingle_hop=shingle_hop)
            z, _ = self.embed(h)
        return z