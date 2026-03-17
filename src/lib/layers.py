import sys
import torch
import math

###################################################################################################


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
        if self.noise:
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


###################################################################################################


class Linear(torch.nn.Module):

    def __init__(self, nin, nout, dim=1, bias=True):
        super().__init__()
        self.lin = torch.nn.Linear(nin, nout, bias=bias)
        self.dim = dim

    def forward(self, h):
        if self.dim != -1:
            h = h.transpose(self.dim, -1)
        h = self.lin(h)
        if self.dim != -1:
            h = h.transpose(self.dim, -1)
        return h


class PadConv2d(torch.nn.Module):

    def __init__(self, nin, nout, kern, stride=1, bias=True):
        super().__init__()
        assert kern % 2 == 1
        pad = kern // 2
        self.conv = torch.nn.Conv2d(
            nin, nout, kern, stride=stride, padding=pad, bias=bias
        )

    def forward(self, h):
        return self.conv(h)


###################################################################################################


class Squeeze(torch.nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        assert type(dim) == int or type(dim) == tuple
        self.dim = dim

    def forward(self, h):
        return torch.squeeze(h, dim=self.dim)


class Unsqueeze(torch.nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        assert type(dim) == int
        self.dim = dim

    def forward(self, h):
        return torch.unsqueeze(h, dim=self.dim)


###################################################################################################


class InstanceBatchNorm1d(torch.nn.Module):

    def __init__(self, ncha, affine=True):
        super().__init__()
        assert ncha % 2 == 0
        self.bn = torch.nn.BatchNorm1d(ncha // 2, affine=affine)
        self.inst = torch.nn.InstanceNorm1d(ncha // 2, affine=affine)

    def forward(self, h):
        h1, h2 = torch.chunk(h, 2, dim=1)
        h1 = self.bn(h1)
        h2 = self.inst(h2)
        h = torch.cat([h1, h2], dim=1)
        return h


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


###################################################################################################


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


class AutoPool(torch.nn.Module):

    def __init__(self, ncha=1, p_init=1):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=-1)
        self.p = torch.nn.Parameter(p_init * torch.ones(1, ncha, 1))

    def forward(self, h):
        h = self.flatten(h)
        a = torch.softmax(self.p * h, -1)
        return (h * a).sum(dim=-1)


class SoftPool(torch.nn.Module):

    def __init__(self, ncha):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=-1)
        self.lin = Linear(ncha, 2 * ncha, dim=1, bias=False)
        self.norm = torch.nn.InstanceNorm1d(ncha, affine=True)

    def forward(self, h):
        h = self.flatten(h)
        h = self.lin(h)
        h, a = torch.chunk(h, 2, dim=1)
        a = torch.softmax(self.norm(a), dim=-1)
        return (h * a).sum(dim=-1)


###################################################################################################


class ResNet50BottBlock(torch.nn.Module):

    def __init__(
        self,
        ncin,
        ncout,
        ncfactor=0.25,
        kern=3,
        stride=1,
        ibn=False,
        se=False,
    ):
        super().__init__()
        assert kern % 2 == 1
        pad = kern // 2
        ncmid = int(max(ncin, ncout) * ncfactor)
        if ncmid % 2 != 0:
            ncmid += 1
        tmp = [torch.nn.Conv2d(ncin, ncmid, 1, bias=False)]
        if ibn:
            tmp += [InstanceBatchNorm2d(ncmid)]
        else:
            tmp += [torch.nn.BatchNorm2d(ncmid)]
        tmp += [
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ncmid, ncmid, kern, stride=stride, padding=pad, bias=False),
            torch.nn.BatchNorm2d(ncmid),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ncmid, ncout, 1, bias=False),
            torch.nn.BatchNorm2d(ncout),
        ]
        if se:
            tmp += [SqueezeExcitation2d(ncout)]
        self.convs = torch.nn.Sequential(*tmp)
        if ncin != ncout or stride != 1:
            self.residual = torch.nn.Sequential(
                torch.nn.Conv2d(
                    ncin, ncout, kern, stride=stride, padding=pad, bias=False
                ),
                torch.nn.BatchNorm2d(ncout),
            )
        else:
            self.residual = torch.nn.Identity()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, h):
        return self.relu(self.convs(h) + self.residual(h))


###################################################################################################


class MyIBNResBlock(torch.nn.Module):

    def __init__(
        self,
        ncin,
        ncout,
        factor=0.5,
        kern=3,
        stride=1,
        ibn="pre",
        se="none",
    ):
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


###################################################################################################


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


###################################################################################################
