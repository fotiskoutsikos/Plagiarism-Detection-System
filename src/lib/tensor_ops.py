import sys
import torch, math
from einops import rearrange

###################################################################################################


def tensor_quantile(x, q, dim=-1, keepdim=False):
    assert x.ndim == q.ndim
    qn = (q.clamp(min=0, max=1) * (x.size(dim) - 1)).round().long()
    sx = x.sort(dim=dim)[0]
    xq = torch.gather(sx, dim, qn)
    if keepdim:
        return xq
    return xq.squeeze(dim)


###################################################################################################


def debug_inf_nan(ten, txt):
    if torch.isnan(ten).float().sum() > 0:
        print()
        print("nan " + txt)
        sys.exit()
    if torch.isinf(ten).float().sum() > 0:
        print()
        print("inf " + txt)
        sys.exit()


###################################################################################################


def force_length(
    x, length, dim=-1, pad_mode="repeat", cut_mode="start", allow_longer=False
):
    assert pad_mode in ("repeat", "zeros", "crazy")
    assert cut_mode in ("start", "end", "random")
    # fast bypass
    if x.size(dim) == length or (x.size(dim) > length and allow_longer):
        return x
    # do otherwise
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


###################################################################################################


def frames(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    if pad_end:
        signal_length = signal.shape[axis]
        frames_overlap = frame_length - frame_step
        rest_samples = abs(signal_length - frames_overlap) % abs(frame_step)
        if rest_samples != 0:
            pad_size = int(frame_length - rest_samples)
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = torch.nn.functional.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames


def get_frames(
    x, length, step, dim=-1, pad_end=True, pad_mode="zeros", cut_mode="start"
):
    if pad_end:
        newlength = (
            max(int(math.ceil((x.size(dim) - length) / step)), 0) * step + length
        )
        x = force_length(
            x,
            newlength,
            dim=dim,
            pad_mode=pad_mode,
            cut_mode=cut_mode,
            allow_longer=False,
        )
    return x.unfold(dim, length, step)


###################################################################################################


def covariance(x, eps=1e-6):
    xx = x - x.mean(0, keepdim=True)
    cov = torch.matmul(xx.T, xx) / (len(xx) - 1)
    weight = torch.triu(torch.ones_like(cov), diagonal=1)
    cov = (weight * cov.pow(2)).sum() / (weight.sum() + eps)
    return cov


###################################################################################################


def roughly_equal(x, y, tol=1e-6):
    return (x - y).abs() < tol


###################################################################################################


def pairwise_euclidean_distance_matrix(x, y, squared=False, eps=1e-6):
    squared_x = x.pow(2).sum(1).view(-1, 1)
    squared_y = y.pow(2).sum(1).view(1, -1)
    dot_product = torch.mm(x, y.t())
    distance_matrix = squared_x - 2 * dot_product + squared_y
    # get rid of negative distances due to numerical instabilities
    distance_matrix[distance_matrix <= 0] = 0
    if not squared:
        # handle numerical stability
        # derivative of the square root operation applied to 0 is infinite
        # we need to handle by setting any 0 to eps
        mask = (distance_matrix == 0.0).type_as(distance_matrix)
        # use this mask to set indices with a value of 0 to eps
        distance_matrix += mask * eps
        # now it is safe to get the square root
        distance_matrix = torch.sqrt(distance_matrix)
        # undo the trick for numerical stability
        distance_matrix *= 1.0 - mask
    return distance_matrix


def pairwise_distance_matrix(x, y, mode="fro", p=2, eps=1e-6):
    assert x.ndim == y.ndim and x.ndim <= 2
    if x.ndim == 1:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    if mode == "euc" or mode == "neuc":
        p = 2
    if mode in ("fro", "nfro", "euc", "neuc"):
        dist = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=p).squeeze(0)
        if mode == "nfro" or mode == "neuc":
            dist = dist / (x.size(-1) ** (1 / p))
    elif mode in ("sqeuc", "nsqeuc"):
        dist = pairwise_euclidean_distance_matrix(x, y, squared=True)
        if mode == "nsqeuc":
            dist = dist / x.size(-1)
    elif mode in ("cos", "cossim", "dot", "dotsim"):
        if mode == "cos" or mode == "cossim":
            x = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
            y = y / (torch.norm(y, dim=-1, keepdim=True) + eps)
        dist = torch.matmul(x, y.T)
        if mode == "cos" or mode == "dot":
            dist = 1 - dist
    else:
        raise NotImplementedError
    return dist


###################################################################################################


def msum(x, mask=None, dim=None, keepdim=False):
    if mask is None:
        included = torch.ones_like(x)
    else:
        included = (~mask).type_as(x)
    if dim is None:
        sum = (included * x).sum()
        if keepdim:
            while sum.ndim < x.ndim:
                sum = sum.unsqueeze(0)
    else:
        sum = (included * x).sum(dim=dim, keepdim=keepdim)
    return sum


def mmean(x, mask=None, dim=None, keepdim=False, eps=1e-7):
    if mask is None:
        included = torch.ones_like(x)
    else:
        included = (~mask).type_as(x)
    if dim is None:
        num = (included * x).sum()
        den = included.sum()
        if keepdim:
            while num.ndim < x.ndim:
                num = num.unsqueeze(0)
                den = den.unsqueeze(0)
    else:
        num = (included * x).sum(dim=dim, keepdim=keepdim)
        den = included.sum(dim=dim, keepdim=keepdim)
    return num / den.clamp(min=eps)


def mmin(x, mask=None, dim=None, keepdim=False, ctt=torch.inf):
    if mask is None:
        tmp = x
    else:
        tmp = torch.where(mask, ctt, x)
    if dim is None:
        tmp = tmp.min()
        if keepdim:
            while tmp.ndim < x.ndim:
                tmp = tmp.unsqueeze(0)
    else:
        if type(dim) == int:
            dim = [dim]
        else:
            dim = list(dim)
        for d in dim:
            tmp = tmp.min(d, keepdim=True)[0]
        if not keepdim:
            for d in dim:
                tmp = tmp.squeeze(d)
    return tmp


def mmax(x, mask=None, dim=None, keepdim=False, ctt=-torch.inf):
    if mask is None:
        tmp = x
    else:
        tmp = torch.where(mask, ctt, x)
    if dim is None:
        tmp = tmp.max()
        if keepdim:
            while tmp.ndim < x.ndim:
                tmp = tmp.unsqueeze(0)
    else:
        if type(dim) == int:
            dim = [dim]
        else:
            dim = list(dim)
        for d in dim:
            tmp = tmp.max(d, keepdim=True)[0]
        if not keepdim:
            for d in dim:
                tmp = tmp.squeeze(d)
    return tmp


def mrand(x, mask=None, dim=None, keepdim=False, ctt=torch.inf, eps=1e-7):
    r = torch.rand_like(x)
    if mask is not None:
        r = torch.where(mask, ctt, r)
    mr = r > mmin(r, mask=mask, dim=dim, keepdim=True, ctt=ctt)
    return mmean(x, mask=mr, dim=dim, keepdim=keepdim, eps=eps)


def mbest(x, k, mask=None, dim=None, keepdim=False, ctt=torch.inf, eps=1e-7):
    assert type(dim) == int
    if mask is not None:
        x = torch.where(mask, ctt, x)
    x = x.topk(k, dim=dim, largest=False)[0]
    return mmean(x, mask=x >= ctt, dim=dim, keepdim=keepdim, eps=eps)


def mworst(x, k, mask=None, dim=None, keepdim=False, ctt=-torch.inf, eps=1e-7):
    assert type(dim) == int
    if mask is not None:
        x = torch.where(mask, ctt, x)
    x = x.topk(k, dim=dim, largest=True)[0]
    return mmean(x, mask=x >= ctt, dim=dim, keepdim=keepdim, eps=eps)


###################################################################################################


def distance_tensor_redux(dist, redux, mask=None, squeeze=True, eps=1e-7, inf=1e12):
    # Expects dist shape to be (b1,b2,s1,s2)
    # Reduces last two dims
    if redux == "min":
        dist = mmin(dist, mask=mask, dim=(-1, -2), keepdim=True, ctt=inf)
    elif redux == "max":
        dist = mmax(dist, mask=mask, dim=(-1, -2), keepdim=True, ctt=-inf)
    elif redux == "mean":
        dist = mmean(dist, mask=mask, dim=(-1, -2), keepdim=True, eps=eps)
    elif redux == "minmean":
        dist = mmean(dist, mask=mask, dim=-1, keepdim=True, eps=eps)
        dist = mmin(dist, mask=mask, dim=(-1, -2), keepdim=True, ctt=inf)
    elif redux == "meanmin":
        dist = mmin(dist, mask=mask, dim=-1, keepdim=True, ctt=inf)
        dist = mmean(dist, mask=mask, dim=(-1, -2), keepdim=True, eps=eps)
    elif redux == "randmin":
        dist = mmin(dist, mask=mask, dim=-1, keepdim=True, ctt=inf)
        dist = mrand(dist, mask=mask, dim=(-1, -2), keepdim=True, ctt=inf, eps=eps)
    elif redux.startswith("bpwr"):  # best pairs without replacement
        # transpose if smaller
        if dist.size(3) < dist.size(2):
            dist = dist.transpose(2, 3)
            if mask is not None:
                mask = mask.transpose(2, 3)
        # set max iters
        if "-" not in redux:
            n = dist.size(2)
        else:
            n = max(1, min(int(redux.split("-")[-1]), dist.size(2)))
        # try to avoid ties
        dist = dist + eps * torch.rand_like(dist)
        # init
        if mask is None:
            mask = dist > inf
        all_sel = dist > inf
        # loop
        for i in range(n):
            mn = mmin(dist, mask=mask, dim=(-1, -2), keepdim=True, ctt=inf)
            sel = (dist <= mn) & (~mask)
            all_sel = all_sel | sel
            if i < n - 1:
                mask = (
                    mask
                    | (mmin(dist, mask=mask, dim=-1, keepdim=True, ctt=inf) <= mn)
                    | (mmin(dist, mask=mask, dim=-2, keepdim=True, ctt=inf) <= mn)
                )
        # average
        dist = mmean(dist, mask=(~all_sel), dim=(-1, -2), keepdim=True, eps=eps)
    elif redux.startswith("best"):
        if "-" not in redux:
            k = 1
        else:
            k = max(1, min(int(redux.split("-")[-1]), dist.size(2) * dist.size(3)))
        dist = rearrange(dist, "b1 b2 s1 s2 -> b1 b2 1 (s1 s2)")
        if mask is not None:
            mask = rearrange(mask, "b1 b2 s1 s2 -> b1 b2 1 (s1 s2)")
        dist = mbest(dist, k, mask=mask, dim=-1, keepdim=True, ctt=inf, eps=eps)
    elif redux.startswith("worst"):
        if "-" not in redux:
            k = 1
        else:
            k = max(1, min(int(redux.split("-")[-1]), dist.size(2) * dist.size(3)))
        dist = rearrange(dist, "b1 b2 s1 s2 -> b1 b2 1 (s1 s2)")
        if mask is not None:
            mask = rearrange(mask, "b1 b2 s1 s2 -> b1 b2 1 (s1 s2)")
        dist = mworst(dist, k, mask=mask, dim=-1, keepdim=True, ctt=-inf, eps=eps)
    elif redux.startswith("bestmin"):
        if "-" not in redux:
            k = 1
        else:
            k = max(1, min(int(redux.split("-")[-1]), dist.size(2)))
        dist = mmin(dist, mask=mask, dim=-1, keepdim=True, ctt=inf)
        dist = mbest(dist, k, mask=mask, dim=(-1, -2), keepdim=True, ctt=inf, eps=eps)
    elif redux[0] == "s":
        aux1 = distance_tensor_redux(dist, redux[1:], mask=mask, squeeze=False)
        dist = dist.transpose(2, 3)
        if mask is not None:
            mask = mask.transpose(2, 3)
        aux2 = distance_tensor_redux(dist, redux[1:], mask=mask, squeeze=False)
        aux2 = aux2.transpose(2, 3)
        dist = 0.5 * (aux1 + aux2)
    else:
        raise NotImplementedError
    if squeeze:
        dist = dist.squeeze((-1, -2))
    return dist


###################################################################################################
