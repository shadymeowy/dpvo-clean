import torch
import torch.nn as nn
import torch_scatter


class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, *, ix=None, jx=None):
        if ix is None and jx is None:
            raise ValueError("provide one of ix or jx")
        if ix is not None and jx is not None:
            raise ValueError("provide only one of ix or jx")

        if jx is None:
            _, jx = torch.unique(ix, return_inverse=True)

        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:, jx]

        return self.h(y)


class SoftAggONNX(nn.Module):
    def __init__(self, dim=512, expand=True):
        super().__init__()
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)
        self.expand = expand

        if not expand:
            raise ValueError("ONNX export only supports expand=True")

    def forward(self, x, *, ix=None, jx=None):
        if x.dim() != 2:
            if x.dim() == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise ValueError("Only batch size 1 supported for ONNX export")

        if ix is None and jx is None:
            raise ValueError("provide one of ix or jx")
        if ix is not None and jx is not None:
            raise ValueError("provide only one of ix or jx")

        if jx is None:
            _, jx = torch.unique(ix, return_inverse=True)  # [N]

        N, D = x.shape
        G = jx.max(dim=0).values + 1

        logits = self.g(x)  # [N, D]
        feats = self.f(x)  # [N, D]
        idx = jx.view(-1, 1).expand(N, D)  # [N, D]

        gmax = torch.full(
            (G, D), -float("inf"), dtype=logits.dtype, device=logits.device
        )
        gmax.scatter_reduce_(0, idx, logits, reduce="amax", include_self=True)

        centered = logits - gmax.index_select(0, jx)
        expv = torch.exp(centered)

        denom = torch.zeros((G, D), dtype=logits.dtype, device=logits.device)
        denom.scatter_reduce_(0, idx, expv, reduce="sum", include_self=True)

        w = expv / (denom.index_select(0, jx) + 1e-12)

        y = torch.zeros((G, D), dtype=logits.dtype, device=logits.device)
        y.scatter_reduce_(0, idx, feats * w, reduce="sum", include_self=True)

        out_group = self.h(y)
        return out_group.index_select(0, jx)


SoftAgg = SoftAggONNX


class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, 1)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:, jx]

        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP = 0.1


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)


class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(
            torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x
        )
        return grad_x


class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.abs().mean())
        return grad_x
