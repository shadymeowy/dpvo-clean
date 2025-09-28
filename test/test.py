import torch
import torch.nn as nn
from torch.export import Dim

DIM = 384


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


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


class Update(nn.Module):
    def __init__(self, p, dim=DIM):
        super(Update, self).__init__()
        self.dim = dim

        self.c1 = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )

        self.c2 = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim, eps=1e-3)

        self.agg_kk = SoftAgg(dim)
        self.agg_ij = SoftAgg(dim)

        self.gru = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
        )

        self.corr = nn.Sequential(
            nn.Linear(2 * 49 * p * p, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False), nn.Linear(dim, 2), GradientClip()
        )

        self.w = nn.Sequential(
            nn.ReLU(inplace=False), nn.Linear(dim, 2), GradientClip(), nn.Sigmoid()
        )

    def forward(self, net, inp, corr, nix, njx, ukk, ujk):
        # net_in = net
        """update operator"""
        net = net + inp + self.corr(corr)
        net = self.norm(net)

        mask_ix = (nix >= 0).float().reshape(1, -1, 1)
        mask_jx = (njx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:, nix])
        net = net + self.c2(mask_jx * net[:, njx])

        net = net + self.agg_kk(net, jx=ukk)
        net = net + self.agg_ij(net, jx=ujk)

        net = self.gru(net)

        # DEBUG
        # import safetensors.torch
        # dct = {
        #     "net_in": net_in.cpu(),
        #     "net_out": net.cpu(),
        #     "inp": inp.cpu(),
        #     "corr": corr.cpu(),
        #     "nix": nix.cpu(),
        #     "njx": njx.cpu(),
        #     "d": self.d(net).cpu(),
        #     "w": self.w(net).cpu(),
        # }
        # safetensors.torch.save_file(dct, "/tmp/debug_update.safetensors")
        # DEBUG END

        return net, self.d(net), self.w(net)


if __name__ == "__main__":
    import safetensors.torch

    update = Update(p=3).eval().to("cuda")
    update.load_state_dict(torch.load("update.pth"))

    dct = safetensors.torch.load_file("debug_update.safetensors", device="cuda")

    with torch.no_grad():
        net_out, d_out, w_out = update(
            dct["net_in"],
            dct["inp"],
            dct["corr"],
            dct["nix"],
            dct["njx"],
            dct["ukk"],
            dct["ujk"],
        )

    print((net_out - dct["net_out"]).abs().max(), net_out.abs().max())
    print((d_out - dct["d"]).abs().max(), d_out.abs().max())
    print((w_out - dct["w"]).abs().max(), w_out.abs().max())

    print("net_in", dct["net_in"].shape)
    print("inp", dct["inp"].shape)
    print("corr", dct["corr"].shape)
    print("nix", dct["nix"].shape)
    print("njx", dct["njx"].shape)
    print("ukk", dct["ukk"].shape)
    print("ujk", dct["ujk"].shape)
    print("net_out", net_out.shape)
    print("d", d_out.shape)
    print("w", w_out.shape)

    # # export to onnx
    # torch.onnx.export(
    #     update,
    #     (
    #         dct["net_in"],
    #         dct["inp"],
    #         dct["corr"],
    #         dct["nix"],
    #         dct["njx"],
    #         dct["ukk"],
    #         dct["ujk"],
    #     ),
    #     "update_trt.onnx",
    #     opset_version=18,
    #     dynamo=True,
    #     input_names=[
    #         "net",
    #         "inp",
    #         "corr",
    #         "nix",
    #         "njx",
    #         "ukk",
    #         "ujk",
    #     ],
    #     output_names=["net_out", "d", "w"],
    #     dynamic_shapes={
    #         # inputs
    #         "net": {1: Dim("N")},  # [1, N, 384]
    #         "inp": {1: Dim("N")},  # [1, N, 384]
    #         "corr": {1: Dim("N")},  # [1, N, 882]
    #         "nix": {0: Dim("N")},  # [N]
    #         "njx": {0: Dim("N")},  # [N]
    #         "ukk": {0: Dim("N")},  # [N]
    #         "ujk": {0: Dim("N")},  # [N]
    #         # outputs
    #         # "net_out": {1: Dim("N")},  # [1, N, 384]
    #         # "d": {1: Dim("N")},  # [1, N, 2]
    #         # "w": {1: Dim("N")},  # [1, N, 2]
    #     },
    #     external_data=False,
    # )

    # benchmark pytorch version
    import time

    torch.cuda.synchronize()
    t0 = time.time()

    niter = 100
    for _ in range(niter):
        with torch.no_grad():
            net_out, d_out, w_out = update(
                dct["net_in"],
                dct["inp"],
                dct["corr"],
                dct["nix"],
                dct["njx"],
                dct["ukk"],
                dct["ujk"],
            )
        torch.cuda.synchronize()
    t1 = time.time()

    print("pytorch:", (t1 - t0) / niter * 1000, "ms")

    from trt_bench import trt_benchmark_same_inputs

    # make sure all tensors are CUDA + contiguous
    for k in ["net_in", "inp", "corr", "nix", "njx", "ukk", "ujk"]:
        dct[k] = dct[k].to("cuda").contiguous()

    # (Optional) If you built an FP16 engine, cast model inputs accordingly before export/run
    # dct["net_in"] = dct["net_in"].half(); dct["inp"] = dct["inp"].half(); dct["corr"] = dct["corr"].half()

    trt_ms = trt_benchmark_same_inputs(
        update,
        dct,
        onnx_path="update_trt.onnx",
        plan_path="update_trt.plan",
        fp16=False,  # set True to try FP16 engine
        workspace_mb=4096,
        n_warm=20,
        n_iter=100,
    )
