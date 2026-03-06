import torch
import cuda_voxel


def to_voxel_grid_cuda(voxel, xs, ys, ts, ps, Nbins=5):
    """Fill pre-allocated voxel buffer with trilinear event accumulation.

    Args:
        voxel: (Nbins+1, H, W) float32 CUDA tensor (pre-allocated, will be zeroed)
        xs, ys, ts, ps: (N,) float CUDA tensors
        Nbins: number of time bins
    """
    if not isinstance(xs, torch.Tensor):
        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()
        ts = torch.from_numpy(ts).float()
        ps = torch.from_numpy(ps).float()

    cuda_voxel.forward(
        voxel,
        xs.float().cuda().contiguous(),
        ys.float().cuda().contiguous(),
        ts.float().cuda().contiguous(),
        ps.float().cuda().contiguous(),
        Nbins,
    )
