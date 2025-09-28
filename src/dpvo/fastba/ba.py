import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject
reproject_s = cuda_ba.reproject_s


def BA(
    poses,
    patches,
    intrinsics,
    intrinsics_s,
    extrinsics,
    target,
    weight,
    target_s,
    weight_s,
    lmbda,
    ii,
    jj,
    kk,
    t0,
    t1,
    M,
    iterations,
    eff_impl=False,
    stereo=False,
):
    return cuda_ba.forward(
        poses.data,
        patches,
        intrinsics,
        intrinsics_s,
        extrinsics,
        target,
        weight,
        target_s,
        weight_s,
        lmbda,
        ii,
        jj,
        kk,
        M,
        t0,
        t1,
        iterations,
        eff_impl,
        stereo,
    )
