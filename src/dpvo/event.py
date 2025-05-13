import math

import cv2
import numpy as np
from numba import cuda, jit


def compute_remap(K, distortion, W, H):
    K, _ = cv2.getOptimalNewCameraMatrix(
        K, distortion, (W, H), alpha=0, newImgSize=(W, H)
    )
    coords = (
        np.stack(np.meshgrid(np.arange(W), np.arange(H)))
        .reshape((2, -1))
        .astype("float32")
    )
    term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
    points = cv2.undistortPointsIter(
        coords, K, distortion, np.eye(3), K, criteria=term_criteria
    )
    rectify_map = points.reshape((H, W, 2))
    # Make out of bounds points to be -1, -1
    mask = (
        (rectify_map[..., 0] < 0)
        | (rectify_map[..., 0] >= W - 1)
        | (rectify_map[..., 1] < 0)
        | (rectify_map[..., 1] >= H - 1)
    )
    rectify_map[mask] = -1

    return K, rectify_map


def voxel_to_img(voxel):
    img = voxel[-1]
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


@jit(
    nopython=True,
    fastmath=True,
    nogil=True,
    cache=True,
)
def to_voxel_grid(voxel, xs, ys, ts, ps, H=480, W=640, nb_of_time_bins=5):
    duration = ts[-1] - ts[0]
    start_timestamp = ts[0]

    HW = H * W

    for i in range(HW * (nb_of_time_bins + 1)):
        voxel[i] = 0.0

    alpha = (nb_of_time_bins - 1) / duration

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        t = (ts[i] - start_timestamp) * alpha
        p = -1 if ps[i] == 0 else 1

        lim_x = np.int32(x)
        lim_y = np.int32(y)
        lim_t = np.int32(t)
        dx = lim_x - x
        dy = lim_y - y
        dt = lim_t - t
        if (
            lim_x < 0
            or lim_y < 0
            or lim_t < 0
            or lim_x >= W - 1
            or lim_y >= H - 1
            or lim_t >= nb_of_time_bins
        ):
            continue

        dxy = dx * dy
        dyt = dy * dt
        dxt = dt * dx

        v000 = dxy * dt  # dx * dy * dt
        v001 = dxy + v000  # dx * dy * (1 + dt)
        v010 = dxt + v000  # dx * (1 + dy) * dt
        v100 = dyt + v000  # (1 + dx) * dy * dt
        v011 = v001 + dx + dxt  # dx * (1 + dy) * (1 + dt)
        tt = dy + dyt
        v101 = v001 + tt  # (1 + dx) * dy * (1 + dt)
        v110 = v010 + dt + dyt  # (1 + dx) * (1 + dy) * dt
        v111 = v011 + 1 + dt + tt  # (1 + dx) * (1 + dy) * (1 + dt)

        idx = lim_t * HW + lim_y * W + lim_x
        voxel[idx] += p * v111
        voxel[idx + 1] -= p * v011
        voxel[idx + W] -= p * v101
        voxel[idx + W + 1] += p * v001
        voxel[idx + HW] -= p * v110
        voxel[idx + HW + 1] += p * v010
        voxel[idx + HW + W] += p * v100
        voxel[idx + HW + W + 1] -= p * v000
    return voxel


@cuda.jit(
    fastmath=True,
    cache=True,
)
def trilinear_kernel_event(voxel, xs, ys, ts, ps, W, H, t0, alpha, Nbins):
    idx = cuda.grid(1)
    if idx >= len(xs):
        return
    x = xs[idx]
    y = ys[idx]
    t = ts[idx]
    p = ps[idx]

    t = (t - t0) * alpha
    p = -1 if p == 0 else 1

    lim_x = np.int32(x)
    lim_y = np.int32(y)
    lim_t = np.int32(t)
    dx = lim_x - x
    dy = lim_y - y
    dt = lim_t - t
    if (
        lim_x < 0
        or lim_y < 0
        or lim_t < 0
        or lim_x >= W - 1
        or lim_y >= H - 1
        or lim_t >= Nbins
    ):
        return

    dxy = dx * dy
    dyt = dy * dt
    dxt = dt * dx

    v000 = dxy * dt  # dx * dy * dt
    v001 = dxy + v000  # dx * dy * (1 + dt)
    v010 = dxt + v000  # dx * (1 + dy) * dt
    v100 = dyt + v000  # (1 + dx) * dy * dt
    v011 = v001 + dx + dxt  # dx * (1 + dy) * (1 + dt)
    tt = dy + dyt
    v101 = v001 + tt  # (1 + dx) * dy * (1 + dt)
    v110 = v010 + dt + dyt  # (1 + dx) * (1 + dy) * dt
    v111 = v011 + 1 + dt + tt  # (1 + dx) * (1 + dy) * (1 + dt)

    # Atomic add
    HW = H * W
    idx = lim_t * HW + lim_y * W + lim_x
    cuda.atomic.add(voxel, idx, p * v111)
    cuda.atomic.add(voxel, idx + 1, -p * v011)
    cuda.atomic.add(voxel, idx + W, -p * v101)
    cuda.atomic.add(voxel, idx + W + 1, p * v001)
    cuda.atomic.add(voxel, idx + HW, -p * v110)
    cuda.atomic.add(voxel, idx + HW + 1, p * v010)
    cuda.atomic.add(voxel, idx + HW + W, p * v100)
    cuda.atomic.add(voxel, idx + HW + W + 1, -p * v000)


def to_voxel_grid_cuda(xs, ys, ts, ps, H=480, W=640, Nbins=5):
    alpha = (Nbins - 1) / (ts[-1] - ts[0])
    voxel = np.zeros((Nbins + 1, H, W), dtype=np.float32)

    threadsperblock = 512
    blockspergrid = math.ceil(len(xs) / threadsperblock)
    t0 = ts[0]
    trilinear_kernel_event[blockspergrid, threadsperblock](
        voxel.ravel(), xs, ys, ts, ps, W, H, t0, alpha, Nbins
    )
    return voxel[:-1]


@jit(nopython=True)
def get_time_indices_offsets(
    time_array: np.ndarray, time_start_us: int, time_end_us: int
) -> tuple:
    assert time_array.ndim == 1

    idx_start = -1
    if time_array[-1] < time_start_us:
        return time_array.size, time_array.size

    for idx_from_start in range(0, time_array.size, 1):
        if time_array[idx_from_start] >= time_start_us:
            idx_start = idx_from_start
            break
    assert idx_start >= 0

    idx_end = time_array.size
    for idx_from_end in range(time_array.size - 1, -1, -1):
        if time_array[idx_from_end] < time_end_us:
            break
        idx_end = idx_from_end

    return idx_start, idx_end


@jit(nopython=True)
def accumulate_events(frame, xs, ys, ts, ps, alpha=0.05, tau=50.0):
    H, W = frame.shape
    t0 = ts[0]
    t1 = ts[-1]

    alpha = 1e6 / ((t1 - t0) * tau)

    # Decay existing event frame
    # tmp = np.exp(-alpha)
    for i in range(H):
        for j in range(W):
            frame[i, j] = 0

    # Accumulate new events
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        t = ts[i]
        p = -1 if ps[i] == 0 else 1

        if x < 0 or y < 0 or x >= W or y >= H:
            continue

        frame[y, x] += p * alpha * np.exp(-alpha * (t - t0) / 1e6)

    # Clamp values to [0, 1]
    for i in range(H):
        for j in range(W):
            frame[i, j] = min(max(frame[i, j], 0), 1)
