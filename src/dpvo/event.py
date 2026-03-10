import cv2
import numpy as np
from numba import jit
from scipy.spatial.transform import Rotation as R


def compute_remap(intr, dist, W, H, fisheye=False):
    K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]])
    K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=0, newImgSize=(W, H))

    if fisheye:
        coords = (
            np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)
            .reshape(-1, 2)
            .astype("float32")
        )
        coords = coords.reshape(-1, 1, 2)
        pts = cv2.fisheye.undistortPoints(coords, K, dist, P=K)
    else:
        coords = (
            np.stack(np.meshgrid(np.arange(W), np.arange(H)))
            .reshape((2, -1))
            .astype("float32")
        )
        term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        pts = cv2.undistortPointsIter(
            coords, K, dist, np.eye(3), K, criteria=term_criteria
        )

    rect_map = pts.reshape((H, W, 2))
    # Make out of bounds points to be -1, -1
    mask = (
        (rect_map[..., 0] < 0)
        | (rect_map[..., 0] >= W - 1)
        | (rect_map[..., 1] < 0)
        | (rect_map[..., 1] >= H - 1)
    )
    rect_map[mask] = -1

    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    return intr, rect_map


def compute_stereo_rect_remap(intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, fisheye):
    K_l = np.array([[intr_l[0], 0, intr_l[2]], [0, intr_l[1], intr_l[3]], [0, 0, 1]])
    K_r = np.array([[intr_r[0], 0, intr_r[2]], [0, intr_r[1], intr_r[3]], [0, 0, 1]])

    T_l2r = T_l @ np.linalg.inv(T_r)
    R_ext, t_ext = T_l2r[:3, :3], T_l2r[:3, 3]

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_l,
        dist_l,
        K_r,
        dist_r,
        (W, H),
        R_ext,
        t_ext,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    if fisheye:
        coords = (
            np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)
            .reshape(-1, 2)
            .astype("float32")
        )
        coords = coords.reshape(-1, 1, 2)

        pts_l = cv2.fisheye.undistortPoints(coords, K_l, dist_l, P=P1, R=R1)
        pts_r = cv2.fisheye.undistortPoints(coords, K_r, dist_r, P=P2, R=R2)
    else:
        term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        coords = (
            np.stack(np.meshgrid(np.arange(W), np.arange(H)))
            .reshape(2, -1)
            .astype("float32")
        )

        pts_l = cv2.undistortPointsIter(
            coords, K_l, dist_l, R1, P1[:3, :3], criteria=term_criteria
        )
        pts_r = cv2.undistortPointsIter(
            coords, K_r, dist_r, R2, P2[:3, :3], criteria=term_criteria
        )

    map_l = pts_l.reshape(H, W, 2)
    map_r = pts_r.reshape(H, W, 2)

    for rect_map in (map_l, map_r):
        oob = (
            (rect_map[..., 0] < 0)
            | (rect_map[..., 0] >= W - 1)
            | (rect_map[..., 1] < 0)
            | (rect_map[..., 1] >= H - 1)
        )
        rect_map[oob] = -1

    intr_l = np.array([P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]])
    intr_r = np.array([P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]])
    baseline = P2[0, 3] / P2[0, 0]
    extr = np.array([baseline, 0, 0, 0, 0, 0, 1])

    return map_l, map_r, intr_l, intr_r, extr


def compute_stereo_remap(intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, fisheye):
    intr_l, map_l = compute_remap(intr_l, dist_l, W, H, fisheye)
    intr_r, map_r = compute_remap(intr_r, dist_r, W, H, fisheye)

    T_l2r = T_l @ np.linalg.inv(T_r)
    r = R.from_matrix(T_l2r[:3, :3])
    q = r.as_quat()
    extr = np.hstack((T_l2r[:3, 3], q[[0, 1, 2, 3]]))

    return map_l, map_r, intr_l, intr_r, extr


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
