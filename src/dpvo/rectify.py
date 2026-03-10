import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_map(intr, dist, H, W, fisheye):
    K = np.array(
        [
            [intr[0], 0, intr[2]],
            [0, intr[1], intr[3]],
            [0, 0, 1],
        ]
    )
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0, (W, H))
    intr = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])

    if fisheye:
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K, dist, None, K_new, (W, H), cv2.CV_32FC1
        )
    else:
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, dist, None, K_new, (W, H), cv2.CV_32FC1
        )

    return intr, (map_x, map_y)


def compute_stereo_map(
    intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, rectify, fisheye
):
    if not rectify:
        intr_l, map_l = compute_map(intr_l, dist_l, H, W, fisheye)
        intr_r, map_r = compute_map(intr_r, dist_r, H, W, fisheye)

        T_l2r = T_l @ np.linalg.inv(T_r)
        r = R.from_matrix(T_l2r[:3, :3])
        q = r.as_quat()
        extr = np.hstack((T_l2r[:3, 3], q[[0, 1, 2, 3]]))

        return map_l, map_r, intr_l, intr_r, extr

    K_l = np.array([[intr_l[0], 0, intr_l[2]], [0, intr_l[1], intr_l[3]], [0, 0, 1]])
    K_r = np.array([[intr_r[0], 0, intr_r[2]], [0, intr_r[1], intr_r[3]], [0, 0, 1]])

    T_l2r = T_l @ np.linalg.inv(T_r)
    R_rel = T_l2r[:3, :3]
    t_rel = T_l2r[:3, 3]

    if fisheye:
        R_l, R_r, P_l, P_r, _ = cv2.fisheye.stereoRectify(
            K_l,
            dist_l,
            K_r,
            dist_r,
            (W, H),
            R_rel,
            t_rel,
            flags=cv2.fisheye.CALIB_ZERO_DISPARITY,
            newImageSize=(W, H),
        )
        map_l = cv2.fisheye.initUndistortRectifyMap(
            K_l, dist_l, R_l, P_l[:, :3], (W, H), cv2.CV_32FC1
        )
        map_r = cv2.fisheye.initUndistortRectifyMap(
            K_r, dist_r, R_r, P_r[:, :3], (W, H), cv2.CV_32FC1
        )
    else:
        R_l, R_r, P_l, P_r, _, _, _ = cv2.stereoRectify(
            K_l, dist_l, K_r, dist_r, (W, H), R_rel, t_rel, alpha=0, newImageSize=(W, H)
        )
        map_l = cv2.initUndistortRectifyMap(
            K_l, dist_l, R_l, P_l[:, :3], (W, H), cv2.CV_32FC1
        )
        map_r = cv2.initUndistortRectifyMap(
            K_r, dist_r, R_r, P_r[:, :3], (W, H), cv2.CV_32FC1
        )

    intr_l_new = np.array([P_l[0, 0], P_l[1, 1], P_l[0, 2], P_l[1, 2]])
    intr_r_new = np.array([P_r[0, 0], P_r[1, 1], P_r[0, 2], P_r[1, 2]])

    tx = P_r[0, 3] / P_r[0, 0]
    extr_new = np.array([tx, 0, 0, 0, 0, 0, 1])

    return map_l, map_r, intr_l_new, intr_r_new, extr_new


def compute_inv_map(intr, dist, W, H, fisheye=False):
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


def compute_stereo_inv_map(
    intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, rectify, fisheye
):
    if not rectify:
        intr_l, map_l = compute_inv_map(intr_l, dist_l, W, H, fisheye)
        intr_r, map_r = compute_inv_map(intr_r, dist_r, W, H, fisheye)

        T_l2r = T_l @ np.linalg.inv(T_r)
        r = R.from_matrix(T_l2r[:3, :3])
        q = r.as_quat()
        extr = np.hstack((T_l2r[:3, 3], q[[0, 1, 2, 3]]))

        return map_l, map_r, intr_l, intr_r, extr

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
