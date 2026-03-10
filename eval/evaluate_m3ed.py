import argparse
import cProfile
import os
import pstats
from itertools import islice

import cv2
import evo.main_ape as main_ape
import h5py
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.parallel import pgenerator
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_ply,
    save_point_cloud,
)
from dpvo.utils import Timer


def image_reader(path, camera_name, start, stop, stride, clahe, scale, H, W, rect_map):
    f = h5py.File(path, "r")
    data = f.get(f"{camera_name}/data")
    ts = f.get(f"{'/'.join(camera_name.split('/')[:-1])}/ts")[...] / 1e6
    N = data.shape[0] // stride

    if clahe:
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    for t, image in tqdm(islice(zip(ts, data), start, stop, stride), total=N):
        if scale != 1.0:
            image = cv2.resize(image, (W, H))
        image = cv2.remap(image, rect_map[0], rect_map[1], cv2.INTER_LINEAR)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if clahe:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        yield t, image


def rgb_stereo_generator(
    path,
    camera_left,
    camera_right,
    scale=1.0,
    fisheye=False,
    rectify=False,
    **kwargs,
):
    f = h5py.File(path, "r")

    dist_l = f.get(f"{camera_left}/calib/distortion_coeffs")[()]
    intr_l = f.get(f"{camera_left}/calib/intrinsics")[()] * scale
    resolution = f.get(f"{camera_left}/calib/resolution")[()] * scale
    H, W = int(resolution[1]), int(resolution[0])

    intr_r = f[f"{camera_right}/calib/intrinsics"][()] * scale
    dist_r = f[f"{camera_right}/calib/distortion_coeffs"][()]

    T_l = f[f"{camera_left}/calib/T_to_prophesee_left"][()]
    T_r = f[f"{camera_right}/calib/T_to_prophesee_left"][()]

    fun = compute_stereo_rect_map if rectify else compute_stereo_map
    map_l, map_r, intr_l, intr_r, extr = fun(
        intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, fisheye
    )

    gen_l = pgenerator(
        image_reader,
        path,
        camera_left,
        scale=scale,
        W=W,
        H=H,
        rect_map=map_l,
        **kwargs,
    )
    gen_r = pgenerator(
        image_reader,
        path,
        camera_right,
        scale=scale,
        W=W,
        H=H,
        rect_map=map_r,
        **kwargs,
    )

    return zip(gen_l, gen_r, strict=False), intr_l, intr_r, (H, W), extr


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


def compute_stereo_map(intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, fisheye):
    intr_l, map_l = compute_map(intr_l, dist_l, H, W, fisheye)
    intr_r, map_r = compute_map(intr_r, dist_r, H, W, fisheye)

    T_l2r = T_l @ np.linalg.inv(T_r)
    r = R.from_matrix(T_l2r[:3, :3])
    q = r.as_quat()
    extr = np.hstack((T_l2r[:3, 3], q[[0, 1, 2, 3]]))

    return map_l, map_r, intr_l, intr_r, extr


def compute_stereo_rect_map(intr_l, intr_r, dist_l, dist_r, T_l, T_r, H, W, fisheye):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_h5")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--network", default="weights/dpvo.pth")
    parser.add_argument("--camera", default="/ovc/left")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_colmap", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--save_point_cloud", action="store_true")
    parser.add_argument("--save_matches", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--timeit-file", type=str, default=None)
    parser.add_argument("--no_rect", action="store_true")

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    if args.scene is not None:
        scene = args.scene
    else:
        scene = os.path.splitext(os.path.basename(args.data_h5))[0]
    print(f"Processing M3ED_{scene}{args.name}")

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    with torch.no_grad():
        gen, intr_l, intr_r, (H, W), extr = rgb_stereo_generator(
            path=args.data_h5,
            camera_left=args.camera,
            camera_right=args.camera.replace("left", "right"),
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            scale=args.scale,
            clahe=args.clahe,
            rectify=not args.no_rect,
        )

        slam = DPVO(
            cfg,
            args.network,
            ht=H,
            wd=W,
            show=args.show,
            extrinsics=extr,
            enable_timing=args.timeit,
            timing_file=args.timeit_file,
        )

        intr_l = torch.from_numpy(intr_l).cuda()
        intr_r = torch.from_numpy(intr_r).cuda()

        for i, ((t1, image1), (_, image2)) in enumerate(gen):
            if args.show:
                concat = cv2.hconcat([image1, image2])
                cv2.imshow("concat", concat)
                cv2.waitKey(1)

            image1 = torch.from_numpy(image1).permute(2, 0, 1).cuda()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).cuda()

            with Timer("total", enabled=args.timeit, file=args.timeit_file):
                slam(t1, (image1, image2), (intr_l, intr_r))

            if args.save_matches and slam.concatenated_image is not None:
                os.makedirs(f"saved_matches/M3ED_{scene}{args.name}", exist_ok=True)
                cv2.imwrite(
                    f"saved_matches/M3ED_{scene}{args.name}/{i:06d}.jpg",
                    slam.concatenated_image,
                )

        points = slam.pg.points_.cpu().numpy()[: slam.m]
        colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]
        points_idx = slam.pg.tstamps_[slam.pg.ix[: slam.m].cpu().numpy()]

        poses, tstamps = slam.terminate()

    if args.profile:
        profile.disable()

        with open(args.profile, "w") as f:
            stats = pstats.Stats(profile, stream=f)
            stats = stats.strip_dirs().sort_stats("cumtime")
            stats.print_stats()

    traj_est = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=tstamps,
    )

    if args.save_trajectory:
        os.makedirs("saved_trajectories", exist_ok=True)
        file_interface.write_tum_trajectory_file(
            f"saved_trajectories/M3ED_{scene}{args.name}.txt", traj_est
        )

    if args.save_ply:
        save_ply(scene, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(scene, traj_est, points, colors, *intr_l, H, W)

    if args.save_point_cloud:
        os.makedirs("saved_point_clouds", exist_ok=True)
        save_point_cloud(
            f"saved_point_clouds/M3ED_{scene}{args.name}.viz.txt",
            traj_est,
            points,
            points_idx,
            colors,
        )

    ate_score = None
    if args.gt is not None:
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

        try:
            result = main_ape.ape(
                traj_ref,
                traj_est,
                est_name="traj",
                pose_relation=PoseRelation.translation_part,
                align=True,
                correct_scale=False,
            )
            ate_score = result.stats["rmse"]
            print(f"ATE: {ate_score:.03f}")
            print(result.stats)
            plot_name = f"M3ED {scene} (ATE: {ate_score:.03f})"
        except np.linalg.LinAlgError:
            print("Error in trajectory association, skipping ATE calculation.")
            ate_score = None
            plot_name = f"M3ED {scene} (ATE: NaN)"
    else:
        plot_name = f"M3ED {scene}"

    if args.plot:
        os.makedirs("trajectory_plots", exist_ok=True)
        plot_trajectory(
            traj_est,
            traj_ref if ate_score is not None else None,
            plot_name,
            f"trajectory_plots/M3ED_{scene}{args.name}.pdf",
            align=True,
            correct_scale=False,
        )


if __name__ == "__main__":
    main()
