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
from tqdm import tqdm

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.utils import Timer

parser = argparse.ArgumentParser()
parser.add_argument("data_h5")
parser.add_argument("--gt", default=None)
parser.add_argument("--scene", default=None)
parser.add_argument("--network", default="weights/dpvo.pth")
parser.add_argument("--camera", default="/ovc/left")
parser.add_argument("--time", default="/ovc/ts")
parser.add_argument("--timeit", action="store_true")
parser.add_argument("--print-h5", action="store_true")
parser.add_argument("--show", action="store_true")
parser.add_argument("--name", default="")
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--profile", type=str, default=None)
parser.add_argument("--clahe", action="store_true")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--opts", nargs="+", default=[])
parser.add_argument("--save_ply", action="store_true")
parser.add_argument("--save_colmap", action="store_true")
parser.add_argument("--save_trajectory", action="store_true")
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--skip", type=int, default=0)

args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.merge_from_list(args.opts)

with h5py.File(args.data_h5) as f:
    if args.print_h5:
        keys = []
        f.visit(keys.append)
        for key in keys:
            print(key)

    camera_model = f.get(f"{args.camera}/calib/camera_model")[()]
    distortion_coeffs = f.get(f"{args.camera}/calib/distortion_coeffs")[()]
    distortion_model = f.get(f"{args.camera}/calib/distortion_model")[()]
    intrinsics = f.get(f"{args.camera}/calib/intrinsics")[()]
    resolution = f.get(f"{args.camera}/calib/resolution")[()]

    print("camera_model", camera_model.decode())
    print("intrinsics", intrinsics)
    print("resolution", resolution)
    print("distortion_model", distortion_model.decode())
    print("distortion_coeffs", distortion_coeffs)

    K = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ]
    )
    K_new, roi = cv2.getOptimalNewCameraMatrix(
        K, distortion_coeffs, resolution, 0, resolution
    )
    intrinsics_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
    print("intrinsics_new", intrinsics_new)
    print("roi", roi)
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, distortion_coeffs, None, K_new, resolution, cv2.CV_32FC1
    )

    data = f.get(f"{args.camera}/data")
    N = data.shape[0] // args.stride
    ts = f.get(f"{args.time}")[...] / 1e6
    image = data[0]
    if args.scale != 1.0:
        image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale)
    H, W, _ = image.shape

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    
    if args.clahe:
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    with torch.no_grad():
        intrinsics_new = torch.from_numpy(intrinsics_new).cuda()
        if args.scale != 1.0:
            intrinsics_new[:] *= args.scale
        slam = DPVO(cfg, args.network, ht=H, wd=W)
        for t, image in tqdm(
            islice(zip(ts, data), args.skip, args.end, args.stride), total=N
        ):
            if args.show:
                cv2.imshow("distorted", image)
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            if args.scale != 1.0:
                image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale)

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if args.clahe:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = clahe.apply(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if args.show:
                cv2.imshow("undistorted", image)
            image = torch.from_numpy(image).permute(2, 0, 1).cuda()

            if slam is None:
                _, H, W = image.shape

            with Timer("SLAM", enabled=args.timeit):
                slam(t, image, intrinsics_new)
            if args.show:
                cv2.waitKey(1)

        points = slam.pg.points_.cpu().numpy()[: slam.m]
        colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]

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

    if args.scene is not None:
        scene = args.scene
    else:
        scene = os.path.splitext(os.path.basename(args.data_h5))[0]

    if args.gt is not None:
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

        result = main_ape.ape(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
        ate_score = result.stats["rmse"]
        print(f"ATE: {ate_score:.03f}")
        print(result.stats)

        if args.plot:
            os.makedirs("trajectory_plots", exist_ok=True)
            plot_trajectory(
                traj_est,
                traj_ref,
                f"M3ED {scene} (ATE: {ate_score:.03f})",
                f"trajectory_plots/M3ED_{scene}{args.name}.pdf",
                align=True,
                correct_scale=True,
            )

        if args.save_trajectory:
            os.makedirs("saved_trajectories", exist_ok=True)
            file_interface.write_tum_trajectory_file(
                f"saved_trajectories/M3ED_{scene}{args.name}.txt", traj_est
            )

    else:
        if args.save_trajectory:
            os.makedirs("saved_trajectories", exist_ok=True)
            file_interface.write_tum_trajectory_file(
                f"saved_trajectories/M3ED_{scene}{args.name}.txt", traj_est
            )

        if args.plot:
            os.makedirs("trajectory_plots", exist_ok=True)
            plot_trajectory(
                traj_est,
                None,
                f"M3ED {scene}",
                f"trajectory_plots/M3ED_{scene}{args.name}.pdf",
                align=True,
                correct_scale=True,
            )

    if args.save_ply:
        save_ply(scene, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(scene, traj_est, points, colors, *intrinsics, H, W)
