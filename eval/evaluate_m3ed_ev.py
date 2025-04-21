import argparse
import cProfile
import os
import pstats
import time

import cv2
import evo.main_ape as main_ape
import h5py
import numpy as np
import torch
from dpvo.config import cfg
from dpvo.devo import DEVO
from dpvo.event import (
    compute_remap,
    to_voxel_grid_cuda,
    voxel_to_img,
)
from dpvo.parallel import pgenerator
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.utils import Timer
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from tqdm import tqdm


def ev_generator(path, camera_name, period, t_limits=(None, None), scale=1.0, bins=5):
    f = h5py.File(path, "r")
    camera_model = f.get(f"{camera_name}/calib/camera_model")[()]
    distortion_coeffs = f.get(f"{camera_name}/calib/distortion_coeffs")[()]
    distortion_model = f.get(f"{camera_name}/calib/distortion_model")[()]
    intrinsics = f.get(f"{camera_name}/calib/intrinsics")[()] * scale
    resolution = f.get(f"{camera_name}/calib/resolution")[()] * scale
    H, W = int(resolution[1]), int(resolution[0])

    x = f.get(f"{camera_name}/x")
    y = f.get(f"{camera_name}/y")
    t = f.get(f"{camera_name}/t")
    p = f.get(f"{camera_name}/p")
    ms_map = f.get(f"{camera_name}/ms_map_idx")[()]

    K = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ]
    )
    K_new, rect_map = compute_remap(K, distortion_coeffs, W, H)
    intrinsics_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
    duration = (t[-1] - t[0]) / 1e6

    N1, N2 = t_limits
    if N1 is None:
        N1 = 0
    if N2 is None:
        N2 = duration
    N1, N2 = int(N1 / period * 1e3), int(N2 / period * 1e3)

    print("camera_model", camera_model.decode())
    print("intrinsics", intrinsics)
    print("intrinsics_new", intrinsics_new)
    print("resolution", resolution)
    print("distortion_model", distortion_model.decode())
    print("distortion_coeffs", distortion_coeffs)
    print("duration", duration)
    print("number of events", len(t))

    for idx in tqdm(range(N1, N2)):
        tperf = time.perf_counter()
        t0_ms = period * idx
        t1_ms = period * (idx + 1)

        idx0 = int(ms_map[t0_ms])
        idx1 = int(ms_map[t1_ms])
        idx1 = min(idx1, len(x))
        if idx0 == idx1:
            break

        tb = t[idx0:idx1]
        tp = p[idx0:idx1]
        tx = x[idx0:idx1]
        ty = y[idx0:idx1]
        if scale != 1.0:
            tx = (tx * scale).astype(np.int32)
            ty = (ty * scale).astype(np.int32)

        print("event count", idx1 - idx0)
        rect = rect_map[ty, tx]
        x_rect = np.ascontiguousarray(rect[..., 0])
        y_rect = np.ascontiguousarray(rect[..., 1])

        tperf = time.perf_counter()
        voxel = to_voxel_grid_cuda(x_rect, y_rect, tb, tp, H, W, bins)
        print("to_voxel_grid time", time.perf_counter() - tperf)

        yield ((t0_ms + t1_ms) / 2e3, voxel, intrinsics_new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_h5")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--network", default="weights/devo.pth")
    parser.add_argument("--camera", default="/prophesee/left")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--save_colmap", action="store_true")
    parser.add_argument("--save_trajectory", action="store_true")
    parser.add_argument("--period", type=int, default=100)
    parser.add_argument("--timeit-file", type=str, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--stop", type=float, default=None)

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
        with h5py.File(args.data_h5, "r") as f:
            resolution = f.get(f"{args.camera}/calib/resolution")[()] * args.scale
            H, W = int(resolution[1]), int(resolution[0])
            bins = 5

        slam = DEVO(cfg, args.network, ht=H, wd=W)

        for t, voxel, intrinsics in pgenerator(
            ev_generator,
            path=args.data_h5,
            camera_name=args.camera,
            period=args.period,
            size=10,
            t_limits=(args.start, args.stop),
            scale=args.scale,
            bins=bins,
        ):
            with Timer("SLAM", enabled=args.timeit, file=args.timeit_file):
                voxel = np.clip(voxel, -32, 32)
                if args.show:
                    img = voxel_to_img(voxel)
                    cv2.imshow("voxel", img)
                    cv2.waitKey(1)

                voxel = torch.from_numpy(voxel).cuda()
                intrinsics = torch.from_numpy(intrinsics).cuda()
                slam(t, voxel, intrinsics)

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

    if args.save_trajectory:
        os.makedirs("saved_trajectories", exist_ok=True)
        file_interface.write_tum_trajectory_file(
            f"saved_trajectories/M3ED_{scene}{args.name}.txt", traj_est
        )

    if args.save_ply:
        save_ply(scene, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(scene, traj_est, points, colors, *intrinsics, H, W)

    ate_score = None
    if args.gt is not None:
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est = sync.associate_trajectories(
            traj_ref, traj_est, max_diff=0.1
        )

        try:
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
            correct_scale=True,
        )


if __name__ == "__main__":
    main()
