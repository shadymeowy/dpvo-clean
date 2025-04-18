import argparse
import cProfile
import os
import pstats

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
from dpvo.devo import DEVO
from dpvo.event import (
    compute_remap,
    get_time_indices_offsets,
    to_voxel_grid,
    voxel_to_img,
)
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.utils import Timer

parser = argparse.ArgumentParser()
parser.add_argument("data_h5")
parser.add_argument("--gt", default=None)
parser.add_argument("--scene", default=None)
parser.add_argument("--network", default="weights/devo.pth")
parser.add_argument("--camera", default="/prophesee/left")
parser.add_argument("--timeit", action="store_true")
parser.add_argument("--print-h5", action="store_true")
parser.add_argument("--show", action="store_true")
parser.add_argument("--name", default="")
parser.add_argument("--profile", type=str, default=None)
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--opts", nargs="+", default=[])
parser.add_argument("--save_ply", action="store_true")
parser.add_argument("--save_colmap", action="store_true")
parser.add_argument("--save_trajectory", action="store_true")
parser.add_argument("--period", type=int, default=80)
parser.add_argument("--timeit-file", type=str, default=None)

args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.merge_from_list(args.opts)

if args.scene is not None:
    scene = args.scene
else:
    scene = os.path.splitext(os.path.basename(args.data_h5))[0]
print(f"Processing M3ED_{scene}{args.name}")


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
    K_new, rect_map = compute_remap(K, distortion_coeffs, resolution[0], resolution[1])
    intrinsics_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
    print("intrinsics_new", intrinsics_new)

    event_x = f.get(f"{args.camera}/x")
    event_y = f.get(f"{args.camera}/y")
    event_t = f.get(f"{args.camera}/t")
    event_p = f.get(f"{args.camera}/p")
    event_ms_map_idx = f.get(f"{args.camera}/ms_map_idx")[()]

    duration = (event_t[-1] - event_t[0]) / 1e6
    print(f"duration: {duration:.3f} s")
    print(f"number of events: {len(event_x)}")
    print(f"first event: {event_t[0] / 1e6:.3f} s")
    print(f"last event: {event_t[-1] / 1e6:.3f} s")

    N = int(duration / args.period * 1e3)
    H, W = int(resolution[1]), int(resolution[0])

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    with torch.no_grad():
        intrinsics_new = torch.from_numpy(intrinsics_new).cuda()
        slam = DEVO(cfg, args.network, ht=H, wd=W)

        bins = 5
        voxel_flat = np.zeros((bins + 1, H, W), dtype=np.float32)
        voxel = voxel_flat[:-1]
        voxel_flat = voxel_flat.ravel()
        for idx in tqdm(range(0, 2 * N - 1)):
            t_start_us = args.period / 2 * idx * 1e3
            t_end_us = args.period / 2 * (idx + 2) * 1e3
            t_offset = 0

            t_start_ms, t_end_ms = (
                int(np.floor((t_start_us - t_offset) / 1000)),
                int(np.ceil((t_end_us - t_offset) / 1000)),
            )
            t_start_ms_idx = int(event_ms_map_idx[t_start_ms])
            t_end_ms_idx = int(event_ms_map_idx[t_end_ms])

            t_batch = event_t[t_start_ms_idx:t_end_ms_idx]

            idx_start_offset, idx_end_offset = get_time_indices_offsets(
                t_batch, t_start_us - t_offset, t_end_us - t_offset
            )

            idx_start = t_start_ms_idx + idx_start_offset
            idx_end = t_start_ms_idx + idx_end_offset

            batch_t = event_t[idx_start:idx_end] + t_offset
            batch_x = event_x[idx_start:idx_end]
            batch_y = event_y[idx_start:idx_end]
            batch_p = event_p[idx_start:idx_end]

            if len(batch_t) == 0:
                continue

            rect = rect_map[batch_y, batch_x]
            x_rect = np.ascontiguousarray(rect[..., 0])
            y_rect = np.ascontiguousarray(rect[..., 1])

            to_voxel_grid(
                voxel_flat,
                batch_x,
                batch_y,
                batch_t,
                batch_p,
                H,
                W,
                bins,
            )
            img = voxel_to_img(voxel)
            voxel_tensor = torch.from_numpy(voxel).cuda()

            with Timer("SLAM", enabled=args.timeit, file=args.timeit_file):
                slam(t_start_us / 1e6, voxel_tensor, intrinsics_new)

            if args.show:
                cv2.imshow("voxel", img)
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
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

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
