import argparse
import cProfile
import os
import pstats

import aedat
import cv2
import evo.main_ape as main_ape
import matplotlib.pyplot as plt
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
from dpvo.parallel import pgenerator
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_ply,
    save_point_cloud,
)
from dpvo.utils import Timer


def ev_generator(
    path,
    resolution,
    intrinsics,
    distortion,
    period,
    t_limits=(None, None),
    scale=1.0,
    bins=5,
    stride=1,
    fisheye=False,
):
    decoder = aedat.Decoder(path)
    xs, ys, ts, ps = [], [], [], []
    for packet in decoder:
        if "events" not in packet:
            continue

        if "imus" in packet:
            print(f"IMU packet: {packet['imus']}")
            exit()

        tb = packet["events"]["t"]
        tx = packet["events"]["x"]
        ty = packet["events"]["y"]
        tp = packet["events"]["on"]
        xs.append(tx)
        ys.append(ty)
        ts.append(tb)
        ps.append(tp)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    ts = np.concatenate(ts)
    ps = np.concatenate(ps)

    print(f"Total number of events: {len(ts)}")
    H = resolution[0]
    W = resolution[1]

    intrinsics = np.array(intrinsics)
    distortion = np.array(distortion)

    K = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ]
    )
    K_new, rect_map = compute_remap(K, distortion, W, H, fisheye=fisheye)
    intrinsics_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])

    voxel = np.zeros((bins + 1, H, W), dtype=np.float32)
    duration = (ts[-1] - ts[0]) / 1e6
    ts -= ts[0]

    N1, N2 = t_limits
    if N1 is None:
        N1 = 0
    if N2 is None:
        N2 = duration
    N1, N2 = int(N1 / period * 1e3), int(N2 / period * 1e3)

    for idx in tqdm(range(N1, N2, stride)):
        t0_us = period * idx * 1e3
        t1_us = period * (idx + 1) * 1e3
        idx0, idx1 = get_time_indices_offsets(ts, t0_us, t1_us)

        tx = xs[idx0:idx1]
        ty = ys[idx0:idx1]
        tb = ts[idx0:idx1]
        tp = ps[idx0:idx1]

        tx = (tx * scale).astype(np.int32)
        ty = (ty * scale).astype(np.int32)

        rect = rect_map[ty, tx]
        x_rect = np.ascontiguousarray(rect[..., 0])
        y_rect = np.ascontiguousarray(rect[..., 1])

        to_voxel_grid(voxel.ravel(), x_rect, y_rect, tb, tp, H, W, bins)

        yield (tb[0] / 1e6, voxel[:-1], intrinsics_new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("aedat4")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--network", default="weights/devo.pth")
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
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--clahe", action="store_true")  # wont work
    parser.add_argument("--save_point_cloud", action="store_true")
    parser.add_argument("--save_matches", action="store_true")
    parser.add_argument("--fisheye", action="store_true")
    parser.add_argument("--resolution", nargs=2, type=int, default=[260, 346])
    parser.add_argument(
        "--distortion",
        nargs=4,
        type=float,
        default=[
            -0.08397448083992665,
            0.0176990077535882,
            -0.019437791621552163,
            0.018560422393927578,
        ],
    )
    parser.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        default=[
            256.1984795112278,
            255.9233416580906,
            166.3533868342309,
            129.20806005047345,
        ],
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    if args.scene is not None:
        scene = args.scene
    else:
        scene = "scene"
    print(f"Processing M3ED_{scene}{args.name}")

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    with torch.no_grad():
        H = args.resolution[0]
        W = args.resolution[1]
        bins = 5

        slam = DEVO(
            cfg,
            args.network,
            ht=H,
            wd=W,
            show=args.show,
            enable_timing=args.timeit,
            timing_file=args.timeit_file,
        )

        for i, (t, voxel, intrinsics) in enumerate(
            pgenerator(
                ev_generator,
                path=args.aedat4,
                intrinsics=args.intrinsics,
                distortion=args.distortion,
                period=args.period,
                t_limits=(args.start, args.stop),
                scale=args.scale,
                bins=bins,
                stride=args.stride,
                resolution=(H, W),
                fisheye=args.fisheye,
            )
        ):
            if args.show:
                img = voxel_to_img(voxel)
                cv2.imshow("voxel", img)
                cv2.waitKey(1)

            voxel = torch.from_numpy(voxel).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            with Timer("total", enabled=args.timeit, file=args.timeit_file):
                slam(t, voxel, intrinsics)

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
            f"saved_trajectories/DAVIS_{scene}{args.name}.txt", traj_est
        )

    if args.save_ply:
        save_ply(scene, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(scene, traj_est, points, colors, *intrinsics, H, W)

    if args.save_point_cloud:
        os.makedirs("saved_point_clouds", exist_ok=True)
        save_point_cloud(
            f"saved_point_clouds/DAVIS_{scene}{args.name}.viz.txt",
            traj_est,
            points,
            points_idx,
            colors,
        )

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
