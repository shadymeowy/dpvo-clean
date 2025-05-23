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
from dpvo.parallel import pgenerator
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_ply,
    save_point_cloud,
)
from dpvo.utils import Timer


def rgb_generator(
    path, camera_name, start=None, stop=None, stride=1, clahe=False, scale=1.0
):
    f = h5py.File(path, "r")
    camera_model = f.get(f"{camera_name}/calib/camera_model")[()]
    distortion_coeffs = f.get(f"{camera_name}/calib/distortion_coeffs")[()]
    distortion_model = f.get(f"{camera_name}/calib/distortion_model")[()]
    intrinsics = f.get(f"{camera_name}/calib/intrinsics")[()] * scale
    resolution = f.get(f"{camera_name}/calib/resolution")[()] * scale
    H, W = int(resolution[1]), int(resolution[0])

    data = f.get(f"{camera_name}/data")
    ts = f.get(f"{'/'.join(camera_name.split('/')[:-1])}/ts")[...] / 1e6

    K = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ]
    )
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, distortion_coeffs, (W, H), 0, (W, H))
    intrinsics_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, distortion_coeffs, None, K_new, (W, H), cv2.CV_32FC1
    )

    if clahe:
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    N = data.shape[0] // stride

    print("camera_model", camera_model.decode())
    print("intrinsics", intrinsics)
    print("intrinsics_new", intrinsics_new)
    print("resolution", resolution)
    print("distortion_model", distortion_model.decode())
    print("distortion_coeffs", distortion_coeffs)

    for t, image in tqdm(islice(zip(ts, data), start, stop, stride), total=N):
        if scale != 1.0:
            image = cv2.resize(image, (W, H))
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if clahe:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        yield t, image, intrinsics_new


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

        slam = DPVO(
            cfg,
            args.network,
            ht=H,
            wd=W,
            show=args.show,
            enable_timing=args.timeit,
            timing_file=args.timeit_file,
        )
        for i, (t, image, intrinsics) in enumerate(
            pgenerator(
                rgb_generator,
                path=args.data_h5,
                camera_name=args.camera,
                start=args.start,
                stop=args.stop,
                stride=args.stride,
                scale=args.scale,
                clahe=args.clahe,
            )
        ):
            if args.show:
                cv2.imshow("image", image)
                cv2.waitKey(1)
            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            with Timer("SLAM", enabled=args.timeit, file=args.timeit_file):
                slam(t, image, intrinsics)

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
        save_output_for_COLMAP(scene, traj_est, points, colors, *intrinsics, H, W)

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
