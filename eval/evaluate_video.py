import argparse
import cProfile
import os
import pstats
from itertools import islice

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation as R

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
    path, intr, dist, start=None, stop=None, stride=1, clahe=False, scale=1.0
):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise IOError(f"Could not open video {path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    intrinsics = np.array(intr) * scale
    distortion_coeffs = np.array(dist)

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

    frame_idx = 0
    while True:
        ret, image = video.read()
        if not ret:
            break
        print(f"{frame_idx} / {frame_length}")

        if (
            frame_idx >= start
            and (stop is None or frame_idx < stop)
            and (frame_idx - start) % stride == 0
        ):
            if scale != 1.0:
                image = cv2.resize(image, (W, H))
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if clahe:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = clahe.apply(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            t = frame_idx / fps
            yield t, image, intrinsics_new

        frame_idx += 1


def check_video_resolution(path, scale):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise IOError(f"Could not open video {path}")

    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    return H, W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_2")
    parser.add_argument("video_1")
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
    parser.add_argument(
        "--intrinsics_2",
        type=float,
        nargs=4,
        default=[1066.9900, 1067.5900, 958.8900, 551.0580],
        help="fx fy cx cy",
    )
    parser.add_argument(
        "--distortion_2",
        type=float,
        nargs=5,
        default=[
            -0.0553349,
            0.0318106,
            0.000520924,
            0.00035819,
            -0.0128129,
        ],
        help="k1 k2 p1 p2 k3",
    )
    parser.add_argument(
        "--intrinsics_1",
        type=float,
        nargs=4,
        default=[1067.6400, 1067.7000, 919.6100, 501.5940],
        help="fx fy cx cy",
    )
    parser.add_argument(
        "--distortion_1",
        type=float,
        nargs=5,
        default=[
            -0.0685837,
            0.0517156,
            0.000521205,
            -0.000234591,
            -0.0217487,
        ],
        help="k1 k2 p1 p2 k3",
    )
    parser.add_argument(
        "--extrinsics",
        type=float,
        nargs=6,
        default=[
            120.0850e-3,
            -0.2124e-3,
            0.6204e-3,
            0.0048,
            0.0000,
            0.0001,
        ],
        help="x y z rx ry rz",
    )

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
        H, W = check_video_resolution(args.video_1, args.scale)

        r = R.from_rotvec(args.extrinsics[3:6])
        q = r.as_quat()
        extrinsics = np.hstack((args.extrinsics[0:3], q))
        print("Extrinsics (x y z qx qy qz qw): ", extrinsics)

        slam = DPVO(
            cfg,
            args.network,
            ht=H,
            wd=W,
            show=args.show,
            extrinsics=extrinsics,
            enable_timing=args.timeit,
            timing_file=args.timeit_file,
        )
        generator1 = pgenerator(
            rgb_generator,
            path=args.video_1,
            intr=args.intrinsics_1,
            dist=args.distortion_1,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            scale=args.scale,
            clahe=args.clahe,
        )
        generator2 = pgenerator(
            rgb_generator,
            path=args.video_2,
            intr=args.intrinsics_2,
            dist=args.distortion_2,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            scale=args.scale,
            clahe=args.clahe,
        )
        for i, ((t1, image1, intrinsics1), (t2, image2, intrinsics2)) in enumerate(
            zip(generator1, generator2, strict=False)
        ):
            if args.show:
                concat = cv2.hconcat([image1, image2])
                cv2.imshow("concat", concat)
                cv2.waitKey(1)

            image1 = torch.from_numpy(image1).permute(2, 0, 1).cuda()
            intrinsics1 = torch.from_numpy(intrinsics1).cuda()

            image2 = torch.from_numpy(image2).permute(2, 0, 1).cuda()
            intrinsics2 = torch.from_numpy(intrinsics2).cuda()

            with Timer("total", enabled=args.timeit, file=args.timeit_file):
                slam(t1, (image1, image2), (intrinsics1, intrinsics2))

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
        save_output_for_COLMAP(scene, traj_est, points, colors, *intrinsics1, H, W)

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
