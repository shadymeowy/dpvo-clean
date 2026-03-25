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
)
from dpvo.rectify import compute_stereo_map
from dpvo.utils import Timer


def rgb_generator(
    path, start=None, stop=None, stride=1, clahe=False, scale=1.0, H=None, W=None, rect_map=None
):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise IOError(f"Could not open video {path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

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
            image = cv2.remap(image, rect_map[0], rect_map[1], cv2.INTER_LINEAR)

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if clahe:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = clahe.apply(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            t = frame_idx / fps
            yield t, image

        frame_idx += 1


def check_video_resolution(path, scale):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise IOError(f"Could not open video {path}")

    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    return H, W


def rgb_stereo_generator(
    path_left,
    path_right,
    intr_l,
    intr_r,
    dist_l,
    dist_r,
    extr,
    scale=1.0,
    rectify=False,
    **kwargs,
):
    H_l, W_l = check_video_resolution(path_left, scale)
    H_r, W_r = check_video_resolution(path_right, scale)
    if (H_l, W_l) != (H_r, W_r):
        raise ValueError(
            f"Video resolutions do not match: {(H_l, W_l)} != {(H_r, W_r)}"
        )

    T_l = np.eye(4)
    T_l2r = np.eye(4)
    T_l2r[:3, :3] = R.from_rotvec(extr[3:6]).as_matrix()
    T_l2r[:3, 3] = extr[:3]
    T_r = np.linalg.inv(T_l2r)

    map_l, map_r, intr_l, intr_r, extr = compute_stereo_map(
        np.array(intr_l) * scale,
        np.array(intr_r) * scale,
        np.array(dist_l),
        np.array(dist_r),
        T_l,
        T_r,
        H_l,
        W_l,
        rectify,
        fisheye=False,
    )

    gen_l = pgenerator(
        rgb_generator,
        path=path_left,
        scale=scale,
        H=H_l,
        W=W_l,
        rect_map=map_l,
        **kwargs,
    )
    gen_r = pgenerator(
        rgb_generator,
        path=path_right,
        scale=scale,
        H=H_l,
        W=W_l,
        rect_map=map_r,
        **kwargs,
    )

    return zip(gen_l, gen_r, strict=False), intr_l, intr_r, (H_l, W_l), extr


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
    parser.add_argument("--save_matches", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--timeit-file", type=str, default=None)
    parser.add_argument("--point_cloud", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no_rect", action="store_true")
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
        scene = os.path.splitext(os.path.basename(args.video_1))[0]
    print(f"Processing M3ED_{scene}{args.name}")

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    if args.gt is not None:
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)

    if args.visualize:
        from dpvo.viz import ProcessViz

        if args.gt is not None:
            t_gt = traj_ref.timestamps
            pos_gt = traj_ref.positions_xyz
            viz = ProcessViz(t_gt, pos_gt)
        else:
            viz = ProcessViz()

    with torch.no_grad():
        gen, intr_l, intr_r, (H, W), extrinsics = rgb_stereo_generator(
            path_left=args.video_1,
            path_right=args.video_2,
            intr_l=args.intrinsics_1,
            intr_r=args.intrinsics_2,
            dist_l=args.distortion_1,
            dist_r=args.distortion_2,
            extr=args.extrinsics,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            scale=args.scale,
            clahe=args.clahe,
            rectify=not args.no_rect,
        )
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
        intr_l = torch.from_numpy(intr_l).cuda()
        intr_r = torch.from_numpy(intr_r).cuda()
        point_cloud = []
        for i, ((t1, image1), (t2, image2)) in enumerate(gen):
            if args.show:
                concat = cv2.hconcat([image1, image2])
                cv2.imshow("concat", concat)
                cv2.waitKey(1)

            image1 = torch.from_numpy(image1).permute(2, 0, 1).cuda()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).cuda()

            with Timer("total", enabled=args.timeit, file=args.timeit_file):
                pose = slam(t1, (image1, image2), (intr_l, intr_r))

            if pose is not None:
                if args.point_cloud or args.visualize:
                    pc = slam.point_cloud().cpu().numpy()

                if args.point_cloud:
                    point_cloud.append(pc)

                if args.visualize:
                    viz.add(t1, pose[:3], pose[3:], pc)

            if args.save_matches and slam.concatenated_image is not None:
                os.makedirs(f"saved_matches/M3ED_{scene}{args.name}", exist_ok=True)
                cv2.imwrite(
                    f"saved_matches/M3ED_{scene}{args.name}/{i:06d}.jpg",
                    slam.concatenated_image,
                )

        points = slam.pg.points_.cpu().numpy()[: slam.m]
        colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]

        poses, tstamps = slam.terminate()

        if args.visualize:
            viz.close()

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

    ate_score = None
    if args.gt is not None:
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

    if args.point_cloud and len(point_cloud) > 0:
        point_cloud = np.stack(point_cloud, axis=0)
        os.makedirs("point_clouds", exist_ok=True)
        np.save(f"point_clouds/M3ED_{scene}{args.name}.npy", point_cloud)
        print(f"Saved point cloud with {point_cloud.shape}")


if __name__ == "__main__":
    main()
