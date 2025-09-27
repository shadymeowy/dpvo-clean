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

from dpvo.bag import bag_image_iterator, read_calibration, sync_generators
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_ply,
    save_point_cloud,
)
from dpvo.utils import Timer


def rgb_generator(
    path_bag,
    cam_topic,
    H,
    W,
    intr,
    dist,
    start=None,
    stop=None,
    clahe=False,
    scale=1.0,
    shift=0,
):
    H, W = int(scale * H), int(scale * W)
    intr = scale * np.array(intr)
    K = np.array(
        [
            [intr[0], 0, intr[2]],
            [0, intr[1], intr[3]],
            [0, 0, 1],
        ]
    )
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0, (W, H))
    intr_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_new, (W, H), cv2.CV_32FC1)

    if clahe:
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    it = bag_image_iterator(path_bag, cam_topic)
    for t, image in islice(it, shift + start, stop):
        if scale != 1.0:
            image = cv2.resize(image, (W, H))
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if clahe:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = clahe.apply(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        yield t, image, intr_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_bag")
    parser.add_argument("path_config")
    parser.add_argument("--cam1", default="/cam0/image_raw")
    parser.add_argument("--cam2", default="/cam1/image_raw")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--network", default="weights/dpvo.pth")
    parser.add_argument("--timeit", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--name", default="test")
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

    print(f"Processing {args.name}")

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    with torch.no_grad():
        res1, res2, intr1, intr2, dist1, dist2, extr = read_calibration(
            args.path_config
        )
        assert res1.tolist() == res2.tolist()

        slam = DPVO(
            cfg,
            args.network,
            ht=int(res1[1] * args.scale),
            wd=int(res1[0] * args.scale),
            show=args.show,
            extrinsics=extr,
            enable_timing=args.timeit,
            timing_file=args.timeit_file,
        )

        generator1 = rgb_generator(
            path_bag=args.path_bag,
            cam_topic=args.cam1,
            W=res1[0],
            H=res1[1],
            intr=intr1,
            dist=dist1,
            start=args.start,
            stop=args.stop,
            clahe=args.clahe,
            scale=args.scale,
        )
        generator2 = rgb_generator(
            path_bag=args.path_bag,
            cam_topic=args.cam2,
            W=res2[0],
            H=res2[1],
            intr=intr2,
            dist=dist2,
            start=args.start,
            stop=args.stop,
            clahe=args.clahe,
            scale=args.scale,
        )

        generator = islice(
            sync_generators(generator1, generator2), None, None, args.stride
        )

        for i, ((t1, image1, intrinsics1), (t2, image2, intrinsics2)) in enumerate(
            generator
        ):
            if t1 != t2:
                raise Exception(
                    f"Error two cams are not sync {t1} != {t2}, try --shift for manual alignment"
                )
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
                os.makedirs(f"saved_matches/{args.name}", exist_ok=True)
                cv2.imwrite(
                    f"saved_matches/{args.name}/{i:06d}.jpg",
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
            f"saved_trajectories/{args.name}.txt", traj_est
        )

    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(
            args.name, traj_est, points, colors, *intrinsics1, res1[1], res1[0]
        )

    if args.save_point_cloud:
        os.makedirs("saved_point_clouds", exist_ok=True)
        save_point_cloud(
            f"saved_point_clouds/{args.name}.viz.txt",
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
            plot_name = f"{args.name} (ATE: {ate_score:.03f})"
        except np.linalg.LinAlgError:
            print("Error in trajectory association, skipping ATE calculation.")
            ate_score = None
            plot_name = f"{args.name} (ATE: NaN)"
    else:
        plot_name = f"{args.name}"

    if args.plot:
        os.makedirs("trajectory_plots", exist_ok=True)
        plot_trajectory(
            traj_est,
            traj_ref if ate_score is not None else None,
            plot_name,
            f"trajectory_plots/{args.name}.pdf",
            align=True,
            correct_scale=False,
        )


if __name__ == "__main__":
    main()
