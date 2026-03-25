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

from dpvo.bag import bag_image_iterator, read_calibration, sync_generators
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import (
    plot_trajectory,
    save_output_for_COLMAP,
    save_ply,
)
from dpvo.rectify import compute_stereo_map
from dpvo.utils import Timer


def rgb_generator(
    path_bag,
    cam_topic,
    start=None,
    stop=None,
    clahe=False,
    scale=1.0,
    shift=0,
    H=None,
    W=None,
    rect_map=None,
):
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    it = bag_image_iterator(path_bag, cam_topic)
    for t, image in islice(it, shift + start, stop):
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
    path_bag,
    path_config,
    cam1,
    cam2,
    scale=1.0,
    rectify=False,
    **kwargs,
):
    res1, res2, intr1, intr2, dist1, dist2, extr = read_calibration(path_config)
    assert res1.tolist() == res2.tolist()

    H = int(res1[1] * scale)
    W = int(res1[0] * scale)
    intr1 = np.array(intr1) * scale
    intr2 = np.array(intr2) * scale

    T_l = np.eye(4)
    T_l2r = np.eye(4)
    T_l2r[:3, :3] = R.from_quat(extr[3:]).as_matrix()
    T_l2r[:3, 3] = extr[:3]
    T_r = np.linalg.inv(T_l2r)

    map_l, map_r, intr_l, intr_r, extr = compute_stereo_map(
        intr1, intr2, dist1, dist2, T_l, T_r, H, W, rectify, fisheye=False
    )

    gen_l = rgb_generator(
        path_bag=path_bag,
        cam_topic=cam1,
        scale=scale,
        H=H,
        W=W,
        rect_map=map_l,
        **kwargs,
    )
    gen_r = rgb_generator(
        path_bag=path_bag,
        cam_topic=cam2,
        scale=scale,
        H=H,
        W=W,
        rect_map=map_r,
        **kwargs,
    )

    return sync_generators(gen_l, gen_r), intr_l, intr_r, (H, W), extr


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
    parser.add_argument("--save_matches", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--timeit-file", type=str, default=None)
    parser.add_argument("--point_cloud", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no_rect", action="store_true")

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print(f"Processing {args.name}")

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
        generator, intr_l, intr_r, (H, W), extr = rgb_stereo_generator(
            path_bag=args.path_bag,
            path_config=args.path_config,
            cam1=args.cam1,
            cam2=args.cam2,
            start=args.start,
            stop=args.stop,
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

        generator = islice(generator, None, None, args.stride)
        point_cloud = []

        for i, ((t1, image1), (t2, image2)) in enumerate(generator):
            if t1 != t2:
                raise Exception(
                    f"Error two cams are not sync {t1} != {t2}, try --shift for manual alignment"
                )
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
                os.makedirs(f"saved_matches/{args.name}", exist_ok=True)
                cv2.imwrite(
                    f"saved_matches/{args.name}/{i:06d}.jpg",
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
            f"saved_trajectories/{args.name}.txt", traj_est
        )

    if args.save_ply:
        save_ply(args.name, points, colors)

    if args.save_colmap:
        save_output_for_COLMAP(args.name, traj_est, points, colors, *intr_l, H, W)

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

    if args.point_cloud and len(point_cloud) > 0:
        point_cloud = np.stack(point_cloud, axis=0)
        os.makedirs("point_clouds", exist_ok=True)
        np.save(f"point_clouds/{args.name}.npy", point_cloud)
        print(f"Saved point cloud with {point_cloud.shape}")


if __name__ == "__main__":
    main()
