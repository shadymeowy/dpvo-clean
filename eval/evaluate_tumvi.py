import argparse
import os
from itertools import islice

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply
from dpvo.utils import Timer
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from tqdm import tqdm


def parse_cam_calib(txt):
    txt = txt.splitlines()
    assert txt[1] == txt[3]
    assert txt[2] == "crop"
    resolution = np.array(tuple(map(int, txt[1].split())))
    compound = txt[0].split()
    assert compound[0] == "EquiDistant"
    compound = tuple(map(float, compound[1:]))
    intrinsics = np.array(compound[0:4])
    intrinsics *= np.concatenate((resolution, resolution))
    distortion = np.array(compound[4:8])
    return resolution, intrinsics, distortion


parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--scene", default=None)
parser.add_argument("--network", default="weights/dpvo.pth")
parser.add_argument("--camera", default="cam0")
parser.add_argument("--timeit", action="store_true")
parser.add_argument("--target-intrinsics", nargs=4, type=float, default=None, metavar=("fx", "fy", "cx", "cz"))

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

path_cam = os.path.join(args.data_dir, "dso", args.camera)
calib = open(os.path.join(path_cam, "camera.txt")).read()
resolution, intrinsics, distortion_coeffs = parse_cam_calib(calib)

print("intrinsics", intrinsics)
print("resolution", resolution)
print("distortion_coeffs", distortion_coeffs)

K = np.array(
    [
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1],
    ]
)
# K_new, roi = cv2.getOptimalNewCameraMatrix(
#     K, distortion_coeffs, resolution, 0, resolution
# )
# intrinsics_new = np.array([K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]])
# print("intrinsics_new", intrinsics_new)
# print("roi", roi)
# mapx, mapy = cv2.initUndistortRectifyMap(
#     K, distortion_coeffs, None, K_new, resolution, cv2.CV_32FC1
# )
if args.target_intrinsics is None:
    intrinsics_new = intrinsics.copy()
else:
    intrinsics_new = np.array(args.target_intrinsics)
K_new = np.array(
    [
        [intrinsics_new[0], 0, intrinsics_new[2]],
        [0, intrinsics_new[1], intrinsics_new[3]],
        [0, 0, 1],
    ]
)
print("intrinsics_new", intrinsics_new)
mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
    K, distortion_coeffs, np.eye(3), K_new, resolution, cv2.CV_32FC1
)

# Load timestamps
path_time = os.path.join(path_cam, "times.txt")
cam_times = np.loadtxt(path_time, delimiter=" ", skiprows=1)
ifilenames = cam_times[:, 0]
ts = cam_times[:, 1]
# exposuretime = cam_times[:, 2]

# Load image data
image_dir = os.path.join(path_cam, "images")
filenames = sorted(os.listdir(image_dir))
N = len(filenames) // args.stride

with torch.no_grad():
    intrinsics_new = torch.from_numpy(intrinsics_new).cuda()
    slam = DPVO(cfg, args.network, ht=resolution[1], wd=resolution[0])

    for t, path, ipath in tqdm(
        islice(zip(ts, filenames, ifilenames), args.skip, None, args.stride), total=N
    ):
        assert int(os.path.splitext(os.path.basename(path))[0]) == ipath
        image_path = os.path.join(image_dir, path)
        image = cv2.imread(image_path)
        cv2.imshow("distorted", image)
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imshow("undistorted", image)
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()

        with Timer("SLAM", enabled=args.timeit):
            slam(t, image, intrinsics_new)
        cv2.waitKey(1)

    points = slam.pg.points_.cpu().numpy()[: slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[: slam.m]
    poses, tstamps = slam.terminate()

traj_est = PoseTrajectory3D(
    positions_xyz=poses[:, :3],
    orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
    timestamps=tstamps,
)

if args.scene is not None:
    scene = args.scene
else:
    scene = os.path.basename(args.data_dir)

# Use provided groundtruth
path_mocap0 = os.path.join(args.data_dir, "mav0", "mocap0", "data.csv")
mocap0 = np.loadtxt(path_mocap0, delimiter=",", skiprows=1).astype(np.float64)
mocap0[:, 0] = mocap0[:, 0] / 1e9
mocap0[:, 4:8] = mocap0[:, 4:8][::-1]
traj_ref = PoseTrajectory3D(
    positions_xyz=mocap0[:, 1:][:, :3],
    orientations_quat_wxyz=mocap0[:, 1:][:, [6, 3, 4, 5]],
    timestamps=mocap0[:, 0],
)
print(traj_ref.timestamps)
print(traj_est.timestamps)
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
        f"trajectory_plots/TUMVI_{scene}.pdf",
        align=True,
        correct_scale=True,
    )

if args.save_trajectory:
    os.makedirs("saved_trajectories", exist_ok=True)
    file_interface.write_tum_trajectory_file(
        f"saved_trajectories/TUMVI_{scene}.txt", traj_est
    )

if args.save_ply:
    save_ply(scene, points, colors)

if args.save_colmap:
    save_output_for_COLMAP(
        scene, traj_est, points, colors, *intrinsics, resolution[1], resolution[0]
    )
