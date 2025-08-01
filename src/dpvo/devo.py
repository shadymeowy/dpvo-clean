import random

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import eVONet
from .patchgraph import PatchGraph
from .utils import Timer, flatmeshgrid

Id = SE3.Identity(1, device="cuda")


class DEVO:
    def __init__(
        self,
        cfg,
        network,
        ht=480,
        wd=640,
        viz=False,
        show=False,
        enable_timing=False,
        timing_file=None,
        **kwargs,
    ):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = enable_timing
        self.timing_file = timing_file
        self.show = show
        self.concatenated_image = None

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht  # image height
        self.wd = wd  # image width

        self.flow_data = {}
        self.images = {}

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36  # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000  # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE  # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(
            1, self.mem, 128, int(ht // 1), int(wd // 1), **kwargs
        )
        self.fmap2_ = torch.zeros(
            1, self.mem, 128, int(ht // 4), int(wd // 4), **kwargs
        )

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure

            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            print(f"Loading from {network}")
            checkpoint = torch.load(network)
            self.network = eVONet(
                patch_selector=self.cfg.PATCH_SELECTOR,
            )
            if "model_state_dict" in checkpoint:
                self.network.load_state_dict(checkpoint["model_state_dict"])
            else:
                # legacy
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if "update.lmbda" not in k:
                        new_state_dict[k.replace("module.", "")] = v
                self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.dim_inet
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_, self.pg.poses_, self.pg.points_, self.pg.colors_, intrinsics_
        )

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i].item()] = self.pg.poses_[i]

        if self.is_initialized:
            poses = [self.get_pose(t) for t in range(self.counter)]
            poses = lietorch.stack(poses, dim=0)
            poses = poses.inv().data.cpu().numpy()
        else:
            print(
                "Warning: Model is not initialized. Using Identity."
            )  # eval still runs bug
            id = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            poses = np.array([id for t in range(self.counter)])
            poses[:, :3] = (
                poses[:, :3] + np.random.randn(self.counter, 3) * 0.01
            )  # small random trans

        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """local correlation volume"""
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """reproject patch k from i -> j"""
        (ii, jj, kk) = (
            indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        )
        coords = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
        )
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat(
                (self.pg.weight_inac, self.pg.weight[:, m]), dim=1
            )
            self.pg.target_inac = torch.cat(
                (self.pg.target_inac, self.pg.target[:, m]), dim=1
            )
        self.pg.weight = self.pg.weight[:, ~m]
        self.pg.target = self.pg.target[:, ~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:, ~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """kinda hacky way to ensure enough motion for initialization"""
        kk = torch.arange(self.m - self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with torch.amp.autocast(
            "cuda", device_type="cuda", enabled=self.cfg.MIXED_PRECISION
        ):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:, kk % (self.M * self.pmem)]
            net, (delta, weight, _) = self.network.update(
                net, ctx, corr, None, ii, jj, kk
            )

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5
        )
        return flow.mean().item()

    def keyframe(self):
        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k - 1].item()
            t1 = self.pg.tstamps_[k].item()

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k - 1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n - 1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i + 1]
                self.pg.colors_[i] = self.pg.colors_[i + 1]
                self.pg.poses_[i] = self.pg.poses_[i + 1]
                self.pg.patches_[i] = self.pg.patches_[i + 1]
                self.images[i] = self.images[i + 1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i + 1]

                self.imap_[i % self.pmem] = self.imap_[(i + 1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i + 1) % self.pmem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0, (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0, (i + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = (
            self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
        )  # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (
                self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW)
            )
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """Global bundle adjustment
        Includes both active and inactive edges"""
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(
            self.poses,
            self.patches,
            self.intrinsics,
            full_target,
            full_weight,
            lmbda,
            full_ii,
            full_jj,
            full_kk,
            t0,
            self.n,
            M=self.M,
            iterations=2,
            eff_impl=True,
        )
        self.ran_global_ba[self.n] = True

    def update(self):
        with Timer("reproject", enabled=self.enable_timing, file=self.timing_file):
            coords = self.reproject()

        with torch.amp.autocast(
            "cuda", device_type="cuda", enabled=self.cfg.MIXED_PRECISION
        ):
            with Timer("corr", enabled=self.enable_timing, file=self.timing_file):
                corr = self.corr(coords)
            with Timer("gru", enabled=self.enable_timing, file=self.timing_file):
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = self.network.update(
                    self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk
                )

        lmbda = torch.as_tensor([1e-4], device="cuda")
        weight = weight.float()
        target = coords[..., self.P // 2, self.P // 2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("ba", enabled=self.enable_timing, file=self.timing_file):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (
                    self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1
                ).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = (
                        self.n - self.cfg.OPTIMIZATION_WINDOW
                        if self.is_initialized
                        else 1
                    )
                    t0 = max(t0, 1)
                    fastba.BA(
                        self.poses,
                        self.patches,
                        self.intrinsics,
                        target,
                        weight,
                        lmbda,
                        self.pg.ii,
                        self.pg.jj,
                        self.pg.kk,
                        t0,
                        self.n,
                        M=self.M,
                        iterations=2,
                        eff_impl=False,
                    )
            except Exception as _:
                print("Warning BA failed...")

            points = pops.point_cloud(
                SE3(self.poses),
                self.patches[:, : self.m],
                self.intrinsics,
                self.ix[: self.m],
            )
            points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
            self.pg.points_[: len(points)] = points[:]

    def __edges_forw(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, tstamp, image, intrinsics, scale=1.0):
        """track new frame"""

        # Create a copy of the current frame
        current_frame = image.clone()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N * 2}"'
            )

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        if False:
            image = 2 * (image[None, None] / 255.0) - 0.5
        else:
            image = image[None, None]

            if self.n == 0:
                nonzero_ev = image != 0.0
                zero_ev = ~nonzero_ev
                num_nonzeros = nonzero_ev.sum().item()
                num_zeros = zero_ev.sum().item()

                if (
                    num_nonzeros / (num_zeros + num_nonzeros) < 2e-2
                ):  # TODO eval hyperparam (add to config.py)
                    print(f"skip voxel at {tstamp} due to lack of events!")
                    return

            b, n, v, h, w = image.shape
            flatten_image = image.view(b, n, -1)

            if self.cfg.NORM.lower() == "none":
                pass
            elif self.cfg.NORM.lower() == "rescale" or self.cfg.NORM.lower() == "norm":
                pos = flatten_image > 0.0
                neg = flatten_image < 0.0
                vx_max = (
                    torch.Tensor([1]).to("cuda")
                    if pos.sum().item() == 0
                    else flatten_image[pos].max(dim=-1, keepdim=True)[0]
                )
                vx_min = (
                    torch.Tensor([1]).to("cuda")
                    if neg.sum().item() == 0
                    else flatten_image[neg].min(dim=-1, keepdim=True)[0]
                )

                if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                    print(f"empty voxel at {tstamp}!")
                    return
                flatten_image[pos] = flatten_image[pos] / vx_max
                flatten_image[neg] = flatten_image[neg] / -vx_min
            elif self.cfg.NORM.lower() == "standard" or self.cfg.NORM.lower() == "std":
                nonzero_ev = flatten_image != 0.0
                num_nonzeros = nonzero_ev.sum(dim=-1)
                if torch.all(num_nonzeros > 0):
                    mean = (
                        torch.sum(flatten_image, dim=-1, dtype=torch.float32)
                        / num_nonzeros
                    )  # force torch.float32 to prevent overflows when using 16-bit precision
                    stddev = torch.sqrt(
                        torch.sum(flatten_image**2, dim=-1, dtype=torch.float32)
                        / num_nonzeros
                        - mean**2
                    )
                    mask = nonzero_ev.type_as(flatten_image)
                    flatten_image = (
                        mask * (flatten_image - mean[..., None]) / stddev[..., None]
                    )
            else:
                print(f"{self.cfg.NORM} not implemented")
                raise NotImplementedError

            image = flatten_image.view(b, n, v, h, w)

        if image.shape[-1] == 346:
            image = image[..., 1:-1]  # hack for MVSEC, FPV,...

        with Timer("patchify", enabled=self.enable_timing, file=self.timing_file):
            with torch.amp.autocast(
                "cuda", device_type="cuda", enabled=self.cfg.MIXED_PRECISION
            ):
                fmap, gmap, imap, patches, _, clr = self.network.patchify(
                    image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    return_color=True,
                    scorer_eval_mode=self.cfg.SCORER_EVAL_MODE,
                    scorer_eval_use_grid=self.cfg.SCORER_EVAL_USE_GRID,
                )

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0, :, [0, 0, 0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == "DAMPED_LINEAR":
                P1 = SE3(self.pg.poses_[self.n - 1])
                P2 = SE3(self.pg.poses_[self.n - 2])

                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n - 1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

        self.pg.patches_[self.n] = patches
        self.images[self.n] = current_frame.cpu().numpy().transpose(1, 2, 0)

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        if self.n > 0 and not self.is_initialized:
            thres = 2.0 if scale == 1.0 else scale**2
            if self.motion_probe() < thres:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()

        if self.show:
            try:
                self.visualize_patches()
            except Exception as e:
                print(f"Error in visualize_patches: {e}")

    def visualize_patches(self):
        return
        # Get the graph
        ii = self.pg.ii.cpu().numpy()
        jj = self.pg.jj.cpu().numpy()
        kk = self.pg.kk.cpu().numpy()

        choose_rand = False

        if choose_rand:
            # First choose a random source frame to visualize
            source_idx = random.randint(
                self.pg.ii.min().item(), self.pg.ii.max().item() - 1
            )
            roi_tmp = np.where(ii == source_idx)

            # Choose a random target frame to visualize
            target_idx = random.randint(jj[roi_tmp].min(), jj[roi_tmp].max())
        else:
            source_idx = self.pg.ii.max().item() - 1
            target_idx = source_idx + 1

        # Now, get the roi
        roi = np.where((ii == source_idx) & (jj == target_idx))

        # Get the frames to visualize
        img1_vis = self.images[source_idx][:, :, -1]
        img1_vis = 255 * (img1_vis - img1_vis.min()) / (img1_vis.max() - img1_vis.min())
        img1_vis = img1_vis.astype(np.uint8)
        img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)
        img2_vis = self.images[target_idx][:, :, -1]
        img2_vis = 255 * (img2_vis - img2_vis.min()) / (img2_vis.max() - img2_vis.min())
        img2_vis = img2_vis.astype(np.uint8)
        img2_vis = cv2.cvtColor(img2_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            img1_vis,
            f"Source Idx : {source_idx}",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img2_vis,
            f"Target Idx : {target_idx}",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        concatenated = np.hstack((img1_vis, img2_vis))

        connections = kk[roi]

        for con_idx, connection_id in enumerate(connections[:20]):
            source_coord = (
                self.pg.patches_[source_idx, con_idx, :2, 2, 2].cpu().numpy() * 4
            ).astype(int)
            target_coord = (
                self.pg.target[0, roi[0][con_idx]].cpu().numpy() * 4
            ).astype(int)
            weight = self.pg.weight[0, roi[0][con_idx]].cpu().numpy().sum() * 0.5

            B = int(255 * (0.5 - abs(0.5 - weight)))
            G = int(255 * weight)
            R = int(255 * (1 - weight))

            BGR_margin = 255 - max(B, G, R)
            B += BGR_margin
            G += BGR_margin
            R += BGR_margin

            color = (B, G, R)

            center1 = (source_coord[0], source_coord[1])
            center2 = (target_coord[0] + img1_vis.shape[1], target_coord[1])
            cv2.line(concatenated, center1, center2, color, 2, cv2.LINE_AA)
            # cv2.circle(concatenated, center1, 3, color, -1)
            # cv2.circle(concatenated, center2, 3, color, -1)

        cv2.imshow("matches", concatenated)
        self.concatenated_image = concatenated
        key = cv2.waitKey(1)
        if key == ord("a"):
            exit()
