import bisect
import copy
import math

import gtsam
import numpy as np
import torch
import torch.nn.functional as F
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation

from . import altcorr, fastba, lietorch
from . import geo as trans
from . import projective_ops as pops
from .lietorch import SE3
from .multi_sensor import MultiSensorState
from .net import VONet
from .patchgraph import PatchGraph
from .utils import Timer, flatmeshgrid

Id = SE3.Identity(1, device="cuda")


class DVIO:
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

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()

        ### event-based DBA (all times should have been offset by a time offset)
        self.state = MultiSensorState()
        # Pose estimation state
        self.last_t0 = 0  # t0 of the previous frame
        self.last_t1 = 0  # t1 of the previous frame
        self.cur_graph = None  # The current graph maintained by gtsam
        self.cur_result = None  # The result of gtsam optimization

        ### Record the marginalized matrix
        self.marg_factor = None  # This is the marginalization factor
        self.prior_factor_map = {}  # Used to store the prior map? See the set_prior() function
        self.cur_ii = None
        self.cur_jj = None
        self.cur_kk = None
        self.cur_target = None  # Stores all the optical flow under the current window
        self.cur_weight = (
            None  # Stores the weights of all the optical flow under the current window
        )

        self.imu_enabled = False  # Initialized to false, the IMU is not used during initialization until certain conditions are met (visual-inertial alignment is performed)
        self.ignore_imu = False

        # IMU-Camera Extrinsics. extrinsics, need to be set in the main .py
        self.Ti1c = None  # shape = (4,4) (the transformation matrix from camera to IMU)
        self.Tbc = None  # gtsam.Pose3 (the pose of the IMU-Camera Extrinsics represented in the form of gtsam.Pose3)
        self.tbg = None  # shape = (3) This is probably the transformation from gravity g to IMU

        self.reinit = False  # Re-initialize
        self.vi_init_time = 0.0  # The time of visual-inertial initialization
        self.vi_init_t1 = -1  # t1 of visual-inertial initialization
        self.vi_warmup = self.cfg.VI_WARM_UP_N
        # The visual warmup starts initialization after 12 frames
        self.init_pose_sigma = np.array([0.1, 0.1, 0.0001, 0.0001, 0.0001, 0.0001])
        self.init_bias_sigma = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

        # local optimization window
        self.t0 = 0  # Starting frame index
        self.t1 = (
            0  # Ending frame index (should also be the index of the current input data)
        )

        self.all_imu = None  # All IMU data (read in from the previous file)
        self.cur_imu_ii = 0  # The index of the current IMU data being processed
        self.is_init = False  # Is IMU initialized
        self.is_init_VI = False  # Is visual-inertial initialized

        # Whether to perform visual estimation only. When cfg.ENALBE_IMU is False, only visual estimation is performed and visual_only is true. When cfg.ENALBE_IMU is True, visual_only is False.
        self.visual_only = False
        self.visual_only_init = False

        self.high_freq_output = False  # True # Whether to perform high-frequency output

        # visualization/output
        self.plt_pos = [[], []]
        # X, Y
        self.plt_pos_ref = [[], []]
        # X, Y
        self.refTw = np.eye(4, 4)
        self.poses_save = []
        # Record poses

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure

            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    # Used to set prior_factor_map
    def set_prior(self, t0, t1):
        for i in range(t0, t0 + 2):
            self.prior_factor_map[i] = []
            init_pose_sigma = self.init_pose_sigma
            if len(self.init_pose_sigma.shape) > 1:
                init_pose_sigma = self.init_pose_sigma[i - t0]
            self.prior_factor_map[i].append(
                gtsam.PriorFactorPose3(
                    X(i),
                    self.state.wTbs[i],
                    gtsam.noiseModel.Diagonal.Sigmas(init_pose_sigma),
                )
            )
            if not self.ignore_imu:
                self.prior_factor_map[i].append(
                    gtsam.PriorFactorConstantBias(
                        B(i),
                        self.state.bs[i],
                        gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma),
                    )
                )
            self.last_t0 = t0
            self.last_t1 = t1

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict

            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace("module.", "")] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
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

        if self.cfg.LOOP_CLOSURE:
            """ interpolate missing poses """
            self.traj = {}
            for i in range(self.n):
                self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

            poses = [self.get_pose(t) for t in range(self.counter)]
            poses = lietorch.stack(poses, dim=0)
            poses = poses.inv().data.cpu().numpy()
            # Note that all are stored as w2c, and need to be changed to c2w when outputting
            tstamps = np.array(self.tlist, dtype=np.float64)
            if self.viewer is not None:
                self.viewer.join()
        else:
            # Get tstamps and poses from self.poses_save. The first element of each row in self.poses_save is the time, and the other seven are the pose
            poses = np.array(self.poses_save)[:, 1:]
            # Get timestamps
            tstamps = np.array(self.poses_save, dtype=np.float64)[:, 0]

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
        # The inserted ii is actually the patch index, kk
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])
        # self.ix[ii], which is self.ix[kk], is the index of ii

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            # If store is True, the edges to be deleted are stored in inactive edges
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

        with torch.amp.autocast("cuda", enabled=self.cfg.MIXED_PRECISION):
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
        i = self.n - self.cfg.KEYFRAME_INDEX - 1  # The 5th to last frame
        j = self.n - self.cfg.KEYFRAME_INDEX + 1  # The 3rd to last frame
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            # If motion is less than the threshold, it is not a keyframe
            k = self.n - self.cfg.KEYFRAME_INDEX  # The 4th to last frame
            t0 = self.pg.tstamps_[k - 1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k - 1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)
            # This will not be stored, because the motion is not enough, so it's not a keyframe

            # Reduce the indices after k
            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            # Perform data movement (from k to the current frame)
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

                # IMU data movement
                if i == k:
                    for iii in range(len(self.state.preintegrations_meas[i])):
                        dd = self.state.preintegrations_meas[i][iii]
                        # Get the IMU information of the kth frame (Acc, Omega, Delta_t, t)
                        if dd[2] > 0:
                            self.state.preintegrations[i - 1].integrateMeasurement(
                                dd[0], dd[1], dd[2]
                            )

                        self.state.preintegrations_meas[i - 1].append(dd)
                    self.state.preintegrations.pop(i)
                    self.state.preintegrations_meas.pop(i)

                    self.state.wTbs.pop(i)
                    self.state.bs.pop(i)
                    self.state.vs.pop(i)

            self.n -= 1  # Since one frame is deleted, move one frame forward
            self.m -= self.M
            # Subtract these patches to get the total number of patches

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        # When ii is 22 frames before the current frame, remove it
        # Remove edges falling outside the optimization window
        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (
                self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW)
            )
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)
        # This needs to be stored, because it is a keyframe, but it has slid out of the window

    # Global BA optimization
    def __run_global_BA(self):
        """Global bundle adjustment
        Includes both active and inactive edges"""
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        # self.pg.normalize()#! normalization, what is the purpose?
        lmbda = torch.as_tensor([1e-4], device="cuda")
        # Given value, unlike droid which needs to be calculated
        t0 = self.pg.ii.min().item()
        # It seems that it just adds global edges, and target weight and other information, and then performs global BA optimization, is there no big difference?
        # The main difference should be that eff_impl=False was used before, and eff_impl=True is used here
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

        # Mark that the current frame has run global BA optimization
        self.last_t0 = t0
        self.last_t1 = self.n

    def __run_DBA(self, target, weight, lmbda, ii, jj, kk, t0, t1, eff_impl):
        """Perform marginalization"""
        if self.last_t1 != t1 or self.last_t0 != t0:
            if self.last_t0 >= t0:
                if not eff_impl:
                    # If not doing GBA
                    t0 = self.last_t0
            else:
                # self.last_t0 < t0, which means the previous t0 is smaller than the current t0
                # print(f"Marginalization!!!!!!!!!!!{self.tstamp[t1-1]}")
                marg_paras = []
                # Construct a temporary factor graph (related to the old states) to obtain the marginalization information
                graph = gtsam.NonlinearFactorGraph()
                # Build a new graph to store the marginalization factors

                # Get the indices to be marginalized
                marg_idx = torch.logical_and(
                    self.last_t0 <= self.cur_ii, self.cur_ii < t0
                )
                # last_t0 <= ii < t0
                marg_idx2 = torch.logical_and(
                    self.cur_ii < self.last_t1 - 2, self.cur_jj < self.last_t1 - 2
                )
                marg_idx = torch.logical_and(marg_idx, marg_idx2)
                # Get the indices to be marginalized
                # marg_idx = (self.cur_ii == self.last_t0)
                marg_ii = self.cur_ii[marg_idx]
                marg_jj = self.cur_jj[marg_idx]
                marg_kk = self.cur_kk[marg_idx]
                marg_t0 = self.last_t0  # Remove the previous t0
                marg_t1 = t0 + 1  # The current t0+1
                if len(marg_ii) > 0:
                    marg_t0 = self.last_t0
                    marg_t1 = torch.max(marg_jj).item() + 1
                    marg_result = gtsam.Values()
                    for i in range(self.last_t0, marg_t1):
                        if i < t0:
                            marg_paras.append(X(i))
                        marg_result.insert(X(i), self.cur_result.atPose3(X(i)))

                    # The visual factors to be marginalized
                    marg_target = self.cur_target[:, marg_idx]
                    marg_weight = self.cur_weight[:, marg_idx]

                    # Next, add the visual factors for marginalization
                    bafactor = fastba.BAFactor()
                    # Initialize the class, ready to build visual factors
                    # It needs to be confirmed that what is obtained are marg_target, marg_weight, marg_ii, marg_jj, marg_t0, marg_t1
                    bafactor.init(
                        self.poses.data,
                        self.patches,
                        self.intrinsics,
                        marg_target,
                        marg_weight,
                        lmbda,
                        marg_ii,
                        marg_jj,
                        marg_kk,
                        self.M,
                        marg_t0,
                        marg_t1,
                        2,
                        eff_impl,
                    )
                    H = torch.zeros(
                        [(marg_t1 - marg_t0) * 6, (marg_t1 - marg_t0) * 6],
                        dtype=torch.float64,
                        device="cpu",
                    )
                    v = torch.zeros(
                        [(marg_t1 - marg_t0) * 6], dtype=torch.float64, device="cpu"
                    )
                    # Marginalization
                    bafactor.hessian(H, v)

                    for i in range(6):
                        H[i, i] += 0.00025  # for stability

                    # Hg,vg = BA2GTSAM(H,v,self.Tbc)
                    Hgg = gtsam.BA2GTSAM(H, v, self.Tbc)
                    # Convert BA's Hessian and v to gtsam's Hessian and v
                    Hg = Hgg[0 : (marg_t1 - marg_t0) * 6, 0 : (marg_t1 - marg_t0) * 6]
                    vg = Hgg[0 : (marg_t1 - marg_t0) * 6, (marg_t1 - marg_t0) * 6]
                    vis_factor = CustomHessianFactor(marg_result, Hg, vg)
                    # Build visual factors
                    # Add visual factors to the marginalization graph
                    # graph.push_back(vis_factor)

                # Add other factors to the marginalization graph
                for i in range(self.last_t0, marg_t1):
                    if i < t0:
                        if X(i) not in marg_paras:
                            marg_paras.append(X(i))
                        if not self.ignore_imu:
                            # If IMU is not ignored
                            marg_paras.append(V(i))
                            marg_paras.append(B(i))
                            graph.push_back(
                                gtsam.gtsam.CombinedImuFactor(
                                    X(i),
                                    V(i),
                                    X(i + 1),
                                    V(i + 1),
                                    B(i),
                                    B(i + 1),
                                    self.state.preintegrations[i],
                                )
                            )
                            # Add IMU factors

                # Get the prior map
                keys = self.prior_factor_map.keys()
                for i in sorted(keys):
                    if i < t0:
                        for iii in range(len(self.prior_factor_map[i])):
                            graph.push_back(self.prior_factor_map[i][iii])
                    del self.prior_factor_map[i]
                # If there is a marg_factor, add it to the graph
                if self.marg_factor is not None:
                    graph.push_back(self.marg_factor)

                # Optimize to get the marginalization factors
                self.marg_factor = gtsam.marginalizeOut(
                    graph, self.cur_result, marg_paras
                )

                # covariance inflation of IMU biases
                if self.reinit:
                    # If re-initializing
                    all_keys = self.marg_factor.keys()
                    for i in range(len(all_keys)):
                        if all_keys[i] == B(t0):
                            all_keys[i] = B(0)
                    graph = gtsam.NonlinearFactorGraph()
                    graph.push_back(self.marg_factor.rekey(all_keys))
                    b_l = gtsam.BetweenFactorConstantBias(
                        B(0),
                        B(t0),
                        gtsam.imuBias.ConstantBias(
                            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
                        ),
                        gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma),
                    )
                    graph.push_back(b_l)
                    result_tmp = self.marg_factor.linearizationPoint()
                    result_tmp.insert(B(0), result_tmp.atConstantBias(B(t0)))
                    self.marg_factor = gtsam.marginalizeOut(graph, result_tmp, [B(0)])
                    self.reinit = False

            # print(f'last_t0 :{self.last_t0},last_t1: {self.last_t1}; t0: {t0}, t1: {t1}; t0_temp: {t0_temp}, t1_temp: {t1_temp}')
            if not eff_impl:
                # If there is GBA, do not update
                self.last_t0 = t0
            self.last_t1 = t1

        """ optimization multi-sensor joint backend optimization """
        self.cur_graph = gtsam.NonlinearFactorGraph()
        # The current graph maintained during gtsam optimization (the naming is the same as for marginalization, but the meaning is different~)
        params = gtsam.LevenbergMarquardtParams()
        # ;params.setMaxIterations(1)

        # Now, start adding various factors to the gtsam graph
        # imu factor
        if not self.ignore_imu:
            for i in range(t0, t1):
                if i > t0:
                    imu_factor = gtsam.gtsam.CombinedImuFactor(
                        X(i - 1),
                        V(i - 1),
                        X(i),
                        V(i),
                        B(i - 1),
                        B(i),
                        self.state.preintegrations[i - 1],
                    )
                    self.cur_graph.add(imu_factor)
                    # Add IMU pre-integration

        # prior factor (this prior factor graph feels like it's from the IMU)
        keys = self.prior_factor_map.keys()
        for i in sorted(keys):
            if i >= t0 and i < t1:
                for iii in range(len(self.prior_factor_map[i])):
                    # Add prior factors
                    self.cur_graph.push_back(self.prior_factor_map[i][iii])

        # marginalization factor
        # if self.marg_factor is not None:
        if not eff_impl and self.marg_factor is not None:
            # No GBA and there is a marginalization factor
            self.cur_graph.push_back(self.marg_factor)
            # Add the factor calculated from the previous marginalization

        # Initialize a series of parameters in gtsam (some of which are used in marginalization~)
        # active_index = torch.logical_and(ii>=t0,jj>=t0)# The maximum value of ii and jj is t0+10. So here, those greater than or equal to t0 should be active_index
        self.cur_ii = ii  # [active_index]
        self.cur_jj = jj  # [active_index]
        self.cur_kk = kk  # [active_index]
        self.cur_target = target  # [:,active_index]
        self.cur_weight = weight  # [:,active_index]

        # Next, start building visual constraint factors and put them into the gtsam graph
        H = torch.zeros(
            [(t1 - t0) * 6, (t1 - t0) * 6], dtype=torch.float64, device="cpu"
        )
        v = torch.zeros([(t1 - t0) * 6], dtype=torch.float64, device="cpu")
        dx = torch.zeros([(t1 - t0) * 6], dtype=torch.float64, device="cpu")
        # Used to get the gtsam result and update dba

        bafactor = fastba.BAFactor()
        # Initialize the class, ready to build visual factors
        # Perform initialization
        # bafactor.init(self.poses.data, self.patches, self.intrinsics,
        #     target, weight, lmbda, ii, jj, kk, self.M, t0, t1, 2) # Note that keywords should not be written into the cuda code
        # ! The following is correct
        bafactor.init(
            self.poses.data,
            self.patches,
            self.intrinsics,
            self.cur_target,
            self.cur_weight,
            lmbda,
            self.cur_ii,
            self.cur_jj,
            self.cur_kk,
            self.M,
            t0,
            t1,
            2,
            eff_impl,
        )

        """ multi-sensor DBA iterations """
        for iter in range(2):
            if iter > 0:
                self.cur_graph.resize(self.cur_graph.size() - 1)
                # The size of the graph was originally 16, resized to 15
            bafactor.hessian(H, v)
            # camera frame
            Hgg = gtsam.BA2GTSAM(H, v, self.Tbc)
            Hg = Hgg[0 : (t1 - t0) * 6, 0 : (t1 - t0) * 6]
            vg = Hgg[0 : (t1 - t0) * 6, (t1 - t0) * 6]

            initial = gtsam.Values()
            for i in range(t0, t1):
                # Give initial values for the states
                initial.insert(X(i), self.state.wTbs[i])
                # the indice need to be handled
            initial_vis = copy.deepcopy(initial)
            vis_factor = CustomHessianFactor(initial_vis, Hg, vg)
            self.cur_graph.push_back(vis_factor)
            # Build visual factors based on the droid result

            if not self.ignore_imu:
                # If IMU is not ignored, add the bias
                for i in range(t0, t1):
                    initial.insert(B(i), self.state.bs[i])
                    initial.insert(V(i), self.state.vs[i])

            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self.cur_graph, initial, params
            )
            # Initialize gtsam optimizer
            self.cur_result = optimizer.optimize()
            # Optimize using gtsam

            # retraction and depth update
            for i in range(t0, t1):
                p0 = initial.atPose3(X(i))
                p1 = self.cur_result.atPose3(X(i))
                xi = gtsam.Pose3.Logmap(p0.inverse() * p1)
                dx[(i - t0) * 6 : (i - t0) * 6 + 6] = torch.tensor(xi)
                if not self.ignore_imu:
                    self.state.bs[i] = self.cur_result.atConstantBias(B(i))
                    self.state.vs[i] = self.cur_result.atVector(V(i))
                self.state.wTbs[i] = self.cur_result.atPose3(X(i))
            dx = torch.tensor(gtsam.GTSAM2BA(dx, self.Tbc))
            # Pose change
            _ = bafactor.retract(dx)
            # ! (needs double check) Update pose and patch for the next iteration
        del bafactor  # Release memory

    def update(self):
        with Timer("reproject", enabled=self.enable_timing, file=self.timing_file):
            coords = self.reproject()

        with torch.amp.autocast(device_type="cuda", enabled=self.cfg.MIXED_PRECISION):
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

        # Perform BA optimization
        with Timer("ba", enabled=self.enable_timing, file=self.timing_file):
            try:
                if self.imu_enabled:
                    # If using imu
                    t1 = self.n
                    eff_impl_flag = False

                    # Decide t0, full_target, full_weight, full_ii, full_jj, full_kk, eff_impl_flag based on different situations
                    if (
                        self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1
                    ).any() and not self.ran_global_ba[self.n]:
                        # If there are values in ii less than n-REMOVAL_WINDOW-1 (i.e., there is a loop closure match), and the current frame has not run global BA optimization, then run global BA optimization
                        eff_impl_flag = True  # Use an efficient implementation for global BA optimization
                        full_target = torch.cat(
                            (self.pg.target_inac, self.pg.target), dim=1
                        )
                        full_weight = torch.cat(
                            (self.pg.weight_inac, self.pg.weight), dim=1
                        )
                        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
                        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
                        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

                        # self.pg.normalize()#! normalization, what is the purpose?
                        t0 = self.pg.ii.min().item()

                        self.ran_global_ba[self.n] = (
                            True  # Mark that the current frame has run global BA optimization
                        )

                    else:
                        # Run local BA optimization
                        t0 = (
                            self.n - self.cfg.OPTIMIZATION_WINDOW
                            if self.is_initialized
                            else 1
                        )
                        t0 = max(t0, 1)
                        full_target = self.pg.target
                        full_weight = self.pg.weight
                        full_ii = self.pg.ii
                        full_jj = self.pg.jj
                        full_kk = self.pg.kk
                        eff_impl_flag = False  # Use an inefficient implementation for local BA optimization

                    self.__run_DBA(
                        target=full_target,
                        weight=full_weight,
                        lmbda=lmbda,
                        ii=full_ii,
                        jj=full_jj,
                        kk=full_kk,
                        t0=t0,
                        t1=t1,
                        eff_impl=eff_impl_flag,
                    )
                else:
                    # Run global BA optimization
                    # run global bundle adjustment if there exist long-range edges
                    if (
                        self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1
                    ).any() and not self.ran_global_ba[self.n]:
                        # If there are values in ii less than n-REMOVAL_WINDOW-1 (i.e., there is a loop closure match), and the current frame has not run global BA optimization, then run global BA optimization
                        self.__run_global_BA()
                    else:
                        # Run local BA optimization
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
                        # Additional records are needed
                        self.last_t0 = t0
                        self.last_t1 = self.n

            except Exception as _:
                print("Warning BA failed...")

            # Update point cloud
            points = pops.point_cloud(
                SE3(self.poses),
                self.patches[:, : self.m],
                self.intrinsics,
                self.ix[: self.m],
            )
            points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
            self.pg.points_[: len(points)] = points[:]

    def flow_viz_step(self):
        # [DEBUG]
        # dij = (self.ii - self.jj).abs()
        # assert (dij==0).sum().item() == len(torch.unique(self.kk))
        # [DEBUG]

        coords_est = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, self.ii, self.jj, self.kk
        )
        # p_ij (B,close_edges,P,P,2)
        self.flow_data[self.counter - 1] = {
            "ii": self.ii,
            "jj": self.jj,
            "kk": self.kk,
            "coords_est": coords_est,
            "img": self.image_,
            "n": self.n,
        }

    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_forw(self):
        # default: 13
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self):
        # default: 13
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def init_IMU(self):
        """initialize IMU states"""
        cur_t = float(self.tlist[self.pg.tstamps_[self.t0]])
        # Get the current timestamp, self.t0 is just the counter for keyframes as 0, corresponding to getting the global time index

        # find the first IMU data (the first IMU data after the time of the first frame)
        for i in range(len(self.all_imu)):
            # if self.all_imu[i][0] < cur_t - 1e-6: continue
            if self.all_imu[i][0] < cur_t:
                continue
            else:
                self.cur_imu_ii = i  # Record the index of the current IMU data
                break

        # For the range from t0 to t1
        for i in range(self.t0, self.t1):
            self.tlist[self.pg.tstamps_[i]]
            # Get the timestamp of the current frame
            if i == self.t0:
                # If it's the first frame
                self.state.init_first_state(cur_t, np.zeros(3), np.eye(3), np.zeros(3))
                # Initialize the state to 0 (except for time)
                # Then insert IMU data
                self.state.append_imu(
                    self.all_imu[self.cur_imu_ii][0],
                    self.all_imu[self.cur_imu_ii][4:7],
                    self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
                )
                self.cur_imu_ii += 1
                self.is_init = True
            else:
                cur_t = float(self.tlist[self.pg.tstamps_[i]])
                while self.all_imu[self.cur_imu_ii][0] < cur_t:
                    # Add all IMU data before the current time
                    self.state.append_imu(
                        self.all_imu[self.cur_imu_ii][0],
                        self.all_imu[self.cur_imu_ii][4:7],
                        self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
                    )
                    self.cur_imu_ii += 1
                self.state.append_imu(
                    cur_t,
                    self.all_imu[self.cur_imu_ii][4:7],
                    self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
                )
                # Here, insert the current time and IMU measurements greater than or equal to the current time, probably to ensure continuity
                self.state.append_img(cur_t)
                # Perform state insertion and update (this is unrelated to img)

                # The following is probably to insert another IMU data for continuity
                self.state.append_imu(
                    self.all_imu[self.cur_imu_ii][0],
                    self.all_imu[self.cur_imu_ii][4:7],
                    self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
                )

                self.cur_imu_ii += 1

            # Initialize as the transformation matrix from camera to IMU
            Twc = np.matmul(
                np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.02 * i], [0, 0, 0, 1]]
                ),
                self.Ti1c,
            )
            #  perturb the camera poses, which benefits the robustness of initial BA
            TTT = torch.tensor(np.linalg.inv(Twc))
            # Convert the inverse of the homogeneous transformation matrix Twc to a PyTorch tensor. i.e. Tcw
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            # Convert the rotation matrix to a quaternion
            t = TTT[:3, 3]
            if not self.imu_enabled:
                # If IMU is not used (if it is false, it is initialized to false, so it will be executed)
                self.pg.poses_[i] = torch.cat([t, q])
                # Assign values
                # gwp_donothing=1

    def __initialize(self):
        """initialize the DEIO system"""
        self.t0 = 0  # Starting frame index
        self.t1 = self.n
        # Ending frame index (should also be the index of the current input data)

        # Initialize imu
        self.init_IMU()

        # The following is to update the graph
        self.imu_enabled = False  # The imu is not used for graph updates here
        for itr in range(12):
            self.update()

        # initialization complete Mark initialization as successful
        # Flag for completion of initialization
        self.is_initialized = True

    def VisualIMUAlignment(self, t0, t1, ignore_lever, disable_scale=False):
        poses = SE3(self.pg.poses_)
        wTcs = poses.inv().matrix().cpu().numpy()

        if not ignore_lever:
            wTbs = np.matmul(wTcs, self.Tbc.inverse().matrix())
        else:
            T_tmp = self.Tbc.inverse().matrix()
            T_tmp[0:3, 3] = 0.0
            wTbs = np.matmul(wTcs, T_tmp)
        cost = 0.0

        # solveGyroscopeBias
        A = np.zeros([3, 3])
        b = np.zeros(3)
        H1 = np.zeros([15, 6], order="F", dtype=np.float64)
        H2 = np.zeros([15, 3], order="F", dtype=np.float64)
        H3 = np.zeros([15, 6], order="F", dtype=np.float64)
        H4 = np.zeros([15, 3], order="F", dtype=np.float64)
        H5 = np.zeros([15, 6], order="F", dtype=np.float64)
        # navstate wrt. bias
        H6 = np.zeros([15, 6], order="F", dtype=np.float64)
        for i in range(t0, t1 - 1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i + 1])
            # Rij = np.matmul(pose_i.rotation().matrix().T, pose_j.rotation().matrix())
            imu_factor = gtsam.gtsam.CombinedImuFactor(
                0, 1, 2, 3, 4, 5, self.state.preintegrations[i]
            )
            err = imu_factor.evaluateErrorCustom(
                pose_i,
                self.state.vs[i],
                pose_j,
                self.state.vs[i + 1],
                self.state.bs[i],
                self.state.bs[i + 1],
                H1,
                H2,
                H3,
                H4,
                H5,
                H6,
            )
            tmp_A = H5[0:3, 3:6]
            tmp_b = err[0:3]
            cost += np.dot(tmp_b, tmp_b)
            A += np.matmul(tmp_A.T, tmp_A)
            b += np.matmul(tmp_A.T, tmp_b)
        bg = -np.matmul(np.linalg.inv(A), b)

        for i in range(0, t1 - 1):
            pim = gtsam.PreintegratedCombinedMeasurements(
                self.state.params,
                gtsam.imuBias.ConstantBias(np.array([0.0, 0.0, 0.0]), bg),
            )
            for iii in range(len(self.state.preintegrations_meas[i])):
                dd = self.state.preintegrations_meas[i][iii]
                if dd[2] > 0:
                    pim.integrateMeasurement(dd[0], dd[1], dd[2])
            self.state.preintegrations[i] = pim
            self.state.bs[i] = gtsam.imuBias.ConstantBias(np.array([0.0, 0.0, 0.0]), bg)
        print("bg: ", bg)

        # linearAlignment
        all_frame_count = t1 - t0
        n_state = all_frame_count * 3 + 3 + 1
        A = np.zeros([n_state, n_state])
        b = np.zeros(n_state)
        i_count = 0
        for i in range(t0, t1 - 1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i + 1])
            R_i = pose_i.rotation().matrix()
            t_i = pose_i.translation()
            R_j = pose_j.rotation().matrix()
            t_j = pose_j.translation()
            pim = self.state.preintegrations[i]
            # tic = self.Tbc.translation()

            tmp_A = np.zeros([6, 10])
            tmp_b = np.zeros(6)
            dt = pim.deltaTij()
            tmp_A[0:3, 0:3] = -dt * np.eye(3, 3)
            tmp_A[0:3, 6:9] = R_i.T * dt * dt / 2
            tmp_A[0:3, 9] = np.matmul(R_i.T, t_j - t_i) / 100.0
            tmp_b[0:3] = pim.deltaPij()
            tmp_A[3:6, 0:3] = -np.eye(3, 3)
            tmp_A[3:6, 3:6] = np.matmul(R_i.T, R_j)
            tmp_A[3:6, 6:9] = R_i.T * dt
            tmp_b[3:6] = pim.deltaVij()

            r_A = np.matmul(tmp_A.T, tmp_A)
            r_b = np.matmul(tmp_A.T, tmp_b)

            A[i_count * 3 : i_count * 3 + 6, i_count * 3 : i_count * 3 + 6] += r_A[
                0:6, 0:6
            ]
            b[i_count * 3 : i_count * 3 + 6] += r_b[0:6]
            A[-4:, -4:] += r_A[-4:, -4:]
            b[-4:] += r_b[-4:]

            A[i_count * 3 : i_count * 3 + 6, n_state - 4 :] += r_A[0:6, -4:]
            A[n_state - 4 :, i_count * 3 : i_count * 3 + 6] += r_A[-4:, 0:6]
            i_count += 1

        A = A * 1000.0
        b = b * 1000.0
        x = np.matmul(np.linalg.inv(A), b)
        s = x[n_state - 1] / 100.0

        g = x[-4:-1]

        # RefineGravity
        g0 = g / np.linalg.norm(g) * 9.81
        # lx = np.zeros(3)
        # ly = np.zeros(3)
        n_state = all_frame_count * 3 + 2 + 1
        A = np.zeros([n_state, n_state])
        b = np.zeros(n_state)

        for k in range(4):
            aa = g / np.linalg.norm(g)
            tmp = np.array([0.0, 0.0, 1.0])

            bb = tmp - np.dot(aa, tmp) * aa
            bb /= np.linalg.norm(bb)
            cc = np.cross(aa, bb)
            bc = np.zeros([3, 2])
            bc[0:3, 0] = bb
            bc[0:3, 1] = cc
            lxly = bc

            i_count = 0
            for i in range(t0, t1 - 1):
                pose_i = gtsam.Pose3(wTbs[i])
                pose_j = gtsam.Pose3(wTbs[i + 1])
                R_i = pose_i.rotation().matrix()
                t_i = pose_i.translation()
                R_j = pose_j.rotation().matrix()
                t_j = pose_j.translation()
                tmp_A = np.zeros([6, 9])
                tmp_b = np.zeros(6)
                pim = self.state.preintegrations[i]
                dt = pim.deltaTij()

                tmp_A[0:3, 0:3] = -dt * np.eye(3, 3)
                tmp_A[0:3, 6:8] = np.matmul(R_i.T, lxly) * dt * dt / 2
                tmp_A[0:3, 8] = np.matmul(R_i.T, t_j - t_i) / 100.0
                tmp_b[0:3] = pim.deltaPij() - np.matmul(R_i.T, g0) * dt * dt / 2

                tmp_A[3:6, 0:3] = -np.eye(3)
                tmp_A[3:6, 3:6] = np.matmul(R_i.T, R_j)
                tmp_A[3:6, 6:8] = np.matmul(R_i.T, lxly) * dt
                tmp_b[3:6] = pim.deltaVij() - np.matmul(R_i.T, g0) * dt

                r_A = np.matmul(tmp_A.T, tmp_A)
                r_b = np.matmul(tmp_A.T, tmp_b)

                A[i_count * 3 : i_count * 3 + 6, i_count * 3 : i_count * 3 + 6] += r_A[
                    0:6, 0:6
                ]
                b[i_count * 3 : i_count * 3 + 6] += r_b[0:6]
                A[-3:, -3:] += r_A[-3:, -3:]
                b[-3:] += r_b[-3:]

                A[i_count * 3 : i_count * 3 + 6, n_state - 3 :] += r_A[0:6, -3:]
                A[n_state - 3 :, i_count * 3 : i_count * 3 + 6] += r_A[-3:, 0:6]
                i_count += 1

            A = A * 1000.0
            b = b * 1000.0
            x = np.matmul(np.linalg.inv(A), b)
            dg = x[-3:-1]
            g0 = g0 + np.matmul(lxly, dg)
            g0 = g0 / np.linalg.norm(g0) * 9.81
            s = x[-1] / 100.0
        print(s, g0, x)

        if disable_scale:
            s = 1.0

        # print('g,s:',g,s)
        print(f"\033[31m the calculate g {g} and scaler {s} \033[0m ")
        if math.fabs(np.linalg.norm(g) - 9.81) < 0.5 and s > 0:
            print("V-I successfully initialized!")

        # visualInitialAlign
        wTbs[:, 0:3, 3] *= s  # !!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(0, t1 - t0):
            self.state.vs[i + t0] = np.matmul(
                wTbs[i + t0, 0:3, 0:3], x[i * 3 : i * 3 + 3]
            )

        # g2R
        ng1 = g0 / np.linalg.norm(g0)
        ng2 = np.array([0, 0, 1.0])
        R0 = trans.FromTwoVectors(ng1, ng2)
        yaw = trans.R2ypr(R0)[0]
        R0 = np.matmul(trans.ypr2R(np.array([-yaw, 0, 0])), R0)

        # align for visualization
        # ppp = np.matmul(R0, wTbs[t1 - 1, 0:3, 3])
        # RRR = np.matmul(R0, wTbs[t1 - 1, 0:3, 0:3])

        g = np.matmul(R0, g0)
        for i in range(0, t1):
            wTbs[i, 0:3, 3] = np.matmul(R0, wTbs[i, 0:3, 3])
            wTbs[i, 0:3, 0:3] = np.matmul(R0, wTbs[i, 0:3, 0:3])
            self.state.vs[i] = np.matmul(R0, self.state.vs[i])
            self.state.wTbs[i] = gtsam.Pose3(wTbs[i])

        self.vi_init_t1 = t1
        self.vi_init_time = self.tlist[self.pg.tstamps_[t1 - 1]]

        if not ignore_lever:
            wTcs = np.matmul(wTbs, self.Tbc.matrix())
        else:
            T_tmp = self.Tbc.matrix()
            T_tmp[0:3, 3] = 0.0
            wTcs = np.matmul(wTbs, T_tmp)

        for i in range(0, t1):
            TTT = np.linalg.inv(wTcs[i])
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            t = torch.tensor(TTT[:3, 3])
            self.pg.poses_[i] = torch.cat([t, q])
            # self.disps[i] /= s
            # Rewrite the depth of all patches
            self.pg.patches_[i, :, 2] /= s

        # For all non-keyframe poses
        # s = torch.tensor(s).to(dtype=self.pg.poses_.dtype, device=self.pg.poses_.device)
        # for t, (t0, dP) in self.pg.delta.items():
        #     self.pg.delta[t] = (t0, dP.scale(s))

    def init_VI(self):
        """initialize the V-I system, referring to VIN-Fusion"""
        sum_g = np.zeros(3, dtype=np.float64)
        ccount = 0
        for i in range(self.t1 - 16, self.t1 - 1):
            dt = self.state.preintegrations[i].deltaTij()
            tmp_g = self.state.preintegrations[i].deltaVij() / dt
            sum_g += tmp_g
            ccount += 1
        aver_g = sum_g * 1.0 / ccount
        var_g = 0.0
        for i in range(self.t1 - 16, self.t1 - 1):
            dt = self.state.preintegrations[i].deltaTij()
            tmp_g = self.state.preintegrations[i].deltaVij() / dt
            var_g += np.linalg.norm(tmp_g - aver_g) ** 2
        var_g = math.sqrt(var_g / ccount)
        norm = np.linalg.norm(self.poses[-1].data[:3].cpu().numpy())
        if var_g < self.cfg.VI_INIT_VAR_G and norm < self.cfg.VI_INIT_NORM:
            print("IMU excitation not enough!", var_g, norm)
        else:
            poses = SE3(self.pg.poses_)
            self.plt_pos = [[], []]
            self.plt_pos_ref = [[], []]
            for i in range(0, self.t1):
                ppp = np.matmul(
                    poses[i].cpu().inv().matrix(), np.linalg.inv(self.Ti1c)
                )[0:3, 3]
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])

            if not self.visual_only:
                # If there is an IMU
                self.VisualIMUAlignment(self.t1 - 8, self.t1, ignore_lever=True)
                self.update()
                # Update graph
                self.VisualIMUAlignment(self.t1 - 8, self.t1, ignore_lever=False)
                self.update()
                # Update graph
                self.VisualIMUAlignment(self.t1 - 8, self.t1, ignore_lever=False)
                self.imu_enabled = True  # Turn on IMU after completing visual-inertial alignment (IMU will be used in BA update after this~)
            else:
                # The following can be ignored
                # Report error
                raise ValueError("Visual only initialization in init_VI???")
                self.visual_only_init = True  # Use only visual, not imu

            self.set_prior(self.last_t0, self.t1)

            self.plt_pos = [[], []]
            self.plt_pos_ref = [[], []]
            for i in range(0, self.t1):
                TTT = self.state.wTbs[i].matrix()
                ppp = TTT[0:3, 3]
                qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
                # Write results
                # self.result_file.writelines('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n'%(self.tlist[i],ppp[0],ppp[1],ppp[2]\
                #                             ,qqq[0],qqq[1],qqq[2],qqq[3]))
                self.poses_save.append(
                    [
                        self.tlist[i],
                        ppp[0],
                        ppp[1],
                        ppp[2],
                        qqq[0],
                        qqq[1],
                        qqq[2],
                        qqq[3],
                    ]
                )
                # Save the current pose to the list, #x,y,z xyzw

                TTTref = np.matmul(self.refTw, TTT)
                # for visualization
                ppp = TTTref[0:3, 3]
                qqq = Rotation.from_matrix(TTTref[:3, :3]).as_quat()
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])

            for itr in range(1):
                self.update()

    def VIO_update(self):
        """perform VIO/EIO update"""
        self.t1 = self.n

        # If there is IMU data, and the time difference between the current time and self.video.vi_init_time is greater than 5s, reset reinit to True
        if self.imu_enabled and (
            self.tlist[self.pg.tstamps_[self.t1 - 1]] - self.vi_init_time > 5.0
        ):
            self.reinit = True
            self.vi_init_time = 1e9

        ## new frame comes, append IMU (insert imu information)
        cur_t = float(self.tlist[self.pg.tstamps_[self.t1 - 1]])
        # Get the current image timestamp

        while True:
            # If the index time of the current IMU is less than the current time
            # Insert IMU data
            if self.cur_imu_ii >= len(self.all_imu):
                return

            if not (self.all_imu[self.cur_imu_ii][0] < cur_t):
                break

            self.state.append_imu(
                self.all_imu[self.cur_imu_ii][0],
                self.all_imu[self.cur_imu_ii][4:7],
                self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
            )
            self.cur_imu_ii += 1  # Update IMU index, keep adding

        # Insert IMU data (at this point self.all_imu[self.cur_imu_ii][0] >= cur_t, but inserting the time cur_t and the next frame's IMU data seems to ensure time continuity)
        self.state.append_imu(
            cur_t,
            self.all_imu[self.cur_imu_ii][4:7],
            self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
        )
        self.state.append_img(cur_t)
        # Update state after inserting IMU

        # To ensure continuity, insert IMU data again, ensuring that the data in the IMU at this time is definitely greater than or equal to the current timestamp
        self.state.append_imu(
            self.all_imu[self.cur_imu_ii][0],
            self.all_imu[self.cur_imu_ii][4:7],
            self.all_imu[self.cur_imu_ii][1:4] / 180 * math.pi,
        )
        self.cur_imu_ii += 1

        ## predict pose (<5 ms)
        if self.imu_enabled:
            # If IMU is used
            Twc = (self.state.wTbs[-1] * self.Tbc).matrix()
            # The latest Twb*Tbc
            TTT = torch.tensor(np.linalg.inv(Twc))
            # Get its inverse, which is Tcw
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            # Get the quaternion
            t = TTT[:3, 3]
            self.pg.poses_[self.t1 - 1] = torch.cat([t, q])
            # Initialize the current pose

        self.update()
        # Perform update operation

        # Save and output the pose result
        poses = SE3(self.pg.poses_)
        # Get pose
        TTT = np.matmul(
            poses[self.t1 - 1].cpu().inv().matrix(), np.linalg.inv(self.Ti1c)
        )
        # Get the latest frame and convert to body frame
        # If IMU is used or only visual is used and it has been initialized
        if self.imu_enabled or (self.visual_only and self.visual_only_init):
            ppp = TTT[0:3, 3]
            qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
            self.poses_save.append(
                [cur_t, ppp[0], ppp[1], ppp[2], qqq[0], qqq[1], qqq[2], qqq[3]]
            )
            # Save the current pose to the list, #x,y,z xyzw

        self.keyframe()
        # Keyframe management
        self.t1 = self.n  # Update t1, because it might have been affected at the keyframe! (This should mainly affect initialization!!!)

        ## Try visual-inertial initialization. try initializing VI (vi_warmup is the number of frames for visual initialization, which is 12)
        if (
            self.t1 > self.vi_warmup
            and self.vi_init_t1 < 0
            and self.tlist[-1] >= self.cfg.VI_WARM_UP_T
        ):
            # The number of frames is greater than 12 and the initialization time is less than 0
            if self.visual_only == 1:
                # IMU is not used, this is a passed parameter
                self.visual_only_init = True
            else:
                self.init_VI()
                # Visual initialization will only be performed when IMU is used

        # End of processing the current frame

    def __call__(self, tstamp, image, intrinsics, scale=1.0):
        """track new frame"""

        # Create a copy of the current frame
        current_frame = image.clone()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            # If classic loop closure (i.e., image matching) is enabled
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--buffer {self.N * 2}"'
            )

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        # if self.viz_flow:
        #     self.image_ = image.detach().cpu().permute((1, 2, 0)).numpy()

        image = 2 * (image[None, None] / 255.0) - 0.5

        # TODO patches with depth is available
        with Timer("patchify", enabled=self.enable_timing, file=self.timing_file):
            with torch.amp.autocast(
                device_type="cuda", enabled=self.cfg.MIXED_PRECISION
            ):
                fmap, gmap, imap, patches, _, clr = self.network.patchify(
                    image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                    return_color=True,
                )

        ### update state attributes ###
        self.tlist.append(tstamp)
        # Timestamp, global time timestamp
        self.pg.tstamps_[self.n] = self.counter
        # Just a number, the index of the global time corresponding to the keyframe is counted
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == "DAMPED_LINEAR":
                P1 = SE3(self.pg.poses_[self.n - 1])
                P2 = SE3(self.pg.poses_[self.n - 2])

                # To deal with varying camera hz
                *_, a, b, c = [1] * 3 + self.tlist
                fac = (c - b) / (b - a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
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
            # Visual is not initialized yet
            thres = 2.0 if scale == 1.0 else scale**2
            # TODO adapt thres for lite version
            if self.motion_probe() < thres:
                # TODO: replace by 8 pixels flow criterion (as described in 3.3 Initialization)
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1  # add one (key)frame
        self.m += self.M  # add patches per (key)frames to patch number

        if self.cfg.LOOP_CLOSURE:
            # If loop closure is enabled (this should be the loop closure implemented in DPVO)
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                # Get the edges for loop closure detection
                if lii.numel() > 0:
                    self.last_global_ba = self.n  # Mark the frame number of the last global BA optimization to control the frequency of global BA optimization
                    self.append_factors(lii, ljj)
                    # Add the edges for loop closure detection

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            # Not initialized yet and meets 8 frames
            self.__initialize()
        # Perform visual and inertial initialization
        elif self.is_initialized:
            self.VIO_update()
            # Perform VIO update, which also includes the above two functions

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            # If classic loop closure is enabled
            self.long_term_lc.attempt_loop_closure(self.n)
            # Attempt to perform loop closure
            self.long_term_lc.lc_callback()

        # if self.viz_flow:
        #     self.flow_viz_step()


def CustomHessianFactor(values: gtsam.Values, H: np.ndarray, v: np.ndarray):
    info_expand = np.zeros([H.shape[0] + 1, H.shape[1] + 1])
    info_expand[0:-1, 0:-1] = H
    info_expand[0:-1, -1] = v
    # This is meaningless.
    info_expand[-1, -1] = 0.0
    h_f = gtsam.HessianFactor(values.keys(), [6] * len(values.keys()), info_expand)
    l_c = gtsam.LinearContainerFactor(h_f, values)
    return l_c
