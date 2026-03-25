from multiprocessing import Process, Queue

import glviskit
import numpy as np
from scipy.spatial.transform import Rotation

from .umeyama import OnlineUmeyama


class Viz:
    def __init__(self, t_ref=None, pos_ref=None):
        # create window
        self.window = glviskit.create_window("dpvo2", 800, 800)

        # set spherical controller
        self.window.controller = glviskit.SphericalController()

        # set perspective projection with far plane at 1000 meters
        self.window.camera.perspective_fov(60, 60, 0.1, 1000.0)
        self.window.camera.distance = 10.0
        self.window.camera.set_axis_rotation(0, 1, 2, True, True, False)

        # create render lists
        self.list_ref = glviskit.create_render_list()
        self.list_est = glviskit.create_render_list()
        self.list_frustrum = glviskit.create_render_list()
        self.list_points = glviskit.create_render_list()

        # add render lists to window
        self.window.add_render_list(self.list_ref)
        self.window.add_render_list(self.list_est)
        self.window.add_render_list(self.list_frustrum)
        self.window.add_render_list(self.list_points)

        # path objects
        self.list_ref.size(3.0)
        self.list_ref.color([1.0, 0.0, 0.0, 1.0])
        self.path_ref = self.list_ref.path_begin()

        self.list_est.size(3.0)
        self.list_est.color([0.0, 1.0, 0.0, 1.0])
        self.path_est = self.list_est.path_begin()

        # set color and size for points
        self.list_points.size(3.0)
        self.list_points.color([1.0, 1.0, 1.0, 0.5])

        # draw camera like frustum
        self.list_frustrum.size(3.0)
        points = np.array(
            [
                [-1, -1, 1],
                [-1, +1, 1],
                [+1, +1, 1],
                [+1, -1, 1],
            ],
            dtype=np.float32,
        )
        points *= np.array([0.5, 0.5, 1.0])
        self.list_frustrum.color([0.0, 1.0, 0.0, 1.0])
        self.list_frustrum.circle(points)
        self.list_frustrum.circle([0, 0, 0])
        self.list_frustrum.line(points, np.zeros_like(points))
        self.list_frustrum.line(points, np.roll(points, 1, axis=0))

        # reference trajectory
        if t_ref is not None and pos_ref is not None:
            self.set_reference(t_ref, pos_ref)
        else:
            self.t_ref = None
            self.pos_ref = None
            self.umeyama = None

        # fps measurement
        self.frame_count = 0
        self.prev_time = glviskit.get_time_seconds()

    def set_reference(self, t_ref, pos_ref):
        # set reference trajectory
        self.t_ref = t_ref
        self.pos_ref = pos_ref

        # index of next reference pose to match
        self.idx_ref = 0

        # online umeyama alignment
        self.umeyama = OnlineUmeyama(t_ref, pos_ref)

    def move_frustrum(self, q, t):
        # move frustum to new position and orientation
        v = Rotation.from_quat(q).as_rotvec()
        self.list_frustrum.clear_instances()
        self.list_frustrum.add_instance(t, v)

    def add(self, t, p, q, ps=None):
        # add point to estimated path
        self.path_est.line_to(p)

        # update online alignment
        if self.umeyama is not None:
            self.umeyama.update(t, p)

        # if reference trajectory is set
        # add poses up to current time to reference path
        if self.pos_ref is not None:
            while self.idx_ref < len(self.t_ref) and self.t_ref[self.idx_ref] <= t:
                self.path_ref.line_to(self.pos_ref[self.idx_ref])
                self.idx_ref += 1

        # move frustum to current estimated pose
        self.move_frustrum(q, p)

        # move camera to current estimated position
        self.window.camera.position = p

        # align reference trajectory
        if self.umeyama is not None:
            R_align, t_align = self.umeyama.get_alignment()
            self.align_reference(R_align, t_align)

        # given points in camera frame, transform to world frame and add to visualization
        if ps is not None:
            R = Rotation.from_quat(q).as_matrix()
            ps_world = (R @ ps.T).T + p
            ps_world = np.ascontiguousarray(ps_world, dtype=np.float32)
            self.list_points.circle(ps_world)

    def align_reference(self, R, t):
        # invert transformation
        R_inv = R.T
        t_inv = -R_inv @ t

        # set pose of reference trajectory
        self.list_ref.clear_instances()
        v_inv = Rotation.from_matrix(R_inv).as_rotvec()
        self.list_ref.add_instance(t_inv, v_inv)

    def fps(self):
        # print FPS every second
        self.frame_count += 1
        curr_time = glviskit.get_time_seconds()

        if curr_time - self.prev_time >= 1.0:
            self.frame_count = 0
            self.prev_time = curr_time


class ProcessViz:
    def __init__(self, *args, **kwargs):
        # create a queue for communication between processes
        self.queue = Queue()

        # create a separate process for visualization
        self.process = Process(
            target=self._process_loop, args=(self.queue, args, kwargs)
        )
        self.process.daemon = True
        self.process.start()

    def _process_loop(self, queue, args, kwargs):
        # create visualization instance
        viz = Viz(*args, **kwargs)

        while glviskit.loop():
            # process all data in queue
            while not queue.empty():
                data = queue.get()

                # signal to break loop
                if data is None:
                    return

                # add data to visualization
                viz.add(*data)

            # measure and print FPS
            viz.fps()

    def add(self, t, p, q, ps=None):
        # add data to queue for process
        self.queue.put((t, p, q, ps))

    def close(self):
        # signal process to terminate
        self.queue.put(None)
        # wait for process to finish
        self.process.join()
