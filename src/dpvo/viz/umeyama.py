import numpy as np
from .welford import OnlineWelford


class OnlineUmeyama:
    def __init__(self, t_ref, pos_ref, time_diff=0.1):
        self.welford = OnlineWelford()
        self.t_ref = t_ref
        self.pos_ref = pos_ref

        self.idx_ref = 0
        self.time_diff = time_diff

    def update(self, t_est, pos_est):
        if self.idx_ref >= len(self.t_ref):
            return None

        while self.idx_ref + 1 < len(self.t_ref):
            dist_current = abs(self.t_ref[self.idx_ref] - t_est)
            dist_next = abs(self.t_ref[self.idx_ref + 1] - t_est)

            if dist_next <= dist_current:
                self.idx_ref += 1
            else:
                break

        if abs(self.t_ref[self.idx_ref] - t_est) <= self.time_diff:
            y_n = self.pos_ref[self.idx_ref]
            x_n = pos_est
            self.welford.update(x_n, y_n)

            matched_idx = self.idx_ref
            self.idx_ref += 1
            return matched_idx

        return None

    def get_alignment(self):
        H = self.welford.H
        mu_x = self.welford.mu_x
        mu_y = self.welford.mu_y
        return umeyama_alignment(H, mu_x, mu_y)


def umeyama_alignment(H, mu_x, mu_y):
    max_val = np.max(np.abs(H))

    if max_val < 1e-12:
        return np.eye(3), np.zeros(3)

    R = H / max_val
    U, _, Vh = np.linalg.svd(H)
    R = U @ Vh

    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vh

    t = mu_y - R @ mu_x
    return R, t
