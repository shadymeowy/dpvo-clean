import numpy as np


class OnlineWelford:
    def __init__(self):
        self.n = 0
        self.mu_x = np.zeros(3)
        self.mu_y = np.zeros(3)
        self.H = np.zeros((3, 3))

    def update(self, x_n, y_n):
        self.n += 1

        dx = x_n - self.mu_x
        dy = y_n - self.mu_y

        self.mu_x = self.mu_x + dx / self.n
        self.mu_y = self.mu_y + dy / self.n

        self.H = self.H + np.outer(dy, dx)
