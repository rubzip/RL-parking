import numpy as np


class Rectangle:
    def __init__(self, x: float, y: float, theta: float, w: float, h: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.w = w
        self.h = h
        
        # Pre-calculate rotation matrix components to save time during high-freq calls
        self.c, self.s = np.cos(self.theta), np.sin(self.theta)
        self.rot_matrix = np.array([[self.c, -self.s], [self.s, self.c]])
        self.inv_rot = np.array([[self.c, self.s], [-self.s, self.c]])
        self.corners = None
        self.sample_points = None

    def _get_sample_points(self, grid_size: int = 5) -> np.ndarray:
        """Generates a grid of points inside this rectangle."""
        if self.sample_points is not None and len(self.sample_points.shape) == grid_size:
            return self.sample_points
        x_range = np.linspace(-self.w/2, self.w/2, grid_size)
        y_range = np.linspace(-self.h/2, self.h/2, grid_size)
        xv, yv = np.meshgrid(x_range, y_range)
        local_points = np.stack([xv.ravel(), yv.ravel()], axis=1)
        self.sample_points = (local_points @ self.rot_matrix.T) + np.array([self.x, self.y])
        return self.sample_points

    def _contains_points(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized check: inputs (N, 2) array of points.
        Returns boolean array (N,) indicating which points are inside.
        """
        local_points = points - np.array([self.x, self.y])
        local_aligned = local_points @ self.inv_rot.T
        in_w = np.abs(local_aligned[:, 0]) <= (self.w / 2 + 1e-6)
        in_h = np.abs(local_aligned[:, 1]) <= (self.h / 2 + 1e-6)
        return in_w & in_h

    def proportion_in(self, other: "Rectangle", grid_size: int = 5) -> float:
        """
        Optimized overlap calculation.
        """
        my_points = self._get_sample_points(grid_size)
        mask = other._contains_points(my_points)
        return np.mean(mask)

    def _get_corners(self) -> np.ndarray:
        if self.corners is not None:
            return self.corners
        dx = np.array([self.w/2, self.w/2, -self.w/2, -self.w/2])
        dy = np.array([self.h/2, -self.h/2, -self.h/2, self.h/2])
        x_corners = self.x + dx * self.c - dy * self.s
        y_corners = self.y + dx * self.s + dy * self.c
        self.corners = np.stack((x_corners, y_corners), axis=1)
        return self.corners

    def is_collision(self, other: "Rectangle") -> bool:
        """SAT Collision Detection."""
        corners1 = self._get_corners()
        corners2 = other._get_corners()
        for corners in [corners1, corners2]:
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                edge = p1 - p2
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm == 0: continue
                axis /= norm
                proj1 = corners1 @ axis
                proj2 = corners2 @ axis
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    return False
        return True
