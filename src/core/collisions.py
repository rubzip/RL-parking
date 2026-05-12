from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Rectangle:
    # Simple and efficent collisionable class. 
    # shape is a rectangle and only can interact with other rectangles
    # Class is not mutable
    x: float
    y: float
    theta: float
    w: float
    h: float

    def __post_init__(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)

        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_s", s)
        object.__setattr__(self, "_rot", np.array([[c, -s], [s, c]]))
        object.__setattr__(self, "_inv_rot", np.array([[c, s], [-s, c]]))

        # Precompute corners (safe because immutable)
        dx = np.array([ self.w/2,  self.w/2, -self.w/2, -self.w/2])
        dy = np.array([ self.h/2, -self.h/2, -self.h/2,  self.h/2])

        corners = np.stack((
            self.x + dx * c - dy * s,
            self.y + dx * s + dy * c
        ), axis=1)

        object.__setattr__(self, "_corners", corners)

    # ---------- geometry ----------

    @property
    def corners(self) -> np.ndarray:
        return self._corners

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        local = (points - np.array([self.x, self.y])) @ self._inv_rot.T
        in_w = np.abs(local[:, 0]) <= self.w / 2 + 1e-6
        in_h = np.abs(local[:, 1]) <= self.h / 2 + 1e-6
        return in_w & in_h

    def proportion_in(self, other: "Rectangle", grid_size: int = 5) -> float:
        xs = np.linspace(-self.w / 2, self.w / 2, grid_size)
        ys = np.linspace(-self.h / 2, self.h / 2, grid_size)
        xv, yv = np.meshgrid(xs, ys)
        local = np.stack([xv.ravel(), yv.ravel()], axis=1)
        world = local @ self._rot.T + np.array([self.x, self.y])
        return np.mean(other.contains_points(world))

    # ---------- collision ----------

    def is_collision(self, other: "Rectangle") -> bool:
        for corners in (self._corners, other._corners):
            for i in range(4):
                edge = corners[i] - corners[(i + 1) % 4]
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm == 0:
                    continue
                axis /= norm
                p1 = self._corners @ axis
                p2 = other._corners @ axis
                if p1.max() < p2.min() or p2.max() < p1.min():
                    return False
        return True
