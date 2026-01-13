import numpy as np


class BykeKinematics:
    def __init__(self, length: float, delta_t: float = 1e-2):
        self.length = length # Distance between wheels
        self.delta_t = delta_t
    
    def update(self, x: float, y: float, v: float, theta: float, a: float, phi: float) -> tuple[float, float, float, float]:
        # 1. First derivative
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = (v / self.length) * np.tan(phi)
        v_dot = a

        # 2. Update state
        new_x = x + x_dot * self.delta_t
        new_y = y + y_dot * self.delta_t
        new_theta = theta + theta_dot * self.delta_t
        new_v = v + v_dot * self.delta_t

        return new_x, new_y, new_theta, new_v
