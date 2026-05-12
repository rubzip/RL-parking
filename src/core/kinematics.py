import numpy as np
from .models import CarState

class BikeKinematics:
    def __init__(self, length: float, delta_t: float = 1e-2):
        if length <= 0:
            raise ValueError(f"`length` should be a positive number. Given value: {length}")
        if delta_t <= 0:
            raise ValueError(f"`delta_t` should be a positive number. Given value: {delta_t}")
        self.length = length # Distance between wheels
        self.delta_t = delta_t # Time Step
    
    def update(self, current_state: CarState, a: float, phi: float) -> CarState:
        # 1. First derivative
        x_dot = current_state.v * np.cos(current_state.theta)
        y_dot = current_state.v * np.sin(current_state.theta)
        v_dot = a
        theta_dot = (current_state.v / self.length) * np.tan(phi)

        # 2. Update state
        new_x = current_state.x + x_dot * self.delta_t
        new_y = current_state.y + y_dot * self.delta_t
        new_v = current_state.v + v_dot * self.delta_t
        new_theta = current_state.theta + theta_dot * self.delta_t
        new_theta = new_theta % (2 * np.pi)
        
        return CarState(x=new_x, y=new_y, theta=new_theta, v=new_v)
