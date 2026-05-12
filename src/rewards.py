from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .models import CarState
from .scenario import Scenario
from .collisions import Rectangle


class RewardFunction(ABC):
    @abstractmethod
    def compute(self, old_state: CarState, new_state: CarState, action: np.ndarray, scenario: Scenario, is_collision: bool) -> Tuple[float, bool, bool]:
        """Returns (reward, terminated, truncated)"""
        pass

class DenseParkingReward(RewardFunction):
    def __init__(self, collision_penalty: float = -100.0, success_bonus: float = 100.0):
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus

    def compute(self, old_state: CarState, new_state: CarState, action: np.ndarray, scenario: Scenario, is_collision: bool) -> Tuple[float, bool, bool]:
        if is_collision:
            return self.collision_penalty, True, False

        # Hardcoding car dimensions for the evaluation metric (can be passed in init)
        car_rect = Rectangle(new_state.x, new_state.y, new_state.theta, w=4.0, h=2.0)
        
        # Distance to slot penalty
        dist_to_slot = np.hypot(scenario.parking_slot.x - new_state.x, scenario.parking_slot.y - new_state.y)
        reward = -dist_to_slot * 0.05 
        
        # Check success condition
        proportion_in = car_rect.proportion_in(scenario.parking_slot)
        is_parked = False
        
        if proportion_in > 0.90 and abs(new_state.v) < 0.5:
            reward += self.success_bonus
            is_parked = True

        return reward, is_parked, False
