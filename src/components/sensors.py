from abc import ABC, abstractmethod

import gymnasium.spaces as spaces
import numpy as np

from .models import CarState
from .scenario import Scenario
from .parking import _ray_segment_intersection


class Sensor(ABC):
    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def observe(self, car_state: CarState, scenario: Scenario) -> np.ndarray:
        pass

class LidarSensor(Sensor):
    def __init__(self, n_points: int = 12, max_range: float = 10.0):
        self.n_points = n_points
        self.max_range = max_range

    @property
    def observation_space(self) -> spaces.Space:
        # Returns normalized distances [-1, 1]
        return spaces.Box(low=-1.0, high=1.0, shape=(self.n_points,), dtype=np.float32)

    def observe(self, car_state: CarState, scenario: Scenario) -> np.ndarray:
        origin = np.array([car_state.x, car_state.y])
        angles = car_state.theta + np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        distances = np.full(len(angles), self.max_range)

        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            for rect in scenario.obstacles:
                corners = rect.corners
                for j in range(4):
                    p1 = corners[j]
                    p2 = corners[(j + 1) % 4]
                    t = _ray_segment_intersection(origin, ray_dir, p1, p2)
                    if t is not None and t < distances[i]:
                        distances[i] = t
                        
        # Normalize to [-1, 1] for neural network stability
        return (2 * distances - self.max_range) / self.max_range
