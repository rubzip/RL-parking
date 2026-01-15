from copy import copy
import numpy as np
from .collisions import Rectangle
from .models import CarState


def _ray_segment_intersection(ray_origin, ray_dir, p1, p2):
    """
    Ray-segment intersection.
    Returns distance t if hit, else None.
    """
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]])

    denom = np.dot(v2, v3)
    if abs(denom) < 1e-8:
        return None  # Parallel

    t = np.cross(v2, v1) / denom
    u = np.dot(v1, v3) / denom

    if t >= 0 and 0.0 <= u <= 1.0:
        return t

    return None


class Parking:
    def __init__(self, obstacles: list[Rectangle], parking_slot: Rectangle, car_init: CarState, n_points_lidar: int = 12):
        self.obstacles = obstacles
        self.parking_slot = parking_slot
        self.car_init = car_init
        self.n_points_lidar = n_points_lidar
    
    def is_collision(self, car: Rectangle) -> bool:
        """Car is crashing"""
        for obstacle in self.obstacles:
            if car.is_collision(obstacle):
                return True
        return False
    
    def get_parking_score(self, car: Rectangle) -> float:
        """Which proportion of car is inside parking_slot"""
        return car.proportion_in(self.parking_slot)
    
    def get_new_car(self) -> CarState:
        """New car state"""
        return copy(self.car_init)

    def get_parking_vector(self, car: Rectangle) -> np.ndarray:
        return np.array([
            self.parking_slot.x - car.x,
            self.parking_slot.y - car.y
        ])
    
    def lidar_scan(
        self,
        car: Rectangle,
        max_range: float = 10,
        normalize: bool = True
    ) -> np.ndarray:
        origin = np.array([car.x, car.y])
        angles = car.theta + np.linspace(0, 2 * np.pi, self.n_points_lidar, endpoint=False)
        distances = np.full(len(angles), max_range)

        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            for rect in self.obstacles:
                corners = rect.corners
                for j in range(4):
                    p1 = corners[j]
                    p2 = corners[(j + 1) % 4]

                    t = _ray_segment_intersection(origin, ray_dir, p1, p2)
                    if t is not None and t < distances[i]:
                        distances[i] = t
        if normalize: # normalizes to [-1, 1]
            return (2 * distances - max_range) / max_range
        return distances
