from copy import copy
import numpy as np
from .collisions import Rectangle
from .models import CarState


class Parking:
    def __init__(self, obstacles: list[Rectangle], parking_slot: Rectangle, car_init: CarState):
        self.obstacles = obstacles
        self.parking_slot = parking_slot
        self.car_init = car_init
    
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
