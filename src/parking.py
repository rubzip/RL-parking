import numpy as np
from .collisions import Rectangle


class Parking:
    def __init__(self, obstacles: list[Rectangle], parking_slot: Rectangle, car_init: tuple[float, float, float, float, float]):
        self.obstacles = obstacles
        self.parking_slot = parking_slot

        x, y, theta, w, h = car_init
        car = Rectangle(x, y, theta, w, h)
        if self.is_collision(car):
            raise ValueError("ERROR initializing car")
        self.x = x
        self.y = y
        self.theta = theta
    
    def is_collision(self, car: Rectangle) -> bool:
        """Car is crashing"""
        for obstacle in self.obstacles:
            if car.is_collision(obstacle):
                return True
        return False
    
    def get_parking_score(self, car: Rectangle) -> float:
        """Which proportion of car is inside parking_slot"""
        return car.proportion_in(self.parking_slot)
    
    def car_initialization(self) -> tuple[float]:
        """New car state"""
        return self.x, self.y, self.theta
