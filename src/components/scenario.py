from dataclasses import dataclass
from typing import List

from .collisions import Rectangle
from .models import CarState


@dataclass
class Scenario:
    obstacles: List[Rectangle]
    parking_slot: Rectangle
    initial_car_state: CarState

    def is_out_of_bounds(self, car: Rectangle, x_bounds: tuple, y_bounds: tuple) -> bool:
        """Simple check to ensure the car hasn't driven off the map."""
        return not (x_bounds[0] <= car.x <= x_bounds[1] and y_bounds[0] <= car.y <= y_bounds[1])
