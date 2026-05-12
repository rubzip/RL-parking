from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..core.collisions import Rectangle
from ..core.models import CarState

@dataclass
class Scenario:
    obstacles: List[Rectangle]
    parking_slot: Rectangle
    initial_car_state: CarState
    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]

    def is_out_of_bounds(self, car: Rectangle) -> bool:
        return not (self.x_bounds[0] <= car.x <= self.x_bounds[1] and 
                    self.y_bounds[0] <= car.y <= self.y_bounds[1])


class ScenarioBuilder:
    """
    The 'Intermediate Factory'.
    Contains hidden (_private) functions to modularly construct scenarios.
    """
    def __init__(self):
        self._obstacles: List[Rectangle] = []
        self._parking_slot: Rectangle = None
        self._initial_state: CarState = None
        self._x_bounds = (-20.0, 20.0)
        self._y_bounds = (-20.0, 20.0)

    # --- Hidden / Private Construction Methods ---

    def _add_obstacle(self, x: float, y: float, theta: float, w: float, h: float) -> 'ScenarioBuilder':
        self._obstacles.append(Rectangle(x=x, y=y, theta=theta, w=w, h=h))
        return self

    def _add_room_walls(self, width: float, height: float, thickness: float = 1.0) -> 'ScenarioBuilder':
        """Wraps the environment in 4 solid walls."""
        self._add_obstacle(0, height/2, 0, width, thickness)    # Top
        self._add_obstacle(0, -height/2, 0, width, thickness)   # Bottom
        self._add_obstacle(-width/2, 0, 0, thickness, height)   # Left
        self._add_obstacle(width/2, 0, 0, thickness, height)    # Right
        return self

    def _add_parked_car(self, x: float, y: float, theta: float, w: float = 4.8, h: float = 2.0) -> 'ScenarioBuilder':
        """Semantic wrapper for adding a vehicle obstacle."""
        return self._add_obstacle(x=x, y=y, theta=theta, w=w, h=h)

    def _set_parking_slot(self, x: float, y: float, theta: float, w: float, h: float) -> 'ScenarioBuilder':
        self._parking_slot = Rectangle(x=x, y=y, theta=theta, w=w, h=h)
        return self

    def _set_initial_state(self, x: float, y: float, theta: float, v: float = 0.0) -> 'ScenarioBuilder':
        self._initial_state = CarState(x=x, y=y, theta=theta, v=v)
        return self

    def _set_bounds(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> 'ScenarioBuilder':
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        return self

    # --- Final Assembly ---

    def build(self) -> Scenario:
        if not self._parking_slot or not self._initial_state:
            raise ValueError("Scenario requires at least a parking slot and an initial state.")
        
        return Scenario(
            obstacles=self._obstacles,
            parking_slot=self._parking_slot,
            initial_car_state=self._initial_state,
            x_bounds=self._x_bounds,
            y_bounds=self._y_bounds
        )


class LevelCreator:
    """
    The 'Level Creator'.
    Uses the ScenarioBuilder to orchestrate specific training levels.
    """

    @staticmethod
    def level_0_dumb(car_init: tuple = (-10.0, 10.0, 0.0), slot_init: tuple = (0.0, 0.0, 0.0)) -> Scenario:
        """0. Dumb Scenario: Just a parking slot and car."""
        builder = ScenarioBuilder()
        return (builder
                ._set_parking_slot(x=slot_init[0], y=slot_init[1], theta=slot_init[2], w=6.0, h=3.0)
                ._set_initial_state(x=car_init[0], y=car_init[1], theta=car_init[2])
                .build())

    @staticmethod
    def level_1_walls(room_size: float = 20.0) -> Scenario:
        """1. Walls Scenario: Dumb scenario but trapped in a box."""
        builder = ScenarioBuilder()
        return (builder
                ._set_parking_slot(x=0.0, y=0.0, theta=0.0, w=6.0, h=3.0)
                ._set_initial_state(x=-room_size/3, y=room_size/3, theta=0.0)
                ._add_room_walls(width=room_size, height=room_size)
                ._set_bounds((-room_size/2, room_size/2), (-room_size/2, room_size/2))
                .build())

    @staticmethod
    def level_2_intermediate_wall() -> Scenario:
        """2. Intermediate Wall: The car must go around a wall to reach the slot."""
        builder = ScenarioBuilder()
        return (builder
                ._set_parking_slot(x=5.0, y=5.0, theta=0.0, w=6.0, h=3.0)
                ._set_initial_state(x=-10.0, y=-10.0, theta=np.pi/4)
                ._add_obstacle(x=0.0, y=0.0, theta=-np.pi/4, w=12.0, h=1.0) # The blocking wall
                .build())

    @staticmethod
    def level_3_parallel(gap_size: float = 6.5, curb_y: float = -2.0) -> Scenario:
        """3. Parallel Parking: Between two parked cars."""
        car_w, car_h = 4.8, 2.0
        builder = ScenarioBuilder()
        
        return (builder
                ._set_parking_slot(x=0.0, y=0.0, theta=0.0, w=gap_size, h=car_h + 0.5)
                ._set_initial_state(x=-10.0, y=4.0, theta=0.0)
                ._add_parked_car(x=gap_size/2 + car_w/2, y=0.0, theta=0.0)   # Front car
                ._add_parked_car(x=-(gap_size/2 + car_w/2), y=0.0, theta=0.0) # Back car
                ._add_obstacle(x=0.0, y=curb_y, theta=0.0, w=30.0, h=1.0)     # Curb/Sidewalk
                ._set_bounds((-20.0, 20.0), (-5.0, 15.0))
                .build())

    @staticmethod
    def level_4_perpendicular(gap_size: float = 3.0) -> Scenario:
        """4. Perpendicular Parking: Side-by-side battery parking."""
        car_w, car_h = 4.8, 2.0
        builder = ScenarioBuilder()
        
        return (builder
                ._set_parking_slot(x=0.0, y=0.0, theta=np.pi/2, w=car_h + 0.5, h=gap_size)
                ._set_initial_state(x=-8.0, y=-8.0, theta=np.pi/2)
                ._add_parked_car(x=-(gap_size/2 + car_h/2), y=0.0, theta=np.pi/2) # Left car
                ._add_parked_car(x=(gap_size/2 + car_h/2), y=0.0, theta=np.pi/2)  # Right car
                ._add_obstacle(x=0.0, y=car_w/2 + 0.5, theta=0.0, w=20.0, h=1.0)   # Back wall
                ._set_bounds((-15.0, 15.0), (-15.0, 10.0))
                .build())
