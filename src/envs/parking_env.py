import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List

from .models import CarState
from .collisions import Rectangle
from .scenario import Scenario
from .sensors import Sensor
from .rewards import RewardFunction
from .kinematics import BikeKinematics


class RLParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 scenario: Scenario, 
                 dynamics: BikeKinematics, 
                 sensors: List[Sensor], 
                 reward_fn: RewardFunction):
        super().__init__()
        self.scenario = scenario
        self.dynamics = dynamics
        self.sensors = sensors
        self.reward_fn = reward_fn
        
        # Actions: [acceleration, steering_angle]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -np.pi/4]), 
            high=np.array([5.0, np.pi/4]), 
            dtype=np.float32
        )
        
        sensor_obs_size = sum(s.observation_space.shape[0] for s in self.sensors)
        # Internal state observations: v, theta, relative x, relative y
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(sensor_obs_size + 4,), 
            dtype=np.float32
        )
        
        self.current_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Deepish copy of initial state
        self.current_state = CarState(
            x=self.scenario.initial_car_state.x,
            y=self.scenario.initial_car_state.y,
            theta=self.scenario.initial_car_state.theta,
            v=self.scenario.initial_car_state.v
        )
        return self._get_obs(), {}

    def step(self, action):
        old_state = self.current_state
        
        # 1. Update Kinematics
        a, phi = action[0], action[1]
        self.current_state = self.dynamics.update(old_state, a, phi)
        
        # 2. Collision Check
        car_rect = Rectangle(self.current_state.x, self.current_state.y, self.current_state.theta, w=4.0, h=2.0)
        is_crash = any(car_rect.is_collision(obs) for obs in self.scenario.obstacles)
        
        # 3. Compute Rewards
        reward, terminated, truncated = self.reward_fn.compute(
            old_state, self.current_state, action, self.scenario, is_crash
        )
        
        # 4. Check boundaries to truncate episode if the agent runs away
        if not terminated and self.scenario.is_out_of_bounds(car_rect, x_bounds=(-30, 30), y_bounds=(-30, 30)):
            truncated = True
            reward -= 50.0 
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        sensor_data = []
        for s in self.sensors:
            sensor_data.append(s.observe(self.current_state, self.scenario))
        
        sensor_arr = np.concatenate(sensor_data) if sensor_data else np.array([])
            
        rel_x = self.scenario.parking_slot.x - self.current_state.x
        rel_y = self.scenario.parking_slot.y - self.current_state.y
        state_arr = np.array([self.current_state.v, self.current_state.theta, rel_x, rel_y], dtype=np.float32)
        
        return np.concatenate([sensor_arr, state_arr])
