# RL Parking Environment (RLParkingEnv)

This document provides a detailed specification of the `RLParkingEnv`, a modular, Gymnasium-compatible Reinforcement Learning environment designed for autonomous parking tasks.

---

## 1. Overview

The `RLParkingEnv` is built using **SOLID** principles to ensure modularity. Instead of hardcoding physics, sensors, and rewards into a single monolithic class, the environment acts as an orchestrator for injected dependencies:
- **Scenario**: Defines the map, obstacles, and parking target.
- **DynamicsModel**: Handles the physics step (e.g., Kinematic Bicycle Model).
- **Sensors**: Generates observations (e.g., Lidar, Camera, Odometry).
- **RewardFunction**: Calculates the step reward.

---

## 2. Action Space

The action space is a continuous `gymnasium.spaces.Box` representing the control commands sent to the vehicle. 

By default, the environment expects a 2D continuous action space:
* **Shape:** `(2,)`
* **Data Type:** `np.float32`

| Index | Action | Min | Max | Unit |
|-------|--------|-----|-----|------|
| 0 | Acceleration ($a$) | `-MAX_ACCEL` | `MAX_ACCEL` | $m/s^2$ |
| 1 | Steering Angle ($\phi$) | `-MAX_STEER` | `MAX_STEER` | radians |

*Note: Bounds can be configured during the environment instantiation via the `DynamicsModel` wrapper.*

---

## 3. Observation Space

The observation space is a continuous `gymnasium.spaces.Box`. Because the environment relies on a modular `Sensor` interface, the exact shape depends on the sensors injected during initialization.

### Default Configuration (Lidar + Odometry)
Assuming a 12-point Lidar and a standard internal state sensor:
* **Shape:** `(16,)` 
* **Data Type:** `np.float32`

| Index | Observation | Min | Max |
|-------|-------------|-----|-----|
| `0-11` | Normalized Lidar Ray Distances | `-1.0` | `1.0` |
| `12` | Vehicle Velocity ($v$) | `-v_max` | `v_max` |
| `13` | Vehicle Heading ($\theta$) | $-\pi$ | $\pi$ |
| `14` | Relative Target X | $-\infty$ | $\infty$ |
| `15` | Relative Target Y | $-\infty$ | $\infty$ |

*(The relative target X/Y is often preferred over absolute global coordinates to ensure the policy generalizes across different parking spot locations).*

---

## 4. Reward Function

Rewards are handled by the `RewardFunction` interface. You can swap between sparse and dense rewards depending on your RL algorithm (e.g., PPO usually prefers dense, while SAC can handle sparse with HER).

### Default: `DenseParkingReward`
Provides a continuous shaping reward to guide the agent toward the slot.
- **Collision Penalty:** $-100.0$ (and terminates the episode).
- **Distance Reward:** Penalizes the Euclidean distance between the car's center and the parking slot's center.
- **Alignment Reward:** Penalizes the angular difference between the car's heading and the parking slot's orientation.
- **Success Bonus:** $+100.0$ if the car's `proportion_in` the parking slot exceeds a threshold (e.g., $0.95$) and velocity is near zero.

---

## 5. Episode Termination & Truncation

The `step()` function returns both `terminated` and `truncated` flags according to the standard Gymnasium API.

**Terminated (Agent reached a terminal state):**
1. **Collision:** The vehicle's bounding box intersects with any `Rectangle` in the `Scenario.obstacles`.
2. **Success:** The vehicle is successfully parked inside the `parking_slot` with $v \approx 0$.

**Truncated (Episode artificially ended):**
1. **Time Limit:** The environment exceeds the maximum number of allowable steps (e.g., 500 steps).
2. **Out of Bounds:** The vehicle drives outside the permitted scenario grid limits.

---

## 6. Architecture & Extensibility

To extend this environment, developers should implement the respective interfaces rather than modifying `RLParkingEnv` directly.

### Adding a New Sensor
Create a class inheriting from `Sensor`.
```python
class CameraSensor(Sensor):
    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def observe(self, car_state: CarState, scenario: Scenario) -> np.ndarray:
        # Render top-down view and return RGB array
        return rgb_array

```

### Implementing Curriculum Learning

Because the map layout is isolated in the `Scenario` dataclass, you can implement Curriculum Learning by passing progressively harder `Scenario` objects to `env.reset()`.

* **Level 1:** Empty lot, massive parking spot.
* **Level 2:** One adjacent obstacle vehicle.
* **Level 3:** Parallel parking between two vehicles with a narrow gap.

---

## 7. Example Usage

```python
import gymnasium as gym
import numpy as np

# 1. Initialize modular components
scenario = Scenario(obstacles=[...], parking_slot=..., initial_car_state=...)
dynamics = BikeKinematics(length=2.5, delta_t=0.1)
sensors = [LidarSensor(n_points=12), OdometrySensor()]
reward = DenseParkingReward()

# 2. Build the environment
env = RLParkingEnv(scenario, dynamics, sensors, reward)

# 3. Standard RL Loop
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample() # Replace with your RL model's action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

```
