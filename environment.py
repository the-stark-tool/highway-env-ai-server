import gymnasium as gym
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

LANES = 3

ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}


ENV_CONFIG = {
    "lanes_count": LANES, 
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "reward_speed_range": [0,40],
    "high_speed_reward": 1,
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0,5,10,15,20,25,30,35,40]},
    "observation": {
        "type": "Kinematics",
        # Defaults highway_env/envs/common/observation.py:191
        # overriden with new range for x that allows values over 200
        "features_range" : { 
            "x": [0, 1000],
            "y": [-AbstractLane.DEFAULT_WIDTH * LANES, AbstractLane.DEFAULT_WIDTH * LANES], 
            "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
            "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
        }
    }
}

BASE_PATH = ""

def get_envirnonment(render_mode):
    env = gym.make("highway-fast-v0", config=ENV_CONFIG, render_mode=render_mode)
    env.reset()
    return env