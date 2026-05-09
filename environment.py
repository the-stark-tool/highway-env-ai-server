import os
import sys
import gymnasium as gym
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

LANES = 4

ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}


# Longer episodes
ENV_CONFIG = {
    "lanes_count": LANES, 
    "collision_reward": -1,
    "duration": 100,
    "right_lane_reward": 0.1,
    "reward_speed_range": [0,40],
    "high_speed_reward": 1,
    "simulation_frequency" : 15,
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0,5,10,15,20,25,30,35,40]},
    "observation": {
        "type": "Kinematics",
        # Defaults highway_env/envs/common/observation.py:191
        # overriden with new range for x that allows values over 200
        "features_range" : {
                "x": [-1000, 1000],
                "y": [-AbstractLane.DEFAULT_WIDTH * LANES, AbstractLane.DEFAULT_WIDTH * LANES], 
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            },
    }
}

# abz: normalized and relative
# ENV_CONFIG = {
#     "lanes_count": 3, 
#     "collision_reward": -1,
#     "duration": 30,
#     "right_lane_reward": 0.1,
#     "reward_speed_range": [0,40],
#     "high_speed_reward": 1,
#     "action": {
#         "type": "DiscreteMetaAction",
#         "target_speeds": [0,5,10,15,20,25,30,35,40]},
#     "observation": {
#         "type": "Kinematics",
#     }
# }


BASE_PATH = ""
TRAINING_ENVIRONMENT = "highway-v0"
TESTING_ENVIRONMENT = "observable-highway-env-v0"

def get_training_environment():
    return gym.make(TRAINING_ENVIRONMENT, 
                   config=ENV_CONFIG, render_mode=None)

def get_testing_environment(render_mode=None, timestamp=None, observation_dir="observations"):
    return gym.make(TESTING_ENVIRONMENT, 
                   config=ENV_CONFIG, render_mode=render_mode,
                   observation_dir=observation_dir, timestamp=timestamp)


def get_paths():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = 'new'

    save_path = os.path.join(BASE_PATH, model_id)
    model_path = os.path.join(save_path, "trained_model")
    video_path = os.path.join(save_path, "videos")
    observation_path = os.path.join(save_path, "observations")

    return save_path, model_path, video_path, observation_path