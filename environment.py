import gymnasium as gym

VEHICLE_COUNT = 5

ACTIONS_ALL = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}


ENV_CONFIG = {
    "lanes_count": 3, 
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "reward_speed_range": [0,40],
    "high_speed_reward": 1,
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0,5,10,15,20,25,30,35,40]},
}

BASE_PATH = ""

def get_envirnonment(render_mode):
    env = gym.make("highway-fast-v0", config=ENV_CONFIG, render_mode=render_mode)
    env.reset()
    return env