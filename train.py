import json
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

import environment

env = environment.get_training_environment()
save_path, model_path, _,_ = environment.get_paths()


# # Settings adapted from
# https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/sb3_highway_dqn.py
learning_config = {
    "policy": "MlpPolicy",
    "policy_kwargs": {
        "net_arch": [256, 256]
    },
    "learning_rate": 5e-4,
    "buffer_size": 15000,
    "learning_starts": 200, 
    "batch_size": 32,
    "gamma": 0.9, 
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update_interval": 50,
    "verbose": 1,
    "tensorboard_log": save_path,
    "training_steps": int(20_000) 
}

# # Claude recommendation for longer episodes
# learning_config = {
#     "policy": "MlpPolicy",
#     "policy_kwargs": {
#         "net_arch": [256, 256]
#     },
#     "learning_rate": 5e-4,
#     "buffer_size": 75000, # up from 15000
#     "learning_starts": 500, # up from 200
#     "batch_size": 32,
#     "gamma": 0.99, # up from .9
#     "exploration_fraction": 0.3,
#     "exploration_initial_eps": 1.0,
#     "exploration_final_eps": 0.05,
#     "train_freq": 1,
#     "gradient_steps": 1,
#     "target_update_interval": 50,
#     "verbose": 1,
#     "tensorboard_log": save_path,
#     "training_steps": int(70_000) # up from 20_000 to fill the larger buffer and train longer
# }

model = DQN(
    learning_config["policy"],
    env,
    policy_kwargs=learning_config["policy_kwargs"],
    learning_rate=learning_config["learning_rate"],
    buffer_size=learning_config["buffer_size"],
    learning_starts=learning_config["learning_starts"],
    batch_size=learning_config["batch_size"],
    gamma=learning_config["gamma"],
    exploration_fraction=learning_config["exploration_fraction"],
    exploration_initial_eps=learning_config["exploration_initial_eps"],
    exploration_final_eps=learning_config["exploration_final_eps"],
    train_freq=learning_config["train_freq"],
    gradient_steps=learning_config["gradient_steps"],
    target_update_interval=learning_config["target_update_interval"],
    verbose=learning_config["verbose"],
    tensorboard_log=learning_config["tensorboard_log"]
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,    
    save_path=save_path,
    name_prefix="checkpoint"
)

model.learn(learning_config["training_steps"],   
    callback=checkpoint_callback,
    tb_log_name="tb_log",
    progress_bar=True)

os.makedirs(save_path, exist_ok=True)

config_path = os.path.join(save_path, "env_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(env.config, f, indent=2)

learning_config_path = os.path.join(save_path, "learning_config.json")
with open(learning_config_path, "w", encoding="utf-8") as f:
    json.dump(learning_config, f, indent=2)

model.save(model_path)
