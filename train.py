import os
import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import time

import environment

# Configuration

def display_script_help():
    print("Usage: python3 train.py train [model_id]")
    print("       python3 train.py test [model_id]")
    print()
    print("model_id: The name of the model to save/load (default: 'new')")

def get_paths():
    if len(sys.argv) > 2:
        model_id = sys.argv[2]
    else:
        model_id = 'new'

    save_path = os.path.join(environment.BASE_PATH, model_id)
    model_path = os.path.join(save_path, "trained_model")

    return save_path, model_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        display_script_help()
        sys.exit(1)

    if sys.argv[1] == 'train':
        env = environment.get_envirnonment(render_mode=None)
        save_path, model_path = get_paths()

        # Settings adapted from
        # https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/sb3_highway_dqn.py
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.9,  # Discount factor
                    exploration_fraction=0.3,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    #tensorboard_log=save_path
                    )
        

        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=save_path,
            name_prefix="rl_model"
        )

        model.learn(int(20_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
        model.save(model_path)


    elif sys.argv[1] == 'test':
        env = environment.get_envirnonment(render_mode='human')
        save_path, model_path = get_paths()
        env.configure({"simulation_frequency": 15})
        model = DQN.load(model_path)

        action_counter = [0]*5 
        crashes = 0
        test_runs = 10

        for _ in range(test_runs):
            state = env.reset()[0]
            done = False
            truncated = False
            while not done and not truncated:
                action = model.predict(state, deterministic=True)[0]
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                env.render()
                #print("state",state)
                action_counter[action] += 1
                print('\r', action_counter, end='')  # Verify multiple actions are taken

                if info and info['crashed']:
                    crashes += 1

        print("\rCrashes:", crashes, "/", test_runs, "runs", f"({crashes/test_runs*100:0.1f} %)")
        env.close()
    else:
        display_script_help()

    env.close()