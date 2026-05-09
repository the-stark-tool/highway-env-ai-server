from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
from datetime import datetime

import environment

RECORD_VIDEO = False

timestamp = datetime.now().strftime("%m%d_%H%M")
save_path, model_path, video_path, observation_path = environment.get_paths()
env = environment.get_testing_environment(render_mode="rgb_array", timestamp=timestamp, observation_dir=observation_path)
model = DQN.load(model_path)

action_counter = [0]*5 
crashes = 0
test_runs = 10

if RECORD_VIDEO:
    env = RecordVideo(
            env, video_folder=video_path, episode_trigger=lambda e: True,
            name_prefix=timestamp
        )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

for _ in range(test_runs):
    state = env.reset()[0]
    done = False
    truncated = False
    while not done and not truncated:
        action = model.predict(state, deterministic=True)[0]
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        env.render()
        # print(state)
        action_counter[action] += 1
        print('\r', action_counter, end='')  # Verify multiple actions are taken

        if info and info['crashed']:
            crashes += 1

print("\rCrashes:", crashes, "/", test_runs, "runs", f"({crashes/test_runs*100:0.1f} %)")
env.close()