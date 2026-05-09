import sys
import os
from stable_baselines3 import DQN
from flask import Flask, request, jsonify
import pickle
import environment
from highway_env import utils
import pandas as pd
import numpy as np
import logging
import noise_and_filter

# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

app = Flask(__name__)


@app.route('/reset', methods=['GET'])
def reset():
    global env, state
    sample_size = request.args.get('sample_size', default=10, type=int)
    next_state = env.reset()[0]
    state = next_state
    noisy_states = noise_and_filter.perturb_state(state, 
                                                  n_samples=sample_size, 
                                                  ranges=env.observation_type.features_range)
    return state_to_json(env, noisy_states, done=False, truncated=False, crashes=0), 200

@app.route('/step', methods=['GET'])
def step():
    global env, state
    sample_size = request.args.get('sample_size', default=10, type=int)
    action = model.predict(observation=state, deterministic=True)[0]
    next_state, _, done, truncated, info = env.step(action)
    state = next_state
    # print(state)
    crashes = 0
    if info and info['crashed']:
        # print(info)
        # print("sample crashed")
        crashes = 1
    noisy_states = noise_and_filter.perturb_state(state, n_samples=sample_size, ranges=env.observation_type.features_range)
    return state_to_json(env, noisy_states, done, truncated, crashes), 200

def state_to_json(env, noisy_states, done, truncated, crashes):
    result = jsonify({"states": 
                    [{
                        "env":  "",
                        "features": env.observation_type.features,
                        "state": s,
                        "crashes": crashes,
                        "done": done,
                        "truncated": truncated,
                        "features_range": env.observation_type.features_range,
                    } for s in noisy_states]})
    # print(result.get_json())
    return result

# def denormalize_observation(s):
#     global env
#     observation = env.observation_type
#     df = pd.DataFrame(s, columns=observation.features)
#     for feature, f_range in observation.features_range.items():
#         if feature in df:
#             df[feature] = utils.lmap(df[feature], [-1, 1], [f_range[0], f_range[1]])
#     return df.values.tolist()


# def renormalize_observation(s):
#     global env
#     observation = env.observation_type
#     df = pd.DataFrame.from_records(s, columns=observation.features)
#     for feature, f_range in observation.features_range.items():
#         if feature in df:
#             df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
#     return df.to_numpy()


def display_help():
    print("Usage: python3 server.py [model_id]")
    print("model_id: The name of the model to save/load (default: 'new')")

def get_paths():
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = 'new'

    save_path = os.path.join(environment.BASE_PATH, model_id)
    model_path = os.path.join(save_path, "trained_model")

    return model_path

if __name__ == '__main__':
    if len(sys.argv) < 1:
        display_help()
        sys.exit(1)
    env = environment.get_envirnonment(render_mode=None)
    MODEL_PATH = get_paths()
    model = DQN.load(MODEL_PATH)

    app.run(port=6000, threaded=True)  # Server runs on http://127.0.0.1:6000