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

# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

app = Flask(__name__)


@app.route('/reset', methods=['POST'])
def reset():
    global env
    state = env.reset()[0]
    return env_to_json(env, state, done=False, truncated=False, crashes=0), 200


@app.route('/step', methods=['POST'])
def step():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    done = data['done']
    truncated = data['truncated']
    
    state = renormalize_observation(data['state'])
    crashes = int(data['crashes'])

    if done:
        return jsonify({"error": "The run is done"}), 400
    if truncated:
        return jsonify({"error": "The run is truncated"}), 400
    
    env = pickle.loads(bytes.fromhex(data['env']))
    action = model.predict(state, deterministic=True)[0]
    new_state, _, done, truncated, info = env.step(action)
    if info and info['crashed']:
        print("sample crashed")
        crashes += 1
    return env_to_json(env, new_state, done, truncated, crashes), 200

def env_to_json(env, state, done, truncated, crashes):
    state = denormalize_observation(state)
    return jsonify({
        "env":  pickle.dumps(env).hex(),
        "features": env.observation_type.features,
        "state": state,
        "crashes": crashes,
        "done": done,
        "truncated": truncated,
        "features_range": env.observation_type.features_range,
    })

def denormalize_observation(state):
    global env
    observation = env.observation_type
    df = pd.DataFrame(state, columns=observation.features)
    for feature, f_range in observation.features_range.items():
        if feature in df:
            df[feature] = utils.lmap(df[feature], [-1, 1], [f_range[0], f_range[1]])
    return df.values.tolist()


def renormalize_observation(state):
    global env
    observation = env.observation_type
    df = pd.DataFrame.from_records(state, columns=observation.features)
    for feature, f_range in observation.features_range.items():
        if feature in df:
            df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
    return df.to_numpy()


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