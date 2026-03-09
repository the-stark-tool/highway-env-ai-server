# Multiple Lanes Highway AI server to measure robustness with STARK 

A Deep Q-Network (DQN) reinforcement learning agent trained to navigate highway traffic using [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv). Includes a Flask server for integration with **STARK**.

This setup was used for the article ["On The Road Again (Safely): Modelling and Analysis of Autonomous Driving with Stark"](https://doi.org/10.1007/978-3-031-94533-5_16) published in the [ABZ 2025](https://abz-conf.org/site/2025/) conference.

Requires Python 3.8+ and pip.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/the-stark-tool/highway-env-ai-server.git
cd highway-env-ai-server
pip install -r requirements.txt
```


## Training the Model

Train a new agent with:

```bash
python train.py train
```

To train and save under a specific model ID:

```bash
python train.py train <model_id>
```

The model will be saved to `./<model_id>/trained_model`. 

**Training configuration** (defined in `environment.py`):
- Environment: `highway-fast-v0`
- 3 lanes, 9 target speeds (0–40)
- 20,000 training steps
- Rewards: high speed (+1), right lane (+0.1), collision (−1)


## Testing the Model

Run the trained agent in a visual simulation:

```bash
python train.py test
```

To test a specific saved model:

```bash
python train.py test <model_id>
```

This runs 10 episodes and renders the environment in real time. A crash summary is printed at the end:

```
Crashes: 2 / 10 runs (20.0%)
```

## Running the Server (STARK Integration)

The Flask server exposes the trained agent as an API for use with STARK. Start the server with:

```bash
python server.py
```

To load a specific model:

```bash
python server.py <model_id>
```

> [!IMPORTANT]
> Once the server is up an running, run the AI experiment in [this STARK scenario](https://github.com/the-stark-tool/STARK/blob/main/examples/ABZ2025/src/main/java/autonomous/driving/Main.java). The results of the experiments are printed in the console and saved as a .txt file in the `STARK` base folder. You may read the experiment script [`AIMultipleLanes.java`](https://github.com/the-stark-tool/STARK/blob/main/examples/ABZ2025/src/main/java/Scenarios/AIMultipleLanes.java) for more details, and tweak the parameters `EVOLUTION_SEQUENCE_SIZE` and `PERTURBATION_SCALE` for shorter (but more imprecise) results.

---
### API Endpoints

The following are technical details, not necessary to run the experiments.

STARK communicates with the AI environment through an HTTP API. This The server runs on **`http://127.0.0.1:6000`**.

#### `POST /reset`
Resets the environment and returns the initial state.

**Response:**
```json
{
  "env": "<serialized environment (hex)>",
  "features": ["presence", "x", "y", "vx", "vy"],
  "state": [[...], ...],
  "crashes": 0,
  "done": false,
  "truncated": false,
  "features_range": { "x": [-100, 100], ... }
}
```

#### `POST /step`
Steps the environment forward using the model's predicted action.

**Request body:**
```json
{
  "env": "<serialized environment (hex)>",
  "state": [[...], ...],
  "done": false,
  "truncated": false,
  "crashes": 0
}
```

**Response:** Same format as `/reset`, updated with the new state and crash count.

> **Note:** States are automatically normalized/denormalized between the API and the model. The client (STARK) should pass the `state` and `env` values returned from each response directly into the next request. These `state` and `env` can be modified by the client, allowing the introduction of _perturbations_.