# "features":["presence","x","y","vx","vy"],
# "features_range":{"vx":[-80.0,80.0],"vy":[-80.0,80.0],"x":[-200.0,200.0],"y":[-12,12]},
# "state":[[1.0,155.3583984375,8.0,25.0,0.0],[1.0,22.299240112304688,-4.000000476837158,-2.2926864624023438,0.0],[1.0,44.90974426269531,-4.000000476837158,-3.9746322631835938,0.0],[1.0,71.29849243164062,-4.000000476837158,-1.6919403076171875,0.0],[1.0,97.7352294921875,0.0,-1.5091171264648438,0.0]],

import numpy as np
from typing import Optional
import copy

# Column indices
PRESENCE = 0
X = 1
Y = 2
VX = 3
VY = 4


def perturb_state(
    state: list[list],
    ranges: dict[str, tuple[int, int]],
    n_samples: int = 10,
    presence_flip_prob: float = 0.05,
    pos_noise_std: float = 0.0,
    vel_noise_std: float = 1.0,
    seed: Optional[int] = None,
) -> list[list[list]]:
    """
    Generate perturbed samples of the input sensor state.

    Args:
        state:              5x5 matrix [presence, x, y, vx, vy] per row (car).
        ranges:             Value ranges in state
        n_samples:          Number of perturbed states to return.
        presence_flip_prob: Probability of flipping a presence bit (false positive/negative).
        pos_noise_std:      Std dev of Gaussian noise added to x, y positions.
        vel_noise_std:      Std dev of Gaussian noise added to vx, vy velocities.
        seed:               Optional random seed for reproducibility.

    Returns:
        List of n_samples perturbed states, each a 5x5 list.
    """
    rng = np.random.default_rng(seed)
    base = np.array(state, dtype=float)   # shape (5, 5)
    samples = []

    for _ in range(n_samples):
        perturbed = copy.deepcopy(base)

        for row in range(perturbed.shape[0]):
            # --- Presence (binary) ---
            if rng.random() < presence_flip_prob:
                perturbed[row, PRESENCE] = 1.0 - perturbed[row, PRESENCE]

            # Only perturb continuous fields when a car is present
            if perturbed[row, PRESENCE] == 1.0:
                # --- Position noise ---
                perturbed[row, X]  += rng.normal(0, pos_noise_std)
                perturbed[row, Y]  += rng.normal(0, pos_noise_std)

                # --- Velocity noise ---
                perturbed[row, VX] += rng.normal(0, vel_noise_std)
                perturbed[row, VY] += rng.normal(0, vel_noise_std)

                # --- Clamp to valid sensor ranges ---
                perturbed[row, X]  = np.clip(perturbed[row, X],  *ranges["x"])
                perturbed[row, Y]  = np.clip(perturbed[row, Y],  *ranges["y"])
                perturbed[row, VX] = np.clip(perturbed[row, VX], *ranges["vx"])
                perturbed[row, VY] = np.clip(perturbed[row, VY], *ranges["vy"])

        samples.append(perturbed.tolist())

    return samples