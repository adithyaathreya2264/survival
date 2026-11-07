from config import (
    GENOME_LENGTH,
    INIT_WEIGHT_RANGE,
    N_CHANNELS,
    N_INPUTS,
    P_SELF,
    P_WALL,
)
from actions import N_ACTIONS

# raw perception value -> one-hot channel index; P_SELF has no channel (constant,
# carries no information, so the center cell contributes nothing but bias).
_CHANNEL = {0: 0, 1: 1, -1: 2, P_WALL: 3}

_BIAS_OFFSET = N_INPUTS * N_ACTIONS


def encode(obs):
    active = []
    for cell, value in enumerate(obs):
        if value == P_SELF:
            continue
        active.append(cell * N_CHANNELS + _CHANNEL[value])
    return active


def random_genome(rng):
    return [
        rng.uniform(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE)
        for _ in range(GENOME_LENGTH)
    ]


def make_policy(genome):

    def policy(obs):
        scores = [genome[_BIAS_OFFSET + a] for a in range(N_ACTIONS)]
        for i in encode(obs):
            base = i * N_ACTIONS
            for a in range(N_ACTIONS):
                scores[a] += genome[base + a]
        best = 0
        for a in range(1, N_ACTIONS):
            if scores[a] > scores[best]:
                best = a
        return best

    return policy
