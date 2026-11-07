from math import tanh

from config import (
    HIDDEN_SIZE,
    INIT_WEIGHT_RANGE,
    MLP_GENOME_LENGTH,
    MLP_W1_INIT_RANGE,
    N_INPUTS,
)
from actions import N_ACTIONS
from linear_policy import encode  # identical observation encoding -- do not fork

W1_SIZE = N_INPUTS * HIDDEN_SIZE
B1_OFFSET = W1_SIZE
W2_OFFSET = B1_OFFSET + HIDDEN_SIZE
B2_OFFSET = W2_OFFSET + HIDDEN_SIZE * N_ACTIONS

assert B2_OFFSET + N_ACTIONS == MLP_GENOME_LENGTH  # == 1684 at the default VIEW_RADIUS=2


def random_genome(rng):
    genome = [
        rng.uniform(-MLP_W1_INIT_RANGE, MLP_W1_INIT_RANGE) for _ in range(W1_SIZE)
    ]
    genome += [
        rng.uniform(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE)
        for _ in range(MLP_GENOME_LENGTH - W1_SIZE)
    ]
    assert len(genome) == MLP_GENOME_LENGTH, len(genome)
    return genome


def forward(genome, obs, activation=tanh):
    hidden = [genome[B1_OFFSET + j] for j in range(HIDDEN_SIZE)]
    for i in encode(obs):
        base = i * HIDDEN_SIZE
        for j in range(HIDDEN_SIZE):
            hidden[j] += genome[base + j]
    hidden = [activation(h) for h in hidden]

    scores = [genome[B2_OFFSET + a] for a in range(N_ACTIONS)]
    for j in range(HIDDEN_SIZE):
        h = hidden[j]
        base = W2_OFFSET + j * N_ACTIONS
        for a in range(N_ACTIONS):
            scores[a] += h * genome[base + a]
    return hidden, scores


def make_policy(genome, activation=tanh):

    def policy(obs):
        _hidden, scores = forward(genome, obs, activation)
        best = 0
        for a in range(1, N_ACTIONS):
            if scores[a] > scores[best]:
                best = a
        return best

    return policy
