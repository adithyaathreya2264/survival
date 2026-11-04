import math
import random

from config import (
    HIDDEN_SIZE, N_INPUTS, MLP_GENOME_LENGTH, MLP_W1_INIT_RANGE,
    INIT_WEIGHT_RANGE, HUBER_DELTA,
)
from actions import N_ACTIONS
from linear_policy import encode  # identical observation encoding as Stages 1-2


def new_net(rng):
    return {
        "W1": [[rng.uniform(-MLP_W1_INIT_RANGE, MLP_W1_INIT_RANGE)
                for _ in range(HIDDEN_SIZE)] for _ in range(N_INPUTS)],
        "b1": [0.0] * HIDDEN_SIZE,
        "W2": [[rng.uniform(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE)
                for _ in range(N_ACTIONS)] for _ in range(HIDDEN_SIZE)],
        "b2": [0.0] * N_ACTIONS,
    }


def clone(net):
    return {
        "W1": [row[:] for row in net["W1"]],
        "b1": net["b1"][:],
        "W2": [row[:] for row in net["W2"]],
        "b2": net["b2"][:],
    }


def forward(net, active):
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
    h = b1[:]
    for i in active:
        row = W1[i]
        for j in range(HIDDEN_SIZE):
            h[j] += row[j]
    a = [math.tanh(x) for x in h]
    q = b2[:]
    for j in range(HIDDEN_SIZE):
        row, av = W2[j], a[j]
        for k in range(N_ACTIONS):
            q[k] += av * row[k]
    return a, q


def huber(e, delta=HUBER_DELTA):
    if abs(e) <= delta:
        return 0.5 * e * e, e
    return delta * (abs(e) - 0.5 * delta), delta * (1.0 if e > 0 else -1.0)


def zero_grads():
    return {
        "W1": {},
        "b1": [0.0] * HIDDEN_SIZE,
        "W2": [[0.0] * N_ACTIONS for _ in range(HIDDEN_SIZE)],
        "b2": [0.0] * N_ACTIONS,
    }


def td_error(net, active, action, target):
    """Raw (pre-Huber-clip) TD error -- used by prioritized_replay.py as the
    priority signal. Additive, does not change accumulate()'s behavior."""
    _a, q = forward(net, active)
    return q[action] - target


def accumulate(net, grads, active, action, target):
    a, q = forward(net, active)
    loss, dq_taken = huber(q[action] - target)

    dq = [0.0] * N_ACTIONS
    dq[action] = dq_taken

    W2 = net["W2"]
    gW2, gb2, gb1 = grads["W2"], grads["b2"], grads["b1"]
    gb2[action] += dq_taken

    dh = [0.0] * HIDDEN_SIZE
    for j in range(HIDDEN_SIZE):
        gW2[j][action] += a[j] * dq_taken
        # da[j] = sum_k W2[j][k]*dq[k]; only k == action is nonzero
        da = W2[j][action] * dq_taken
        dh[j] = da * (1.0 - a[j] * a[j])   # tanh'(x) = 1 - tanh(x)^2
        gb1[j] += dh[j]

    gW1 = grads["W1"]
    for i in active:                        # x_i == 1.0, so dW1[i][j] = dh[j]
        row = gW1.get(i)
        if row is None:
            row = gW1[i] = [0.0] * HIDDEN_SIZE
        for j in range(HIDDEN_SIZE):
            row[j] += dh[j]
    return loss


def apply_grads(net, grads, lr, scale=1.0, clip=None):
    if clip is not None:
        total = sum(v * v for v in grads["b2"])
        total += sum(v * v for row in grads["W2"] for v in row)
        total += sum(v * v for v in grads["b1"])
        total += sum(v * v for row in grads["W1"].values() for v in row)
        norm = math.sqrt(total) * scale
        if norm > clip:
            scale *= clip / norm

    step = lr * scale
    b2, W2, b1, W1 = net["b2"], net["W2"], net["b1"], net["W1"]
    for k in range(N_ACTIONS):
        b2[k] -= step * grads["b2"][k]
    for j in range(HIDDEN_SIZE):
        gr, wr = grads["W2"][j], W2[j]
        for k in range(N_ACTIONS):
            wr[k] -= step * gr[k]
        b1[j] -= step * grads["b1"][j]
    for i, gr in grads["W1"].items():
        wr = W1[i]
        for j in range(HIDDEN_SIZE):
            wr[j] -= step * gr[j]


def to_genome(net):
    g = [0.0] * MLP_GENOME_LENGTH
    for i in range(N_INPUTS):
        row = net["W1"][i]
        base = i * HIDDEN_SIZE
        for j in range(HIDDEN_SIZE):
            g[base + j] = row[j]
    off = N_INPUTS * HIDDEN_SIZE
    for j in range(HIDDEN_SIZE):
        g[off + j] = net["b1"][j]
    off += HIDDEN_SIZE
    for j in range(HIDDEN_SIZE):
        row = net["W2"][j]
        for k in range(N_ACTIONS):
            g[off + j * N_ACTIONS + k] = row[k]
    off += HIDDEN_SIZE * N_ACTIONS
    for k in range(N_ACTIONS):
        g[off + k] = net["b2"][k]
    return g


def greedy_action(net, obs):
    _a, q = forward(net, encode(obs))
    best = 0
    for k in range(1, N_ACTIONS):
        if q[k] > q[best]:
            best = k
    return best
