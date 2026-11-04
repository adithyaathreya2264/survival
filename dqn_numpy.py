import numpy as np

from actions import N_ACTIONS
from config import N_INPUTS, HIDDEN_SIZE, MLP_W1_INIT_RANGE, INIT_WEIGHT_RANGE, HUBER_DELTA
from linear_policy import encode


def new_net(rng):
    return {
        "W1": rng.uniform(-MLP_W1_INIT_RANGE, MLP_W1_INIT_RANGE, size=(N_INPUTS, HIDDEN_SIZE)),
        "b1": np.zeros(HIDDEN_SIZE),
        "W2": rng.uniform(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE, size=(HIDDEN_SIZE, N_ACTIONS)),
        "b2": np.zeros(N_ACTIONS),
    }


def clone(net):
    return {k: v.copy() for k, v in net.items()}


def encode_dense(obs):
    x = np.zeros(N_INPUTS)
    x[encode(obs)] = 1.0
    return x


def forward_batch(net, X):
    h = X @ net["W1"] + net["b1"]
    a = np.tanh(h)
    q = a @ net["W2"] + net["b2"]
    return a, q


def huber_batch(e, delta=HUBER_DELTA):
    abs_e = np.abs(e)
    quadratic = 0.5 * e * e
    linear = delta * (abs_e - 0.5 * delta)
    loss = np.where(abs_e <= delta, quadratic, linear)
    grad = np.where(abs_e <= delta, e, delta * np.sign(e))
    return loss, grad


def compute_batch_grads(net, X, actions, targets):
    B = X.shape[0]
    a, q = forward_batch(net, X)
    q_taken = q[np.arange(B), actions]
    loss, dq_taken = huber_batch(q_taken - targets)

    dQ = np.zeros_like(q)
    dQ[np.arange(B), actions] = dq_taken

    gW2 = a.T @ dQ
    gb2 = dQ.sum(axis=0)
    dA = dQ @ net["W2"].T
    dH = dA * (1.0 - a * a)
    gW1 = X.T @ dH
    gb1 = dH.sum(axis=0)

    grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
    return grads, loss.mean()


def apply_grads(net, grads, lr, scale=1.0, clip=None):
    if clip is not None:
        total = sum(float(np.sum(g * g)) for g in grads.values())
        norm = total ** 0.5 * scale
        if norm > clip:
            scale *= clip / norm
    step = lr * scale
    for k in net:
        net[k] -= step * grads[k]


def greedy_action(net, x):
    _a, q = forward_batch(net, x.reshape(1, -1))
    return int(np.argmax(q[0]))


def to_genome(net):
    # dqn.py's layout: g[i*HIDDEN_SIZE + j] = W1[i][j], g[...+j*N_ACTIONS+k] = W2[j][k]
    # -- both row-major over the same (rows, cols) shape numpy already uses.
    parts = [net["W1"].reshape(-1), net["b1"], net["W2"].reshape(-1), net["b2"]]
    return np.concatenate(parts).tolist()
