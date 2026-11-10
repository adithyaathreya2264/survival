import math
import random
import statistics as st
import time

from actions import N_ACTIONS
from config import N_INPUTS, HIDDEN_SIZE, REWARD_SCALE
import env as envmod
from linear_policy import encode

BASELINE_DECAY = 0.99  # exponential moving average of episode returns


def new_net(rng):
    w1_range = 1.0 / math.sqrt(N_INPUTS)
    w2_range = 1.0 / math.sqrt(HIDDEN_SIZE)
    return {
        "W1": [[rng.uniform(-w1_range, w1_range) for _ in range(HIDDEN_SIZE)]
               for _ in range(N_INPUTS)],
        "b1": [0.0] * HIDDEN_SIZE,
        "W2": [[rng.uniform(-w2_range, w2_range) for _ in range(N_ACTIONS)]
               for _ in range(HIDDEN_SIZE)],
        "b2": [0.0] * N_ACTIONS,
    }


def forward(net, active):
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
    h = b1[:]
    for i in active:
        row = W1[i]
        for j in range(HIDDEN_SIZE):
            h[j] += row[j]
    a = [math.tanh(x) for x in h]
    z = b2[:]
    for j in range(HIDDEN_SIZE):
        row, av = W2[j], a[j]
        for k in range(N_ACTIONS):
            z[k] += av * row[k]
    return a, z


def softmax(z):
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]


def sample_action(probs, rng):
    r, c = rng.random(), 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1


def greedy_action(net, active):
    _a, z = forward(net, active)
    return max(range(N_ACTIONS), key=lambda k: z[k])


def zero_grads():
    return {"W1": {}, "b1": [0.0] * HIDDEN_SIZE,
            "W2": [[0.0] * N_ACTIONS for _ in range(HIDDEN_SIZE)], "b2": [0.0] * N_ACTIONS}


def accumulate(net, grads, active, action, advantage):
    a, z = forward(net, active)
    probs = softmax(z)
    dz = [probs[k] * advantage for k in range(N_ACTIONS)]
    dz[action] -= advantage

    W2 = net["W2"]
    gW2, gb2, gb1 = grads["W2"], grads["b2"], grads["b1"]
    for k in range(N_ACTIONS):
        gb2[k] += dz[k]

    dh = [0.0] * HIDDEN_SIZE
    for j in range(HIDDEN_SIZE):
        row, av = W2[j], a[j]
        for k in range(N_ACTIONS):
            gW2[j][k] += av * dz[k]
        da = sum(row[k] * dz[k] for k in range(N_ACTIONS))
        dh[j] = da * (1.0 - av * av)
        gb1[j] += dh[j]

    gW1 = grads["W1"]
    for i in active:
        row = gW1.get(i)
        if row is None:
            row = gW1[i] = [0.0] * HIDDEN_SIZE
        for j in range(HIDDEN_SIZE):
            row[j] += dh[j]


def apply_grads(net, grads, lr, scale=1.0):
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


def run_episode(net, rng, greedy=False):
    e = envmod.GridEnv(envmod.return_matched(REWARD_SCALE), rng)
    obs = e.reset(rng)
    trace = []
    done = False
    while not done:
        active = encode(obs)
        if greedy:
            action = greedy_action(net, active)
        else:
            _a, z = forward(net, active)
            probs = softmax(z)
            action = sample_action(probs, rng)
        obs, _r, done, info = e.step(action)
        trace.append((active, action))
    return trace, info["fitness"]


def evaluate_greedy(net, n_episodes=200, seed=12345):
    rng = random.Random(seed)
    fits, foods = [], []
    for _ in range(n_episodes):
        _trace, fitness = run_episode(net, rng, greedy=True)
        fits.append(fitness)
    return {"fitness": sum(fits) / len(fits)}


def train(seed=1, n_episodes=20_000, lr=1e-2, log_every=2_000, verbose=True):
    rng = random.Random(seed)
    net = new_net(rng)
    baseline = 0.0
    baseline_init = False
    history = []

    for ep in range(n_episodes):
        trace, fitness = run_episode(net, rng, greedy=False)
        scaled_return = fitness * REWARD_SCALE  # same scale as dqn.py's Q targets

        if not baseline_init:
            baseline = scaled_return
            baseline_init = True
        advantage = scaled_return - baseline
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * scaled_return

        grads = zero_grads()
        for (active, action) in trace:
            accumulate(net, grads, active, action, advantage)
        apply_grads(net, grads, lr, scale=1.0 / len(trace))

        if verbose and (ep + 1) % log_every == 0:
            ev = evaluate_greedy(net, n_episodes=100)
            history.append({"episode": ep + 1, "greedy_fitness": ev["fitness"]})
            print(f"  ep={ep+1:>7} baseline={baseline/REWARD_SCALE:7.1f} "
                  f"greedy_fitness={ev['fitness']:7.1f}", flush=True)

    return net, history


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    t0 = time.time()
    net, hist = train(seed=seed, n_episodes=n_episodes)
    ev = evaluate_greedy(net, n_episodes=2000)
    print(f"\nREINFORCE seed={seed}: final greedy fitness (2000 eval episodes) "
          f"= {ev['fitness']:.1f}  time={time.time()-t0:.0f}s")
