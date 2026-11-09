import random
import statistics as st
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions import N_ACTIONS
from config import (
    N_INPUTS, HIDDEN_SIZE, GAMMA, EPSILON_START, EPSILON_FLOOR,
    EPSILON_DECAY_FRACTION, REPLAY_CAPACITY, REPLAY_WARMUP, BATCH_SIZE,
    TRAIN_EVERY, TARGET_SYNC_STEPS, GRAD_CLIP, TOTAL_ENV_STEPS, REWARD_SCALE,
    HUBER_DELTA, MLP_W1_INIT_RANGE, INIT_WEIGHT_RANGE,
)
from linear_policy import encode
import env as envmod

torch.set_num_threads(1)  # many small ops -- threading overhead costs more than it saves


class QNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        # Match dqn.py's new_net() init exactly (PyTorch's default Linear init
        # uses a different range) -- isolates whether an init/optimizer
        # difference explains a fitness gap, rather than an actual bug.
        nn.init.uniform_(self.fc1.weight, -MLP_W1_INIT_RANGE, MLP_W1_INIT_RANGE)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight, -INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def encode_dense(obs):
    x = torch.zeros(N_INPUTS)
    for i in encode(obs):
        x[i] = 1.0
    return x


def evaluate_greedy(net, n_episodes=2000, seed=12345):
    e = envmod.GridEnv(envmod.return_matched(), random.Random(seed))
    rng = random.Random(seed)
    fits, foods = [], []
    net.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = e.reset(rng)
            done = False
            while not done:
                q = net(encode_dense(obs).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())
                obs, _r, done, info = e.step(action)
            fits.append(info["fitness"])
            foods.append(info["food"])
    net.train()
    n = len(fits)
    return {"fitness": sum(fits) / n, "food": sum(foods) / n}


def train(seed, total_steps=TOTAL_ENV_STEPS, lr=1e-2, verbose=False):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    e = envmod.GridEnv(envmod.return_matched(REWARD_SCALE), rng)

    net = QNet()
    target = QNet()
    target.load_state_dict(net.state_dict())
    opt = torch.optim.SGD(net.parameters(), lr=lr)  # plain SGD -- matches dqn.py's apply_grads
    buffer = []
    buf_pos = 0

    decay_steps = max(1, int(total_steps * EPSILON_DECAY_FRACTION))
    obs = e.reset(rng)

    for t in range(total_steps):
        eps = max(EPSILON_FLOOR,
                  EPSILON_START + (EPSILON_FLOOR - EPSILON_START) * t / decay_steps)
        x = encode_dense(obs)
        if rng.random() < eps:
            action = rng.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                action = int(torch.argmax(net(x.unsqueeze(0)), dim=1).item())

        nobs, reward, done, _info = e.step(action)
        nx = encode_dense(nobs)
        transition = (x, action, reward, nx, done)
        if len(buffer) < REPLAY_CAPACITY:
            buffer.append(transition)
        else:
            buffer[buf_pos] = transition
            buf_pos = (buf_pos + 1) % REPLAY_CAPACITY

        obs = e.reset(rng) if done else nobs

        if len(buffer) >= REPLAY_WARMUP and t % TRAIN_EVERY == 0:
            batch = [buffer[rng.randrange(len(buffer))] for _ in range(BATCH_SIZE)]
            xs = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            nxs = torch.stack([b[3] for b in batch])
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            with torch.no_grad():
                y = rewards + GAMMA * target(nxs).max(dim=1).values * (1.0 - dones)

            q = net(xs).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.huber_loss(q, y, delta=HUBER_DELTA)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

        if t % TARGET_SYNC_STEPS == 0:
            target.load_state_dict(net.state_dict())

        if verbose and (t + 1) % 50_000 == 0:
            ev = evaluate_greedy(net, n_episodes=100)
            print(f"  t={t+1:>7} eps={eps:.2f} greedy_fitness={ev['fitness']:.1f}")

    return net


if __name__ == "__main__":
    import sys
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    rows = []
    for s in seeds:
        t0 = time.time()
        net = train(s, verbose=True)
        ev = evaluate_greedy(net, n_episodes=2000)
        dt = time.time() - t0
        rows.append(ev)
        print(f"seed {s}: fitness={ev['fitness']:.1f}  food={ev['food']:.2f}  "
              f"time={dt:.0f}s", flush=True)

    fits = [r["fitness"] for r in rows]
    print(f"\nPyTorch reference DQN ({len(seeds)} seeds): "
          f"mean={st.mean(fits):.1f}" +
          (f" +/- {st.stdev(fits):.1f}" if len(fits) > 1 else ""))
    print("hand-rolled dqn.py (lr=1e-2, 5 seeds): mean=185.2 +/- 52.2  "
          "(data/dqn_lr2_results.json)")
