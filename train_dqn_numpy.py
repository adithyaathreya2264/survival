import random

import numpy as np

from config import (
    GAMMA, EPSILON_START, EPSILON_FLOOR, EPSILON_DECAY_FRACTION,
    REPLAY_CAPACITY, REPLAY_WARMUP, BATCH_SIZE, TRAIN_EVERY, TARGET_SYNC_STEPS,
    GRAD_CLIP, TOTAL_ENV_STEPS, REWARD_SCALE,
)
from actions import N_ACTIONS
import env as envmod
import dqn_numpy as dqnp


def train(seed=1, total_steps=TOTAL_ENV_STEPS, lr=1e-2, verbose=False, log_every=25_000,
          grad_clip=GRAD_CLIP, target_sync_steps=TARGET_SYNC_STEPS):
    py_rng = random.Random(seed)      # environment / epsilon-greedy randomness
    np_rng = np.random.default_rng(seed)  # net init only

    e = envmod.GridEnv(envmod.return_matched(REWARD_SCALE), py_rng)
    net = dqnp.new_net(np_rng)
    target_net = dqnp.clone(net)

    buffer = []  # list of (x, action, reward, next_x, done); x are dense numpy vectors
    buf_pos = 0

    decay_steps = max(1, int(total_steps * EPSILON_DECAY_FRACTION))
    obs = e.reset(py_rng)
    x = dqnp.encode_dense(obs)

    for t in range(total_steps):
        eps = max(EPSILON_FLOOR,
                  EPSILON_START + (EPSILON_FLOOR - EPSILON_START) * t / decay_steps)
        if py_rng.random() < eps:
            action = py_rng.randrange(N_ACTIONS)
        else:
            action = dqnp.greedy_action(net, x)

        nobs, reward, done, info = e.step(action)
        nx = dqnp.encode_dense(nobs)
        transition = (x, action, reward, nx, done)
        if len(buffer) < REPLAY_CAPACITY:
            buffer.append(transition)
        else:
            buffer[buf_pos] = transition
            buf_pos = (buf_pos + 1) % REPLAY_CAPACITY

        if done:
            obs = e.reset(py_rng)
            x = dqnp.encode_dense(obs)
        else:
            x = nx

        if len(buffer) >= REPLAY_WARMUP and t % TRAIN_EVERY == 0:
            idx = [py_rng.randrange(len(buffer)) for _ in range(BATCH_SIZE)]
            batch = [buffer[i] for i in idx]
            X = np.stack([b[0] for b in batch])
            actions = np.array([b[1] for b in batch])
            rewards = np.array([b[2] for b in batch])
            NX = np.stack([b[3] for b in batch])
            dones = np.array([b[4] for b in batch], dtype=bool)

            _a, q_next = dqnp.forward_batch(target_net, NX)
            targets = rewards + (~dones) * GAMMA * q_next.max(axis=1)

            grads, _loss = dqnp.compute_batch_grads(net, X, actions, targets)
            dqnp.apply_grads(net, grads, lr, scale=1.0, clip=grad_clip)

        if t % target_sync_steps == 0:
            target_net = dqnp.clone(net)

        if verbose and (t + 1) % log_every == 0:
            import architectures
            genome = dqnp.to_genome(net)
            ev_policy = architectures.get("mlp").make_policy(genome)
            fits = []
            rng2 = random.Random(12345)
            for _ in range(50):
                o = e.reset(rng2)
                d = False
                while not d:
                    a = ev_policy(o)
                    o, _r, d, i = e.step(a)
                fits.append(i["fitness"])
            print(f"  t={t+1:>7} eps={eps:.2f} greedy_fitness={sum(fits)/len(fits):7.1f}",
                  flush=True)

    return net


if __name__ == "__main__":
    import sys
    import json
    import time
    import statistics as st
    import architectures
    from analyze_champion import profile

    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    rows = []
    for s in seeds:
        t0 = time.time()
        net = train(s, verbose=True)
        genome = dqnp.to_genome(net)
        p = profile(architectures.get("mlp").make_policy(genome), n=2000)
        p["seed"] = s
        rows.append(p)
        print(f"seed {s}: fitness={p['mean_fitness']:.1f}  food={p['mean_food']:.2f}  "
              f"time={time.time()-t0:.0f}s", flush=True)

    fits = [r["mean_fitness"] for r in rows]
    json.dump(rows, open("data/dqn_numpy_results.json", "w"), indent=2)
    print(f"\nnumpy-batched DQN, {len(seeds)} seeds: "
          f"mean={st.mean(fits):.1f}" + (f" +/- {st.stdev(fits):.1f}" if len(fits) > 1 else ""))
    print("pure-Python DQN (lr=1e-2, 5 seeds): mean=185.2 +/- 52.2  (data/dqn_lr2_results.json)")
