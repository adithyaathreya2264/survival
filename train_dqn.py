import json
import random
from collections import deque

from config import (
    GAMMA, LEARNING_RATE, EPSILON_START, EPSILON_FLOOR, EPSILON_DECAY_FRACTION,
    REPLAY_CAPACITY, REPLAY_WARMUP, BATCH_SIZE, TRAIN_EVERY, TARGET_SYNC_STEPS,
    GRAD_CLIP, TOTAL_ENV_STEPS, REWARD_SCALE, FOOD_FITNESS_BONUS,
)
from actions import N_ACTIONS
from linear_policy import encode
import env as envmod
import dqn
import paths


def evaluate_greedy(net, n_episodes=200, seed=12345):
    e = envmod.GridEnv(envmod.return_matched(), random.Random(seed))
    rng = random.Random(seed)
    fits, foods, cells, causes = [], [], [], {"hazard": 0, "starve": 0, "cap": 0}
    for _ in range(n_episodes):
        obs = e.reset(rng)
        while True:
            obs, _r, done, info = e.step(dqn.greedy_action(net, obs))
            if done:
                break
        fits.append(info["fitness"])
        foods.append(info["food"])
        cells.append(info["cells"])
        causes[info["cause"]] += 1
    n = len(fits)
    return {
        "fitness": sum(fits) / n, "food": sum(foods) / n,
        "cells": sum(cells) / n, "hazard": causes["hazard"] / n,
    }


def train(seed=1, reward_name="return_matched", total_steps=TOTAL_ENV_STEPS,
          lr=LEARNING_RATE, log_every=25_000, verbose=True, double=False, gamma=None,
          return_curves=False):
    gamma = GAMMA if gamma is None else gamma
    rng = random.Random(seed)
    reward_fn = (envmod.return_matched(REWARD_SCALE) if reward_name == "return_matched"
                 else envmod.sparse(REWARD_SCALE))
    e = envmod.GridEnv(reward_fn, rng)

    net = dqn.new_net(rng)
    target_net = dqn.clone(net)
    buffer = deque(maxlen=REPLAY_CAPACITY)

    decay_steps = max(1, int(total_steps * EPSILON_DECAY_FRACTION))
    obs = e.reset(rng)
    active = encode(obs)

    history, losses, qmags, ep_fitness = [], [], [], []
    action_counts = [0] * N_ACTIONS

    for t in range(total_steps):
        eps = max(EPSILON_FLOOR,
                  EPSILON_START + (EPSILON_FLOOR - EPSILON_START) * t / decay_steps)
        if rng.random() < eps:
            action = rng.randrange(N_ACTIONS)
        else:
            _a, q = dqn.forward(net, active)
            action = max(range(N_ACTIONS), key=lambda k: q[k])
            qmags.append(max(abs(v) for v in q))
        action_counts[action] += 1

        nobs, reward, done, info = e.step(action)
        nactive = encode(nobs)
        buffer.append((active, action, reward, nactive, done))

        if done:
            ep_fitness.append(info["fitness"])
            obs = e.reset(rng)
            active = encode(obs)
        else:
            active = nactive

        if len(buffer) >= REPLAY_WARMUP and t % TRAIN_EVERY == 0:
            batch = [buffer[rng.randrange(len(buffer))] for _ in range(BATCH_SIZE)]
            grads = dqn.zero_grads()
            total_loss = 0.0
            for (s, a, r, s2, d) in batch:
                if d:
                    y = r
                elif double:
                    # Double DQN: ONLINE net selects the action, TARGET net
                    # evaluates it. Decoupling the two removes the max operator's
                    # systematic overestimation bias.
                    _ho, qo = dqn.forward(net, s2)
                    a_star = max(range(N_ACTIONS), key=lambda k: qo[k])
                    _ht, qt = dqn.forward(target_net, s2)
                    y = r + gamma * qt[a_star]
                else:
                    _ha, q2 = dqn.forward(target_net, s2)
                    y = r + gamma * max(q2)
                total_loss += dqn.accumulate(net, grads, s, a, y)
            dqn.apply_grads(net, grads, lr, scale=1.0 / BATCH_SIZE, clip=GRAD_CLIP)
            losses.append(total_loss / BATCH_SIZE)

        if t % TARGET_SYNC_STEPS == 0:
            target_net = dqn.clone(net)

        if (t + 1) % log_every == 0:
            ev = evaluate_greedy(net, n_episodes=100)
            tot_a = sum(action_counts) or 1
            row = {
                "step": t + 1, "epsilon": eps,
                "loss": sum(losses[-500:]) / max(1, len(losses[-500:])),
                "mean_abs_q": sum(qmags[-2000:]) / max(1, len(qmags[-2000:])),
                "train_ep_fitness": sum(ep_fitness[-200:]) / max(1, len(ep_fitness[-200:])),
                "greedy_fitness": ev["fitness"], "greedy_food": ev["food"],
                "greedy_cells": ev["cells"], "greedy_hazard": ev["hazard"],
                "action_spread": max(action_counts) / tot_a,
            }
            history.append(row)
            if verbose:
                print(f"  t={row['step']:>7} eps={eps:4.2f} loss={row['loss']:7.4f} "
                      f"|Q|={row['mean_abs_q']:6.2f} greedy_fit={ev['fitness']:7.1f} "
                      f"food={ev['food']:5.2f} cells={ev['cells']:6.1f} "
                      f"haz={ev['hazard']:5.1%}")
            action_counts = [0] * N_ACTIONS

    if return_curves:
        # ep_fitness/losses accumulate across the WHOLE run, unlike `history`
        # (which only snapshots every log_every). Full-resolution training
        # curves for plotting -- not needed by the training loop itself.
        return net, history, {"ep_fitness": ep_fitness, "losses": losses}
    return net, history


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    reward = sys.argv[2] if len(sys.argv) > 2 else "return_matched"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else TOTAL_ENV_STEPS
    print(f"=== DQN seed={seed} reward={reward} steps={steps} ===")
    net, hist = train(seed=seed, reward_name=reward, total_steps=steps)
    tag = f"dqn_{reward}_s{seed}"
    with open(paths.champion(tag), "w") as f:
        json.dump({"arch": "mlp", "seed": seed, "reward": reward,
                   "genome": dqn.to_genome(net), "history": hist}, f)
    print(f"wrote {paths.champion(tag)}")
