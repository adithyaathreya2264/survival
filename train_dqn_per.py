import random

from config import (
    GAMMA, EPSILON_START, EPSILON_FLOOR, EPSILON_DECAY_FRACTION,
    REPLAY_CAPACITY, REPLAY_WARMUP, BATCH_SIZE, TRAIN_EVERY, TARGET_SYNC_STEPS,
    GRAD_CLIP, TOTAL_ENV_STEPS, REWARD_SCALE,
)
from actions import N_ACTIONS
from linear_policy import encode
import env as envmod
import dqn
from prioritized_replay import PrioritizedReplay


def train(seed=1, total_steps=TOTAL_ENV_STEPS, lr=1e-2, verbose=False, log_every=25_000):
    rng = random.Random(seed)
    reward_fn = envmod.return_matched(REWARD_SCALE)
    e = envmod.GridEnv(reward_fn, rng)

    net = dqn.new_net(rng)
    target_net = dqn.clone(net)
    buffer = PrioritizedReplay(REPLAY_CAPACITY, rng=rng)

    decay_steps = max(1, int(total_steps * EPSILON_DECAY_FRACTION))
    obs = e.reset(rng)
    active = encode(obs)
    history = []

    for t in range(total_steps):
        eps = max(EPSILON_FLOOR,
                  EPSILON_START + (EPSILON_FLOOR - EPSILON_START) * t / decay_steps)
        if rng.random() < eps:
            action = rng.randrange(N_ACTIONS)
        else:
            _a, q = dqn.forward(net, active)
            action = max(range(N_ACTIONS), key=lambda k: q[k])

        nobs, reward, done, info = e.step(action)
        nactive = encode(nobs)
        buffer.add((active, action, reward, nactive, done))

        active = encode(e.reset(rng)) if done else nactive

        if len(buffer) >= REPLAY_WARMUP and t % TRAIN_EVERY == 0:
            indices, batch = buffer.sample(BATCH_SIZE)
            grads = dqn.zero_grads()
            errors = []
            for (s, a, r, s2, d) in batch:
                y = r if d else r + GAMMA * max(dqn.forward(target_net, s2)[1])
                errors.append(dqn.td_error(net, s, a, y))
                dqn.accumulate(net, grads, s, a, y)
            dqn.apply_grads(net, grads, lr, scale=1.0 / BATCH_SIZE, clip=GRAD_CLIP)
            buffer.update_priorities(indices, errors)

        if t % TARGET_SYNC_STEPS == 0:
            target_net = dqn.clone(net)

        if verbose and (t + 1) % log_every == 0:
            from train_dqn import evaluate_greedy
            ev = evaluate_greedy(net, n_episodes=100)
            history.append({"step": t + 1, "greedy_fitness": ev["fitness"]})
            print(f"  t={t+1:>7} eps={eps:.2f} greedy_fitness={ev['fitness']:7.1f}", flush=True)

    return net, history


if __name__ == "__main__":
    import sys
    import time
    import json
    import statistics as st
    import architectures
    from analyze_champion import profile

    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
    rows = []
    for s in seeds:
        t0 = time.time()
        net, _hist = train(seed=s, lr=1e-2, verbose=True)
        p = profile(architectures.get("mlp").make_policy(dqn.to_genome(net)), n=2000)
        p["seed"] = s
        rows.append(p)
        print(f"seed {s}: fitness={p['mean_fitness']:.1f}  food={p['mean_food']:.2f}  "
              f"time={time.time()-t0:.0f}s", flush=True)

    fits = [r["mean_fitness"] for r in rows]
    json.dump(rows, open("data/dqn_per_results.json", "w"), indent=2)
    print(f"\nPrioritized-replay DQN, {len(seeds)} seeds: "
          f"mean={st.mean(fits):.1f}" + (f" +/- {st.stdev(fits):.1f}" if len(fits) > 1 else ""))
    print("uniform-replay DQN (lr=1e-2, 5 seeds): mean=185.2 +/- 52.2  "
          "(data/dqn_lr2_results.json)")
