import math
import random
import time

import dqn
import train_dqn
import architectures
from analyze_champion import profile
from config import N_INPUTS, HIDDEN_SIZE

SEEDS = [1, 2]


def small_init_new_net(rng):
    w1_range = 1.0 / math.sqrt(N_INPUTS)
    w2_range = 1.0 / math.sqrt(HIDDEN_SIZE)
    return {
        "W1": [[rng.uniform(-w1_range, w1_range) for _ in range(HIDDEN_SIZE)]
               for _ in range(N_INPUTS)],
        "b1": [0.0] * HIDDEN_SIZE,
        "W2": [[rng.uniform(-w2_range, w2_range) for _ in range(4)]
               for _ in range(HIDDEN_SIZE)],
        "b2": [0.0] * 4,
    }


if __name__ == "__main__":
    original_new_net = dqn.new_net
    dqn.new_net = small_init_new_net
    try:
        rows = []
        for s in SEEDS:
            t0 = time.time()
            net, _hist = train_dqn.train(seed=s, reward_name="return_matched",
                                          total_steps=500_000, lr=1e-2,
                                          log_every=10**9, verbose=False)
            p = profile(architectures.get("mlp").make_policy(dqn.to_genome(net)), n=2000)
            p["seed"] = s
            rows.append(p)
            print(f"seed {s}: fitness={p['mean_fitness']:.1f}  food={p['mean_food']:.2f}  "
                  f"time={time.time()-t0:.0f}s", flush=True)
    finally:
        dqn.new_net = original_new_net

    print("\nsmall-init hand-rolled DQN (this run)     : "
          + "  ".join(f"seed{r['seed']}={r['mean_fitness']:.1f}" for r in rows))
    print("production-init hand-rolled DQN (recorded): seed1=137.6  seed2=130.3")
