import time

import dqn
import train_dqn
import architectures
from analyze_champion import profile
from dqn_init_ablation import small_init_new_net

SEEDS = [3, 4, 5]

if __name__ == "__main__":
    original_new_net = dqn.new_net
    dqn.new_net = small_init_new_net
    try:
        for s in SEEDS:
            t0 = time.time()
            net, _hist = train_dqn.train(seed=s, reward_name="return_matched",
                                          total_steps=500_000, lr=1e-2,
                                          log_every=10**9, verbose=False)
            p = profile(architectures.get("mlp").make_policy(dqn.to_genome(net)), n=2000)
            print(f"seed {s}: fitness={p['mean_fitness']:.1f}  food={p['mean_food']:.2f}  "
                  f"time={time.time()-t0:.0f}s", flush=True)
    finally:
        dqn.new_net = original_new_net
