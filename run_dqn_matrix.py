import json
import time

from config import DQN_SEEDS, TOTAL_ENV_STEPS
from analyze_champion import profile
import architectures
import train_dqn
import dqn
import paths

ARMS = ["return_matched", "sparse"]


def main():
    results = {}
    for arm in ARMS:
        rows = []
        print(f"\n=== DQN arm: {arm} ({len(DQN_SEEDS)} seeds x {TOTAL_ENV_STEPS} steps) ===",
              flush=True)
        for seed in DQN_SEEDS:
            t0 = time.time()
            net, hist = train_dqn.train(seed=seed, reward_name=arm,
                                        total_steps=TOTAL_ENV_STEPS,
                                        log_every=100_000, verbose=False)
            genome = dqn.to_genome(net)
            s = profile(architectures.get("mlp").make_policy(genome), n=2000)
            s["seed"] = seed
            s["arm"] = arm
            rows.append(s)
            with open(paths.champion(f"dqn_{arm}_s{seed}"), "w") as f:
                json.dump({"arch": "mlp", "seed": seed, "reward": arm,
                           "genome": genome, "history": hist}, f)
            print(f"  seed {seed}: eval_mean={s['mean_fitness']:7.1f} "
                  f"food={s['mean_food']:5.2f} cells={s['mean_cells']:6.1f} "
                  f"haz={s['hz_all']:5.1%} jam={s['jam_rate']:5.1%} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results[arm] = rows

    with open(paths.DATA_DIR + "/dqn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {paths.DATA_DIR}/dqn_results.json")


if __name__ == "__main__":
    main()
