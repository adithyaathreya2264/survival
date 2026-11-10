import json
import statistics as st
import time

import reinforce as rf

SEEDS = [1, 2, 3, 4, 5]
N_EPISODES = 20_000

if __name__ == "__main__":
    rows = []
    for s in SEEDS:
        t0 = time.time()
        net, _hist = rf.train(seed=s, n_episodes=N_EPISODES, verbose=False)
        ev = rf.evaluate_greedy(net, n_episodes=2000)
        rows.append({"seed": s, "mean_fitness": ev["fitness"]})
        print(f"seed {s}: fitness={ev['fitness']:.1f}  time={time.time()-t0:.0f}s", flush=True)

    fits = [r["mean_fitness"] for r in rows]
    json.dump(rows, open("data/reinforce_results.json", "w"), indent=2)
    print(f"\nREINFORCE, {N_EPISODES} episodes, 5 seeds: "
          f"mean={st.mean(fits):.1f} +/- {st.stdev(fits):.1f}  "
          f"range={min(fits):.1f}-{max(fits):.1f}")
