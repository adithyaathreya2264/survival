import json

import architectures
import paths
from evaluate import profile, show
from heuristic_policy import make_policy as make_heuristic
from policies import random_policy


def load_policy(tag):
    champ = json.load(open(paths.champion(tag)))
    arch = architectures.get(champ.get("arch", architectures.DEFAULT_ARCH))
    return arch.make_policy(champ["genome"])


if __name__ == "__main__":
    policies = {
        "RANDOM BASELINE": random_policy,
        "HEURISTIC (greedy food / avoid hazard)": make_heuristic(),
        "GA basin C (linear_mr100_s9)": load_policy("linear_mr100_s9"),
        "DQN tuned (dqn_lr2_s5)": load_policy("dqn_lr2_s5"),
    }

    results = {}
    for name, p in policies.items():
        results[name] = profile(p)
        show(name, results[name])

    print("\n=== SUMMARY (mean fitness) ===")
    for name, r in results.items():
        print(f"  {name:42s} {r['mean']:7.1f}   food={r['mean_food']:.2f}")

    with open(paths.DATA_DIR + "/heuristic_baseline_results.json", "w") as f:
        json.dump(
            {name: {"mean": r["mean"], "median": r["median"], "mean_food": r["mean_food"],
                     "min": r["min"], "p10": r["p10"], "p90": r["p90"], "max": r["max"],
                     "causes": dict(r["causes"])}
             for name, r in results.items()},
            f, indent=2,
        )
    print(f"\nwrote {paths.DATA_DIR}/heuristic_baseline_results.json")
