import json
import os
import re
import subprocess
import sys

from view_radius_ablation import CONFIG_PATH, SEEDS, GENERATIONS, _set_view_radius

RADII_AND_MUTATION_RATES = {
    3: 0.10 * 404 / 788,
    4: 0.10 * 404 / 1300,
}

_TRAIN_SNIPPET = """
import json
from train import train
import architectures
from analyze_champion import profile

results = []
for seed in {seeds}:
    _hist, best = train(arch_name="linear", n_generations={gens}, seed=seed,
                         mutation_rate={mutation_rate}, verbose=False)
    policy = architectures.get("linear").make_policy(best[1])
    p = profile(policy, n=2000)
    p["seed"] = seed
    p["train_fitness"] = best[0]
    results.append(p)
print(json.dumps(results))
"""


def main():
    with open(CONFIG_PATH) as f:
        original = f.read()

    all_results = {}
    try:
        for radius, mutation_rate in RADII_AND_MUTATION_RATES.items():
            with open(CONFIG_PATH, "w") as f:
                f.write(_set_view_radius(original, radius))

            proc = subprocess.run(
                [sys.executable, "-c",
                 _TRAIN_SNIPPET.format(seeds=SEEDS, gens=GENERATIONS,
                                        mutation_rate=mutation_rate)],
                cwd=os.path.dirname(CONFIG_PATH), capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"training subprocess failed for VIEW_RADIUS={radius}")

            rows = json.loads(proc.stdout.strip().splitlines()[-1])
            all_results[radius] = rows
            print(f"VIEW_RADIUS={radius} (mutation_rate={mutation_rate:.4f}): " +
                  "  ".join(f"seed{r['seed']}={r['mean_fitness']:.1f}" for r in rows),
                  flush=True)
    finally:
        with open(CONFIG_PATH, "w") as f:
            f.write(original)

    with open("data/view_radius_ablation_scaled_mutation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY (mutation-rate scaled to genome size) ===")
    for radius, rows in all_results.items():
        n = len(rows)
        mean_fit = sum(r["mean_fitness"] for r in rows) / n
        mean_food = sum(r["mean_food"] for r in rows) / n
        print(f"VIEW_RADIUS={radius}: mean_fitness={mean_fit:7.1f}  mean_food={mean_food:5.2f}")


if __name__ == "__main__":
    main()
