import json
import os
import re
import subprocess
import sys

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
RADII = [2, 3, 4]
SEEDS = [1, 2, 3]
GENERATIONS = 200

_TRAIN_SNIPPET = """
import json
from train import train
import architectures
from analyze_champion import profile

results = []
for seed in {seeds}:
    _hist, best = train(arch_name="linear", n_generations={gens}, seed=seed, verbose=False)
    policy = architectures.get("linear").make_policy(best[1])
    p = profile(policy, n=2000)
    p["seed"] = seed
    p["train_fitness"] = best[0]
    results.append(p)
print(json.dumps(results))
"""


def _set_view_radius(text, radius):
    new_text, n = re.subn(r"^VIEW_RADIUS = \d+", f"VIEW_RADIUS = {radius}",
                           text, flags=re.MULTILINE)
    assert n == 1, "expected exactly one VIEW_RADIUS line in config.py"
    return new_text


def main():
    with open(CONFIG_PATH) as f:
        original = f.read()

    all_results = {}
    try:
        for radius in RADII:
            with open(CONFIG_PATH, "w") as f:
                f.write(_set_view_radius(original, radius))

            proc = subprocess.run(
                [sys.executable, "-c", _TRAIN_SNIPPET.format(seeds=SEEDS, gens=GENERATIONS)],
                cwd=os.path.dirname(CONFIG_PATH), capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"training subprocess failed for VIEW_RADIUS={radius}")

            rows = json.loads(proc.stdout.strip().splitlines()[-1])
            all_results[radius] = rows
            print(f"VIEW_RADIUS={radius} (window {2*radius+1}x{2*radius+1}): " +
                  "  ".join(f"seed{r['seed']}={r['mean_fitness']:.1f}" for r in rows),
                  flush=True)
    finally:
        with open(CONFIG_PATH, "w") as f:
            f.write(original)

    os.makedirs("data", exist_ok=True)
    with open("data/view_radius_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY ===")
    for radius, rows in all_results.items():
        n = len(rows)
        mean_fit = sum(r["mean_fitness"] for r in rows) / n
        mean_food = sum(r["mean_food"] for r in rows) / n
        mean_cells = sum(r["mean_cells"] for r in rows) / n
        print(f"VIEW_RADIUS={radius}  (window {2*radius+1}x{2*radius+1}, "
              f"n={n} seeds): mean_fitness={mean_fit:7.1f}  "
              f"mean_food={mean_food:5.2f}  mean_cells={mean_cells:6.1f}")


if __name__ == "__main__":
    main()
