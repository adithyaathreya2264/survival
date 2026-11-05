import json
import statistics as st

from train import train
import architectures
from analyze_champion import profile
import paths

NEW_SEEDS = list(range(6, 21))
ORIG_SEEDS = [1, 2, 3, 4, 5]
GENERATIONS = 200
RATE = 0.10
EVAL_EPISODES = 2000
ARCH = "linear"


def train_and_profile(seed):
    tag = f"{ARCH}_mr100_s{seed}"
    _hist, best = train(
        arch_name=ARCH, n_generations=GENERATIONS, seed=seed, mutation_rate=RATE,
        champion_path=paths.champion(tag), log_path=paths.log(tag), verbose=False,
    )
    policy = architectures.get(ARCH).make_policy(best[1])
    s = profile(policy, n=EVAL_EPISODES)
    s["seed"] = seed
    s["train_fitness"] = best[0]
    return s


def profile_existing(seed):
    tag = f"{ARCH}_mr100_s{seed}"
    champ = json.load(open(paths.champion(tag)))
    policy = architectures.get(champ.get("arch", "linear")).make_policy(champ["genome"])
    s = profile(policy, n=EVAL_EPISODES)
    s["seed"] = seed
    s["train_fitness"] = champ["fitness"]
    return s


if __name__ == "__main__":
    rows = []
    for seed in ORIG_SEEDS:
        s = profile_existing(seed)
        rows.append(s)
        print(f"seed {seed:2d} (existing): eval_mean={s['mean_fitness']:6.1f}  "
              f"food={s['mean_food']:5.2f}  cells={s['mean_cells']:6.1f}  "
              f"haz/100={s['per100']:.3f}")
    for seed in NEW_SEEDS:
        s = train_and_profile(seed)
        rows.append(s)
        print(f"seed {seed:2d} (new)     : eval_mean={s['mean_fitness']:6.1f}  "
              f"food={s['mean_food']:5.2f}  cells={s['mean_cells']:6.1f}  "
              f"haz/100={s['per100']:.3f}")

    means = sorted(r["mean_fitness"] for r in rows)
    print("\n=== 20 LINEAR SEEDS ===")
    print("sorted eval mean fitness:", [round(m) for m in means])
    print(f"median={st.median(means):.1f}  mean={st.mean(means):.1f}  "
          f"stdev={st.stdev(means):.1f}")
    print(f"min={means[0]:.1f}  max={means[-1]:.1f}")
    over_500 = [m for m in means if m > 500]
    over_400 = [m for m in means if m > 400]
    print(f"seeds > 400: {len(over_400)}/20   seeds > 500: {len(over_500)}/20")

    best = max(rows, key=lambda r: r["mean_fitness"])
    print(f"\nbest seed: {best['seed']}  eval_mean={best['mean_fitness']:.1f}  "
          f"food={best['mean_food']:.2f}  cells={best['mean_cells']:.1f}")

    with open(paths.DATA_DIR + "/linear_20seeds.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {paths.DATA_DIR}/linear_20seeds.json")
