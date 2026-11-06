import json
import statistics as st

from train import train
import architectures
from analyze_champion import profile
import paths

ORIG_SEEDS = [1, 2, 3, 4, 5]
NEW_SEEDS = list(range(6, 16))
GENERATIONS = 200
EVAL_EPISODES = 2000
ARCH = "mlp"
ARMS = {"mlp_mr100": 0.10, "mlp_mr24": 0.024}


def train_and_profile(seed, tag_prefix, mutation_rate):
    tag = f"{tag_prefix}_s{seed}"
    _hist, best = train(
        arch_name=ARCH, n_generations=GENERATIONS, seed=seed, mutation_rate=mutation_rate,
        champion_path=paths.champion(tag), log_path=paths.log(tag), verbose=False,
    )
    policy = architectures.get(ARCH).make_policy(best[1])
    s = profile(policy, n=EVAL_EPISODES)
    s["seed"] = seed
    s["train_fitness"] = best[0]
    return s


def profile_existing(seed, tag_prefix):
    tag = f"{tag_prefix}_s{seed}"
    champ = json.load(open(paths.champion(tag)))
    policy = architectures.get(champ.get("arch", ARCH)).make_policy(champ["genome"])
    s = profile(policy, n=EVAL_EPISODES)
    s["seed"] = seed
    s["train_fitness"] = champ["fitness"]
    return s


if __name__ == "__main__":
    all_arms = {}
    for tag_prefix, rate in ARMS.items():
        rows = []
        for seed in ORIG_SEEDS:
            s = profile_existing(seed, tag_prefix)
            rows.append(s)
            print(f"[{tag_prefix}] seed {seed:2d} (existing): "
                  f"eval_mean={s['mean_fitness']:6.1f}  food={s['mean_food']:5.2f}",
                  flush=True)
        for seed in NEW_SEEDS:
            s = train_and_profile(seed, tag_prefix, rate)
            rows.append(s)
            print(f"[{tag_prefix}] seed {seed:2d} (new)     : "
                  f"eval_mean={s['mean_fitness']:6.1f}  food={s['mean_food']:5.2f}",
                  flush=True)
        all_arms[tag_prefix] = rows

    print("\n=== SUMMARY (15 seeds/arm) ===")
    for tag_prefix, rows in all_arms.items():
        means = [r["mean_fitness"] for r in rows]
        print(f"{tag_prefix} (mutation_rate={ARMS[tag_prefix]}): "
              f"mean={st.mean(means):.1f}  stdev={st.stdev(means):.1f}  "
              f"median={st.median(means):.1f}  n={len(means)}")

    with open(paths.DATA_DIR + "/mlp_15seeds.json", "w") as f:
        json.dump(all_arms, f, indent=2)
    print(f"\nwrote {paths.DATA_DIR}/mlp_15seeds.json")
