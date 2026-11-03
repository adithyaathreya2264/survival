import json
import statistics
import time

from train import train
import architectures
import paths
from analyze_champion import profile

SEEDS = [1, 2, 3, 4, 5]
ARMS = [
    ("linear", 0.10),
    ("mlp", 0.10),
    ("mlp", 0.024),  # matches Stage 1's ~40 absolute genes mutated per child
]
GENERATIONS = 200
EVAL_EPISODES = 2000

METRICS = [
    ("mean_fitness", "mean fitness", "{:.1f}"),
    ("median_fitness", "median fitness", "{:.0f}"),
    ("mean_food", "food eaten", "{:.2f}"),
    ("mean_cells", "distinct cells", "{:.1f}"),
    ("hz_all", "hazard rate", "{:.1%}"),
    ("hz_moving", "hazard (moving)", "{:.1%}"),
    ("nav_error", "nav error", "{:.1%}"),
    ("per100", "hazard/100 entries", "{:.3f}"),
    ("jam_rate", "jam rate", "{:.1%}"),
]


def arm_label(arch, rate):
    return f"{arch} @{rate:.1%}"


def run_arm(arch, rate):
    stats = []
    for seed in SEEDS:
        t0 = time.time()
        tag = f"{arch}_mr{int(rate*1000)}_s{seed}"
        _hist, best = train(
            arch_name=arch, n_generations=GENERATIONS, seed=seed,
            mutation_rate=rate, champion_path=paths.champion(tag),
            log_path=paths.log(tag), verbose=False,
        )
        policy = architectures.get(arch).make_policy(best[1])
        s = profile(policy, n=EVAL_EPISODES)
        s["train_fitness"] = best[0]
        stats.append(s)
        print(f"  seed {seed}: train_best={best[0]:5.0f}  "
              f"eval_mean={s['mean_fitness']:6.1f}  hazard={s['hz_all']:5.1%}  "
              f"({time.time()-t0:.0f}s)")
    return stats


def summarize(stats, key):
    vals = [s[key] for s in stats]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


if __name__ == "__main__":
    results = {}
    for arch, rate in ARMS:
        label = arm_label(arch, rate)
        print(f"\n=== training arm: {label} ({len(SEEDS)} seeds x {GENERATIONS} gens) ===")
        results[label] = run_arm(arch, rate)

    labels = [arm_label(a, r) for a, r in ARMS]
    width = max(len(l) for l in labels) + 2

    print("\n\n" + "=" * 78)
    print("STAGE 1 vs STAGE 2 -- mean +/- stdev across seeds")
    print("=" * 78)
    header = f"{'metric':<22}" + "".join(f"{l:>{width+8}}" for l in labels)
    print(header)
    print("-" * len(header))
    for key, name, fmt in METRICS:
        row = f"{name:<22}"
        for label in labels:
            m, sd = summarize(results[label], key)
            row += f"{fmt.format(m) + ' +/- ' + fmt.format(sd):>{width+8}}"
        print(row)

    print("\n=== VERDICT ===")
    base = results[labels[0]]
    bm, bsd = summarize(base, "mean_fitness")
    for label in labels[1:]:
        m, sd = summarize(results[label], "mean_fitness")
        delta = m - bm
        pooled = (bsd ** 2 + sd ** 2) ** 0.5
        verdict = ("BEATS baseline beyond seed spread" if abs(delta) > pooled
                   else "TIE -- within seed spread, not evidence")
        print(f"{label:<16} mean fitness {m:6.1f} vs {bm:6.1f} "
              f"(delta {delta:+6.1f}, pooled spread {pooled:5.1f})  -> {verdict}")

    with open(paths.COMPARISON_RESULTS, "w") as f:
        json.dump(
            {l: [{k: v for k, v in s.items()} for s in results[l]] for l in labels},
            f, indent=2,
        )
    print(f"\nwrote {paths.COMPARISON_RESULTS}")
