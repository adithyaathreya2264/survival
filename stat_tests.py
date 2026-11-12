import json
import random
import statistics as st

N_PERMUTATIONS = 100_000
N_BOOTSTRAP = 100_000
SEED = 0


def permutation_test(a, b, rng, n=N_PERMUTATIONS):
    observed = st.mean(a) - st.mean(b)
    pooled = list(a) + list(b)
    na = len(a)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        diff = st.mean(pooled[:na]) - st.mean(pooled[na:])
        if abs(diff) >= abs(observed):
            count += 1
    return observed, count / n


def bootstrap_ci(a, b, rng, n=N_BOOTSTRAP, alpha=0.05):
    diffs = []
    for _ in range(n):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        diffs.append(st.mean(ra) - st.mean(rb))
    diffs.sort()
    lo = diffs[int(n * alpha / 2)]
    hi = diffs[int(n * (1 - alpha / 2))]
    return lo, hi


def report(name_a, a, name_b, b, rng):
    print(f"\n=== {name_a} (n={len(a)}, mean={st.mean(a):.1f}) vs "
          f"{name_b} (n={len(b)}, mean={st.mean(b):.1f}) ===")
    observed, p = permutation_test(a, b, rng)
    lo, hi = bootstrap_ci(a, b, rng)
    print(f"observed difference in means: {observed:+.1f}")
    print(f"permutation test (two-sided, {N_PERMUTATIONS} shuffles): p = {p:.4f}")
    print(f"95% bootstrap CI for the difference: [{lo:+.1f}, {hi:+.1f}]")
    if lo <= 0 <= hi:
        print("-> CI includes 0: cannot distinguish the two means at 95% confidence")
    else:
        print("-> CI excludes 0: the difference is statistically significant at 95%")
    return {"observed_diff": observed, "p_value": p, "ci_95": [lo, hi]}


if __name__ == "__main__":
    rng = random.Random(SEED)

    dqn = json.load(open("data/dqn_lr2_results.json"))
    dqn_fits = [r["mean_fitness"] for r in dqn]

    ga = json.load(open("data/linear_20seeds.json"))
    basin_a_fits = [r["mean_fitness"] for r in ga if r["mean_fitness"] < 230]

    results = {}
    results["dqn_vs_ga_basin_a"] = report(
        "DQN tuned (lr=1e-2)", dqn_fits, "GA basin A", basin_a_fits, rng
    )

    with open("data/stat_tests_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote data/stat_tests_results.json")
