import json
import random
import sys
from collections import Counter

from simulation import run_episode_debug, run_episode_verbose
from linear_policy import make_policy
from policies import random_policy
import paths

CHAMPION_PATH = paths.DEFAULT_CHAMPION
N_EVAL = 2000


def load_champion(path=None):
    with open(path or CHAMPION_PATH) as f:
        return json.load(f)


def profile(policy, n=N_EVAL, seed=0):
    rng = random.Random(seed)
    rows = [run_episode_debug(policy, rng) for _ in range(n)]
    fits = sorted(r[0] for r in rows)
    causes = Counter(r[1] for r in rows)
    return {
        "min": fits[0],
        "p10": fits[n // 10],
        "median": fits[n // 2],
        "mean": sum(fits) / n,
        "p90": fits[9 * n // 10],
        "max": fits[-1],
        "mean_food": sum(r[3] for r in rows) / n,
        "causes": causes,
        "n": n,
    }


def show(name, p):
    c, n = p["causes"], p["n"]
    print(f"\n=== {name} ({n} episodes) ===")
    print(f"min={p['min']}  p10={p['p10']}  median={p['median']}  "
          f"mean={p['mean']:.1f}  p90={p['p90']}  max={p['max']}")
    print(f"mean food eaten: {p['mean_food']:.2f}")
    print("termination cause: " + ", ".join(
        f"{k}={c[k]} ({c[k]/n:.1%})" for k in ("hazard", "starve", "cap")
    ))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    champ = load_champion(paths.champion(args[0]) if args else None)
    policy = make_policy(champ["genome"])

    if "--verbose" in sys.argv:
        run_episode_verbose(policy, random.Random(champ["spawn_seed"]))
        sys.exit(0)

    baseline = profile(random_policy)
    trained = profile(policy)

    show("RANDOM BASELINE", baseline)
    show("TRAINED CHAMPION", trained)

    print("\n=== DELTA ===")
    print(f"mean fitness:  {baseline['mean']:.1f} -> {trained['mean']:.1f}  "
          f"({trained['mean'] - baseline['mean']:+.1f})")
    bh = baseline["causes"]["hazard"] / baseline["n"]
    th = trained["causes"]["hazard"] / trained["n"]
    print(f"hazard deaths: {bh:.1%} -> {th:.1%}  ({th - bh:+.1%})")
    print(f"mean food:     {baseline['mean_food']:.2f} -> {trained['mean_food']:.2f}")
