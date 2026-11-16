import json
import random
import statistics as st

import architectures
import paths
from config import FOOD_FITNESS_BONUS, MAP_SEED
from creature import spawn
from policies import random_policy
from simulation import step
from world import build_static_map, copy_map

N_EPISODES = 500
NEW_MAP_SEEDS = list(range(1, 11))  # 10 unseen maps, distinct from MAP_SEED=12345

GA_CHAMPION = "linear_mr100_s9"         # basin C, README-reported mean 757.6
DQN_CHAMPION = "dqn_lr2_s5"  # tuned (lr=1e-2), best of 5 DQN seeds, README-reported 253.3


def load_policy(tag):
    champ = json.load(open(paths.champion(tag)))
    arch = architectures.get(champ.get("arch", architectures.DEFAULT_ARCH))
    return arch.make_policy(champ["genome"])


def run_episode(policy, grid_template, rng):
    grid = copy_map(grid_template)
    c = spawn(grid, rng)
    visited = {(c.row, c.col)}
    done = False
    while not done:
        done = step(grid, c, policy)
        visited.add((c.row, c.col))
    cause = "hazard" if not c.alive else ("starve" if c.energy <= 0 else "cap")
    fitness = c.steps + c.food_eaten * FOOD_FITNESS_BONUS
    return fitness, cause, c.food_eaten, len(visited)


def profile_on_map(policy, grid, n=N_EPISODES, seed=0):
    rng = random.Random(seed)
    rows = [run_episode(policy, grid, rng) for _ in range(n)]
    fits = [r[0] for r in rows]
    return {
        "mean_fitness": sum(fits) / n,
        "median_fitness": sorted(fits)[n // 2],
        "mean_food": sum(r[2] for r in rows) / n,
        "mean_cells": sum(r[3] for r in rows) / n,
        "hazard_rate": sum(1 for r in rows if r[1] == "hazard") / n,
    }


def main():
    policies = {
        "random baseline": random_policy,
        f"GA basin C ({GA_CHAMPION})": load_policy(GA_CHAMPION),
        f"DQN tuned ({DQN_CHAMPION})": load_policy(DQN_CHAMPION),
    }

    train_grid = build_static_map(MAP_SEED)
    new_grids = {seed: build_static_map(seed) for seed in NEW_MAP_SEEDS}

    results = {}
    for name, policy in policies.items():
        print(f"\n=== {name} ===")
        row = {"train_map": profile_on_map(policy, train_grid)}
        t = row["train_map"]
        print(f"  {'training map (seed=' + str(MAP_SEED) + ')':28s} "
              f"fitness={t['mean_fitness']:7.1f}  food={t['mean_food']:5.2f}  "
              f"cells={t['mean_cells']:6.1f}  hazard={t['hazard_rate']:5.1%}")

        for seed in NEW_MAP_SEEDS:
            row[f"map_{seed}"] = profile_on_map(policy, new_grids[seed])

        new_fits = [row[f"map_{s}"]["mean_fitness"] for s in NEW_MAP_SEEDS]
        new_food = [row[f"map_{s}"]["mean_food"] for s in NEW_MAP_SEEDS]
        new_cells = [row[f"map_{s}"]["mean_cells"] for s in NEW_MAP_SEEDS]
        new_haz = [row[f"map_{s}"]["hazard_rate"] for s in NEW_MAP_SEEDS]

        for seed in NEW_MAP_SEEDS:
            m = row[f"map_{seed}"]
            print(f"  {'unseen map seed=' + str(seed):28s} "
                  f"fitness={m['mean_fitness']:7.1f}  food={m['mean_food']:5.2f}  "
                  f"cells={m['mean_cells']:6.1f}  hazard={m['hazard_rate']:5.1%}")

        summary = {
            "mean_fitness_unseen": st.mean(new_fits),
            "stdev_fitness_unseen": st.stdev(new_fits),
            "mean_food_unseen": st.mean(new_food),
            "mean_cells_unseen": st.mean(new_cells),
            "mean_hazard_unseen": st.mean(new_haz),
        }
        row["summary"] = summary
        results[name] = row

        print(f"  {'--- unseen-map mean (n=' + str(len(NEW_MAP_SEEDS)) + ') ---':28s} "
              f"fitness={summary['mean_fitness_unseen']:7.1f} "
              f"(+/-{summary['stdev_fitness_unseen']:.1f})  "
              f"food={summary['mean_food_unseen']:5.2f}  "
              f"cells={summary['mean_cells_unseen']:6.1f}  "
              f"hazard={summary['mean_hazard_unseen']:5.1%}")
        drop = t["mean_fitness"] - summary["mean_fitness_unseen"]
        print(f"  {'train -> unseen fitness drop':28s} {drop:+7.1f}  "
              f"({drop / t['mean_fitness']:+.1%})")

    out_path = paths.DATA_DIR + "/generalization_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
