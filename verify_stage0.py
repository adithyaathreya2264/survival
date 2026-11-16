import random

from config import GRID_SIZE, N_FOOD, N_HAZARD, MAX_STEPS, FOOD_FITNESS_BONUS
from world import STATIC_MAP, render, FOOD, HAZARD
from perception import perceive
from simulation import run_episode_debug, run_episode_verbose
from policies import random_policy


def check_static_map():
    print("=== STATIC MAP ===")
    print(render(STATIC_MAP))

    food = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            if STATIC_MAP[r][c] == FOOD]
    haz = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
           if STATIC_MAP[r][c] == HAZARD]

    assert len(food) == N_FOOD, len(food)
    assert len(haz) == N_HAZARD, len(haz)
    assert not (set(food) & set(haz)), "food/hazard overlap"
    for (r, c) in food + haz:
        assert 0 < r < GRID_SIZE - 1 and 0 < c < GRID_SIZE - 1, (r, c)

    # Determinism: rebuilding gives the identical map.
    from world import build_static_map
    assert build_static_map() == STATIC_MAP, "map not deterministic"
    print(f"OK: {len(food)} food, {len(haz)} hazard, interior-only, deterministic")


def check_perception_edges():
    obs = perceive(STATIC_MAP, (0, 0))
    assert len(obs) == 25
    assert obs[12] == 5, "center must be P_SELF"
    assert obs[0] == 9 and obs[6] == 9, "corner window must have wall cells"
    print("OK: perception window handles edges and self-marker")


def check_random_episodes(n=2000):
    rng = random.Random(0)
    rows = [run_episode_debug(random_policy, rng) for _ in range(n)]

    for fit, cause, steps, food in rows:
        assert fit >= 0, fit
        assert steps <= MAX_STEPS, steps
        assert fit == steps + food * FOOD_FITNESS_BONUS

    from collections import Counter
    causes = Counter(r[1] for r in rows)
    fits = sorted(r[0] for r in rows)
    best = max(rows, key=lambda r: r[0])

    print(f"\n=== {n} RANDOM EPISODES ===")
    print(f"min={fits[0]} mean={sum(fits)/n:.1f} max={fits[-1]}")
    print(f"p10={fits[n//10]} median={fits[n//2]} p90={fits[9*n//10]}")
    print("termination cause: " + ", ".join(
        f"{k}={causes[k]} ({causes[k]/n:.1%})" for k in ("hazard", "starve", "cap")
    ))
    print(f"max-fitness episode: fitness={best[0]} cause={best[1]} "
          f"steps={best[2]} food={best[3]}  "
          f"[{best[2]} + {best[3]}*{FOOD_FITNESS_BONUS}]")
    print("OK: all terminated, step count hard-capped at MAX_STEPS, fitness math checks out")


if __name__ == "__main__":
    check_static_map()
    check_perception_edges()
    check_random_episodes()

    print("\n=== 2 VERBOSE EPISODES (eyeball movement/food/hazard) ===")
    for i in range(2):
        print(f"\n--- verbose episode {i} ---")
        run_episode_verbose(random_policy, random.Random(100 + i))
