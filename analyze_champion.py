import json
import random

from world import STATIC_MAP, copy_map, HAZARD
from creature import spawn
from simulation import step
from policies import random_policy
from config import GRID_SIZE, FOOD_FITNESS_BONUS
import architectures
import paths

JAM_THRESHOLD = 5  # distinct cells visited at or below this = "inert"
N = 2000


def straight_line_policy(action=1):
    return lambda obs: action


def trace(policy, rng):
    grid = copy_map(STATIC_MAP)
    c = spawn(grid, rng)
    visited = set()
    entered = 0  # distinct-cell transitions, i.e. actual exposure to new cells
    prev = (c.row, c.col)
    while True:
        visited.add((c.row, c.col))
        done = step(grid, c, policy)
        if (c.row, c.col) != prev:
            entered += 1
            prev = (c.row, c.col)
        if done:
            break
    visited.add((c.row, c.col))
    cause = "hazard" if not c.alive else ("starve" if c.energy <= 0 else "cap")
    return {
        "cause": cause,
        "cells": len(visited),
        "entered": entered,
        "steps": c.steps,
        "food": c.food_eaten,
        "fitness": c.steps + c.food_eaten * FOOD_FITNESS_BONUS,
    }


def hazard_adjacent_cells():
    out = set()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if STATIC_MAP[r][c] == HAZARD:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if STATIC_MAP[nr][nc] != HAZARD:
                            out.add((nr, nc))
    return out


FAST_DEATH_STEPS = 10  # below this, the creature died before it could navigate


def profile(policy, n=N, seed=0, name=None):
    rng = random.Random(seed)
    rows = [trace(policy, rng) for _ in range(n)]

    inert = [r for r in rows if r["cells"] <= JAM_THRESHOLD]
    moving = [r for r in rows if r["cells"] > JAM_THRESHOLD]
    # The inert stratum is two different phenomena: creatures that spawned boxed
    # in by hazards and died in ~2 steps (irreducible -- no policy can avoid it),
    # and creatures that jammed against a wall for 100 steps (0% hazard risk).
    fast_death = [r for r in rows if r["steps"] < FAST_DEATH_STEPS]
    jammed = [r for r in inert if r["steps"] >= FAST_DEATH_STEPS]

    def hz(rs):
        return sum(1 for r in rs if r["cause"] == "hazard") / len(rs) if rs else 0.0

    total_entered = sum(r["entered"] for r in rows)
    hazard_deaths = sum(1 for r in rows if r["cause"] == "hazard")
    spawn_floor = sum(1 for r in fast_death if r["cause"] == "hazard") / n
    stats = {
        "n": n,
        "mean_fitness": sum(r["fitness"] for r in rows) / n,
        "median_fitness": sorted(r["fitness"] for r in rows)[n // 2],
        "mean_cells": sum(r["cells"] for r in rows) / n,
        "mean_food": sum(r["food"] for r in rows) / n,
        "hz_all": hz(rows),
        "hz_moving": hz(moving),
        "hz_inert": hz(inert),
        "spawn_floor": spawn_floor,          # irreducible, ~1.6%
        "nav_error": hz(rows) - spawn_floor,  # the part a policy can control
        "jam_rate": len(jammed) / n,
        "per100": hazard_deaths / total_entered * 100 if total_entered else 0.0,
    }

    if name:
        print(f"\n=== {name} ({n} episodes) ===")
        print(f"mean fitness          : {stats['mean_fitness']:6.1f}  "
              f"(median {stats['median_fitness']})")
        print(f"aggregate hazard rate : {stats['hz_all']:6.1%}")
        print(f"inert episodes (<={JAM_THRESHOLD} cells): {len(inert):4d} "
              f"({len(inert)/n:5.1%})   hazard rate within: {stats['hz_inert']:6.1%}")
        print(f"moving episodes (>{JAM_THRESHOLD} cells): {len(moving):4d} "
              f"({len(moving)/n:5.1%})   hazard rate within: {stats['hz_moving']:6.1%}")
        print(f"  spawn-luck floor (died <{FAST_DEATH_STEPS} steps): "
              f"{spawn_floor:5.1%}   navigation error: {stats['nav_error']:5.1%}")
        print(f"jammed (>=100 steps, <={JAM_THRESHOLD} cells): {stats['jam_rate']:5.1%}")
        print(f"mean distinct cells   : {stats['mean_cells']:6.1f}   "
              f"mean food: {stats['mean_food']:.2f}")
        print(f"EXPOSURE-ADJUSTED     : {stats['per100']:6.3f} "
              f"hazard deaths per 100 cell-entries")
    return stats


def load_policy(path):
    champ = json.load(open(path))
    arch = architectures.get(champ.get("arch", architectures.DEFAULT_ARCH))
    return arch.make_policy(champ["genome"])


if __name__ == "__main__":
    import sys

    path = paths.champion(sys.argv[1]) if len(sys.argv) > 1 else paths.DEFAULT_CHAMPION
    policies = [
        ("RANDOM WALK", random_policy),
        ("STRAIGHT LINE (true gen-1 analogue)", straight_line_policy()),
        ("TRAINED CHAMPION", load_policy(path)),
    ]
    results = {name: profile(p, name=name) for name, p in policies}

    print("\n=== VERDICT ===")
    c = results["TRAINED CHAMPION"]
    s = results["STRAIGHT LINE (true gen-1 analogue)"]
    r = results["RANDOM WALK"]
    print(f"champion hazard rate on MOVING episodes only : {c['hz_moving']:.1%}")
    print(f"  vs straight-line moving episodes           : {s['hz_moving']:.1%}")
    print(f"  vs random-walk moving episodes             : {r['hz_moving']:.1%}")
    print("\nper 100 cell-entries (exposure-adjusted, jamming cannot game this):")
    for name in results:
        print(f"  {name:38s} {results[name]['per100']:6.3f}")

    danger = hazard_adjacent_cells()
    print(f"\n(the map has {len(danger)} hazard-adjacent cells out of 900)")
