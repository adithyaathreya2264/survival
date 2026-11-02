from collections import deque

from world import STATIC_MAP, FOOD, HAZARD, empty_cells
from config import GRID_SIZE, START_ENERGY, FOOD_ENERGY, FOOD_FITNESS_BONUS, MAX_STEPS

NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UNREACHABLE = float("inf")


def food_cells(grid):
    cells = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if grid[r][c] == FOOD
    ]
    assert len(cells) == 18, f"expected 18 food cells, found {len(cells)}"
    return cells


def bfs(grid, source):
    dist = {source: 0}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if (nr, nc) in dist or grid[nr][nc] == HAZARD:
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            queue.append((nr, nc))
    return dist


def greedy_run(spawn, foods, dist_from):
    energy = START_ENERGY
    pos = spawn
    remaining = set(foods)
    steps = 0
    route = []

    while remaining:
        best, best_d = None, UNREACHABLE
        for f in remaining:
            d = dist_from[f].get(pos, UNREACHABLE)
            if d < best_d:
                best, best_d = f, d
        if best is None or best_d > energy:
            break
        energy -= best_d
        steps += best_d
        energy += FOOD_ENERGY
        remaining.discard(best)
        pos = best
        route.append(best)

    return steps, len(route), route, energy


def main():
    foods = food_cells(STATIC_MAP)
    dist_from = {f: bfs(STATIC_MAP, f) for f in foods}

    spawns = empty_cells(STATIC_MAP)
    results = []
    for spawn in spawns:
        steps, eaten, route, leftover = greedy_run(spawn, foods, dist_from)
        total_steps = steps + leftover
        fitness = total_steps + eaten * FOOD_FITNESS_BONUS
        results.append((total_steps, steps, eaten, fitness, spawn, route))

    results.sort(reverse=True)
    best = results[0]
    total_steps, purposeful, eaten, fitness, spawn, route = best

    print(f"candidate spawns evaluated : {len(spawns)}")
    print(f"food cells                 : {len(foods)}")

    print("\n=== GREEDY LOWER-BOUND ESTIMATE ===")
    print(f"best spawn              : {spawn}")
    print(f"food collected          : {eaten} of {len(foods)}")
    print(f"purposeful steps        : {purposeful}")
    print(f"+ final wander to death : {total_steps - purposeful}")
    print(f"TOTAL STEPS SURVIVED    : {total_steps}")
    print(f"implied fitness         : {total_steps} + {eaten}*{FOOD_FITNESS_BONUS} "
          f"= {fitness}")
    print(f"current MAX_STEPS cap   : {MAX_STEPS}  -> "
          f"{'BINDS' if total_steps >= MAX_STEPS else 'does not bind'}")

    print("\ntop 5 spawns by steps survived:")
    for ts, ps, ea, fit, sp, _ in results[:5]:
        print(f"  spawn={str(sp):9s} steps={ts:4d} food={ea:2d} fitness={fit:4d}")

    steps_all = [r[0] for r in results]
    eaten_all = [r[2] for r in results]
    print(f"\nacross all {len(spawns)} spawns: "
          f"min={min(steps_all)} mean={sum(steps_all)/len(steps_all):.1f} "
          f"max={max(steps_all)}")
    print(f"food collected: min={min(eaten_all)} "
          f"mean={sum(eaten_all)/len(eaten_all):.2f} max={max(eaten_all)}")

    # How much do hazards actually distort distances? (No interior walls exist,
    # so this isolates the hazard-detour effect Manhattan would have missed.)
    worst_ratio, worst_pair, detours = 1.0, None, 0
    for i, a in enumerate(foods):
        for b in foods[i + 1:]:
            man = abs(a[0] - b[0]) + abs(a[1] - b[1])
            true_d = dist_from[a].get(b, UNREACHABLE)
            if true_d > man:
                detours += 1
                if true_d / man > worst_ratio:
                    worst_ratio, worst_pair = true_d / man, (a, b, man, true_d)
    print(f"\nBFS vs Manhattan across {len(foods)*(len(foods)-1)//2} food pairs: "
          f"{detours} pairs require a detour")
    if worst_pair:
        a, b, man, td = worst_pair
        print(f"  worst: {a}->{b} manhattan={man} true={td} ({worst_ratio:.2f}x)")

    print("\nCAVEAT: greedy nearest-first is a LOWER BOUND, not the true maximum.")
    print("A smarter route may skip a near food to reach a cluster more cheaply.")


if __name__ == "__main__":
    main()
