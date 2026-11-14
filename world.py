import random

from config import GRID_SIZE, N_FOOD, N_HAZARD, MAP_SEED

EMPTY = 0
FOOD = 1
HAZARD = -1


def build_static_map(seed=MAP_SEED):
    rng = random.Random(seed)
    grid = [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    interior = [
        (r, c)
        for r in range(1, GRID_SIZE - 1)
        for c in range(1, GRID_SIZE - 1)
    ]
    rng.shuffle(interior)

    chosen = interior[: N_FOOD + N_HAZARD]
    for (r, c) in chosen[:N_FOOD]:
        grid[r][c] = FOOD
    for (r, c) in chosen[N_FOOD:]:
        grid[r][c] = HAZARD

    return grid


STATIC_MAP = build_static_map()


def copy_map(grid):
    return [row[:] for row in grid]


def empty_cells(grid):
    return [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[r]))
        if grid[r][c] == EMPTY
    ]


_GLYPHS = {EMPTY: ".", FOOD: "F", HAZARD: "X"}


def render(grid, creature_pos=None):
    lines = []
    for r, row in enumerate(grid):
        chars = []
        for c, val in enumerate(row):
            if creature_pos is not None and (r, c) == tuple(creature_pos):
                chars.append("C")
            else:
                chars.append(_GLYPHS.get(val, "?"))
        lines.append("".join(chars))
    return "\n".join(lines)
