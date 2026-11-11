import random

from config import (
    GRID_SIZE,
    ENERGY_PER_STEP,
    FOOD_ENERGY,
    MAX_STEPS,
    FOOD_FITNESS_BONUS,
)
from world import STATIC_MAP, copy_map, FOOD, HAZARD
from creature import spawn
from perception import perceive
from actions import ACTIONS


def apply_action(grid, creature, action):
    dr, dc = ACTIONS[action]
    nr, nc = creature.row + dr, creature.col + dc

    ate = False
    blocked = False
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        # Out of bounds: rejected move, creature stays put (still costs a step).
        blocked = True
    else:
        target = grid[nr][nc]
        creature.row, creature.col = nr, nc
        if target == HAZARD:
            creature.alive = False
        elif target == FOOD:
            grid[nr][nc] = 0
            creature.energy += FOOD_ENERGY
            creature.food_eaten += 1
            ate = True

    creature.energy -= ENERGY_PER_STEP
    creature.steps += 1

    terminated = (
        not creature.alive
        or creature.energy <= 0
        or creature.steps >= MAX_STEPS
    )
    info = {
        "ate": ate,
        "blocked": blocked,
        "died_hazard": not creature.alive,
        "starved": creature.alive and creature.energy <= 0,
    }
    return terminated, info


def step(grid, creature, policy):
    obs = perceive(grid, (creature.row, creature.col))
    terminated, _info = apply_action(grid, creature, policy(obs))
    return terminated


def run_episode(policy, rng=None):
    rng = rng or random.Random()
    grid = copy_map(STATIC_MAP)
    creature = spawn(grid, rng)

    while not step(grid, creature, policy):
        pass

    return creature.steps + creature.food_eaten * FOOD_FITNESS_BONUS


def run_episode_debug(policy, rng=None):
    rng = rng or random.Random()
    grid = copy_map(STATIC_MAP)
    creature = spawn(grid, rng)

    while not step(grid, creature, policy):
        pass

    if not creature.alive:
        cause = "hazard"
    elif creature.energy <= 0:
        cause = "starve"
    else:
        cause = "cap"
    fitness = creature.steps + creature.food_eaten * FOOD_FITNESS_BONUS
    return fitness, cause, creature.steps, creature.food_eaten


def run_episode_verbose(policy, rng=None):
    from world import render

    rng = rng or random.Random()
    grid = copy_map(STATIC_MAP)
    creature = spawn(grid, rng)

    print(f"spawn at ({creature.row}, {creature.col})")
    done = False
    while not done:
        done = step(grid, creature, policy)
        print(
            f"\nstep={creature.steps} energy={creature.energy} "
            f"food={creature.food_eaten} alive={creature.alive}"
        )
        print(render(grid, (creature.row, creature.col)))

    fitness = creature.steps + creature.food_eaten * FOOD_FITNESS_BONUS
    print(f"\nEPISODE END  fitness={fitness}")
    return fitness
