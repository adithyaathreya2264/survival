import random

from config import (
    R_STEP, R_FOOD, R_HAZARD,
    R_SPARSE_STEP, R_SPARSE_FOOD, R_SPARSE_HAZARD,
    FOOD_FITNESS_BONUS,
)
from world import STATIC_MAP, copy_map
from creature import spawn
from perception import perceive
from simulation import apply_action


class Reward:

    def __init__(self, step_r, food_r, hazard_r, scale=1.0, name="reward"):
        self.step_r, self.food_r, self.hazard_r = step_r, food_r, hazard_r
        self.scale, self.name = scale, name

    def __call__(self, info):
        r = self.step_r
        if info["ate"]:
            r += self.food_r
        if info["died_hazard"]:
            r += self.hazard_r
        return r * self.scale


def return_matched(scale=1.0):
    return Reward(R_STEP, R_FOOD, R_HAZARD, scale, "return_matched")


def sparse(scale=1.0):
    return Reward(R_SPARSE_STEP, R_SPARSE_FOOD, R_SPARSE_HAZARD, scale, "sparse")


class GridEnv:

    def __init__(self, reward_fn=None, rng=None):
        self.reward_fn = reward_fn or return_matched()
        self.rng = rng or random.Random()
        self.grid = None
        self.creature = None
        self.visited = None

    def reset(self, rng=None, spawn_at=None):
        self.grid = copy_map(STATIC_MAP)
        if spawn_at is None:
            self.creature = spawn(self.grid, rng or self.rng)
        else:  # forced spawn, for the problem-tile test
            self.creature = spawn(self.grid, random.Random(0))
            self.creature.row, self.creature.col = spawn_at
        self.visited = {(self.creature.row, self.creature.col)}
        return self.observe()

    def observe(self):
        return perceive(self.grid, (self.creature.row, self.creature.col))

    def step(self, action):
        done, info = apply_action(self.grid, self.creature, action)
        c = self.creature
        self.visited.add((c.row, c.col))
        reward = self.reward_fn(info)
        if done:
            info["cause"] = ("hazard" if not c.alive
                             else "starve" if c.energy <= 0 else "cap")
            info["fitness"] = c.steps + c.food_eaten * FOOD_FITNESS_BONUS
            info["steps"] = c.steps
            info["food"] = c.food_eaten
            info["cells"] = len(self.visited)
        return self.observe(), reward, done, info
