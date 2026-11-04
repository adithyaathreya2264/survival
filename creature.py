from dataclasses import dataclass

from config import START_ENERGY
from world import empty_cells


@dataclass
class Creature:
    row: int
    col: int
    energy: int = START_ENERGY
    steps: int = 0
    food_eaten: int = 0
    alive: bool = True


def spawn(grid, rng):
    r, c = rng.choice(empty_cells(grid))
    return Creature(row=r, col=c)
