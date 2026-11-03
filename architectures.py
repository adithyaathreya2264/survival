from dataclasses import dataclass
from typing import Callable

import linear_policy
import mlp_policy
from config import GENOME_LENGTH, MLP_GENOME_LENGTH


@dataclass(frozen=True)
class Architecture:
    name: str
    genome_length: int
    random_genome: Callable
    make_policy: Callable


ARCHITECTURES = {
    "linear": Architecture(
        "linear", GENOME_LENGTH, linear_policy.random_genome, linear_policy.make_policy
    ),
    "mlp": Architecture(
        "mlp", MLP_GENOME_LENGTH, mlp_policy.random_genome, mlp_policy.make_policy
    ),
}

DEFAULT_ARCH = "linear"  # champion.json files predating Stage 2 have no "arch" key


def get(name):
    if name not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {name!r}; have {list(ARCHITECTURES)}")
    return ARCHITECTURES[name]
