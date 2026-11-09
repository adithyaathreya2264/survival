import random

from actions import N_ACTIONS

_rng = random.Random()


def random_policy(obs):
    return _rng.randrange(N_ACTIONS)
