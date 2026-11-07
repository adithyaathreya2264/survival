from actions import ACTIONS, N_ACTIONS
from config import VIEW_RADIUS
from world import FOOD, HAZARD

_SIDE = 2 * VIEW_RADIUS + 1
_DEFAULT_ORDER = [3, 1, 0, 2]  # right, down, up, left -- arbitrary but fixed


def _cell(obs, dr, dc):
    r, c = VIEW_RADIUS + dr, VIEW_RADIUS + dc
    return obs[r * _SIDE + c]


def _manhattan(dr, dc):
    return abs(dr) + abs(dc)


def make_policy(genome=None):  # genome accepted for interface parity, unused

    def policy(obs):
        food, hazard = [], []
        for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                if dr == 0 and dc == 0:
                    continue
                val = _cell(obs, dr, dc)
                if val == FOOD:
                    food.append((dr, dc))
                elif val == HAZARD:
                    hazard.append((dr, dc))

        def is_safe(a):
            dr, dc = ACTIONS[a]
            return _cell(obs, dr, dc) != HAZARD

        if food:
            tr, tc = min(food, key=lambda p: _manhattan(*p))
            ranked = sorted(
                range(N_ACTIONS),
                key=lambda a: _manhattan(tr - ACTIONS[a][0], tc - ACTIONS[a][1]),
            )
        elif hazard:
            hr, hc = min(hazard, key=lambda p: _manhattan(*p))
            ranked = sorted(
                range(N_ACTIONS),
                key=lambda a: -_manhattan(hr - ACTIONS[a][0], hc - ACTIONS[a][1]),
            )
        else:
            ranked = _DEFAULT_ORDER

        for a in ranked:
            if is_safe(a):
                return a
        return ranked[0]

    return policy
