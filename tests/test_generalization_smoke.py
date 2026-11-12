"""Regression guard for Phase 1's generalization finding (ROADMAP.md): GA
basin C's headline fitness collapses on unseen maps while DQN degrades far
less. Uses a small episode count (fast enough for CI) -- this checks the
code path and direction of the effect, not the precise numbers from the
full 500-episode/10-map sweep (see verify_generalization.py for that)."""
from config import MAP_SEED
from verify_generalization import (
    GA_CHAMPION, DQN_CHAMPION, load_policy, profile_on_map,
)
from world import build_static_map

N_SMOKE_EPISODES = 100
UNSEEN_SEED = 999


def test_ga_basin_c_degrades_hard_on_an_unseen_map():
    policy = load_policy(GA_CHAMPION)
    train_map = build_static_map(MAP_SEED)
    unseen_map = build_static_map(UNSEEN_SEED)

    on_train = profile_on_map(policy, train_map, n=N_SMOKE_EPISODES)
    on_unseen = profile_on_map(policy, unseen_map, n=N_SMOKE_EPISODES)

    assert on_train["mean_fitness"] > 400, (
        "GA basin C should score well above random on its training map"
    )
    assert on_unseen["hazard_rate"] > on_train["hazard_rate"] + 0.3, (
        "expected hazard rate to jump sharply on an unseen map "
        f"(train={on_train['hazard_rate']:.1%}, unseen={on_unseen['hazard_rate']:.1%})"
    )


def test_dqn_degrades_less_than_ga_on_an_unseen_map():
    ga_policy = load_policy(GA_CHAMPION)
    dqn_policy = load_policy(DQN_CHAMPION)
    unseen_map = build_static_map(UNSEEN_SEED)

    ga_unseen = profile_on_map(ga_policy, unseen_map, n=N_SMOKE_EPISODES)
    dqn_unseen = profile_on_map(dqn_policy, unseen_map, n=N_SMOKE_EPISODES)

    assert dqn_unseen["mean_fitness"] > ga_unseen["mean_fitness"], (
        "DQN is expected to outperform GA basin C on an unseen map "
        f"(dqn={dqn_unseen['mean_fitness']:.1f}, ga={ga_unseen['mean_fitness']:.1f})"
    )
