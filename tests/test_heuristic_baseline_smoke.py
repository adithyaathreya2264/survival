"""Regression guard for Phase 3's heuristic-baseline finding (ROADMAP.md):
the hand-coded heuristic should reliably beat random and avoid hazards, but
stay well below GA basin C's learned foraging. Small n for CI speed."""
from evaluate import profile
from heuristic_policy import make_policy as make_heuristic
from policies import random_policy

N_SMOKE_EPISODES = 200


def test_heuristic_beats_random():
    random_stats = profile(random_policy, n=N_SMOKE_EPISODES)
    heuristic_stats = profile(make_heuristic(), n=N_SMOKE_EPISODES)

    assert heuristic_stats["mean"] > random_stats["mean"], (
        f"heuristic ({heuristic_stats['mean']:.1f}) should beat "
        f"random ({random_stats['mean']:.1f})"
    )


def test_heuristic_avoids_hazards_reliably():
    stats = profile(make_heuristic(), n=N_SMOKE_EPISODES)
    hazard_rate = stats["causes"].get("hazard", 0) / stats["n"]
    assert hazard_rate < 0.05, (
        f"heuristic's local reactive hazard-avoidance should keep hazard "
        f"deaths rare, got {hazard_rate:.1%}"
    )
