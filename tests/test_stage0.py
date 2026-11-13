"""Wraps verify_stage0.py's checks as pytest tests -- same assertions,
same functions, no new verification logic. Run standalone with
`python verify_stage0.py` for the printed narrative version."""
from verify_stage0 import check_static_map, check_perception_edges, check_random_episodes


def test_static_map():
    check_static_map()


def test_perception_edges():
    check_perception_edges()


def test_random_episodes():
    check_random_episodes(n=500)  # smaller n than the 2000 default -- fast in CI
