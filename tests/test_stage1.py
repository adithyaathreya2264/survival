"""Wraps verify_stage1.py's checks as pytest tests."""
from verify_stage1 import (
    check_encoding,
    check_bias_breaks_ties,
    check_determinism,
    check_shared_spawn,
    check_ga_operators,
    check_generation_one_matches_baseline,
)


def test_encoding():
    check_encoding()


def test_bias_breaks_ties():
    check_bias_breaks_ties()


def test_determinism():
    check_determinism()


def test_shared_spawn():
    check_shared_spawn()


def test_ga_operators():
    check_ga_operators()


def test_generation_one_matches_baseline():
    check_generation_one_matches_baseline(n=200)  # smaller than POP_SIZE*5 -- fast in CI
