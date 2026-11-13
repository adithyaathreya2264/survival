"""Wraps verify_stage2.py's checks as pytest tests."""
import pytest

from verify_stage2 import (
    check_genome_length,
    check_saturation,
    check_nonlinearity_engaged,
    check_encoding_shared,
    gen1_degeneracy,
)


def test_genome_length():
    check_genome_length()


def test_saturation():
    check_saturation(n=100)  # smaller than the 300 default -- fast in CI


def test_nonlinearity_engaged():
    check_nonlinearity_engaged()


def test_encoding_shared():
    check_encoding_shared()


@pytest.mark.parametrize("arch_name", ["linear", "mlp"])
def test_gen1_degeneracy_runs(arch_name):
    # gen1_degeneracy only prints a descriptive summary (no assertions of its
    # own) -- wrapped here as a smoke test that the arch/policy plumbing runs
    # end to end without raising, for both architectures.
    gen1_degeneracy(arch_name, n=50)
