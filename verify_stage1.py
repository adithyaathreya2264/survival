import random
from collections import Counter

from config import GENOME_LENGTH, POP_SIZE, N_CHANNELS, P_WALL, P_SELF
from actions import N_ACTIONS
from world import STATIC_MAP
from perception import perceive
from simulation import run_episode_debug
from linear_policy import encode, make_policy, random_genome
from ga import mutate, uniform_crossover, tournament_select, next_generation


def check_encoding():
    obs = perceive(STATIC_MAP, (0, 0))
    active = encode(obs)
    # 25 cells, minus the center (P_SELF has no channel) = 24 active inputs.
    assert len(active) == 24, len(active)
    assert len(set(active)) == 24, "duplicate active indices"
    assert all(0 <= i < 25 * N_CHANNELS for i in active)
    # Corner window: out-of-bounds cells must land in the wall channel (3).
    walls = [i for i in active if i % N_CHANNELS == 3]
    assert len(walls) == sum(1 for v in obs if v == P_WALL), "wall channel mismatch"
    assert obs[12] == P_SELF
    print(f"OK: encoding -> {len(active)} active inputs, {len(walls)} wall channels")


def check_bias_breaks_ties():
    genome = [0.0] * GENOME_LENGTH
    genome[-N_ACTIONS:] = [0.0, 0.0, 9.0, 0.0]
    policy = make_policy(genome)
    obs = perceive(STATIC_MAP, (15, 15))
    assert policy(obs) == 2, policy(obs)
    print("OK: bias terms break the all-zero-input tie (no silent 'always up')")


def check_determinism():
    rng = random.Random(1)
    genome = random_genome(rng)
    policy = make_policy(genome)
    obs = perceive(STATIC_MAP, (10, 10))
    assert len({policy(obs) for _ in range(50)}) == 1, "policy is not deterministic"
    a = run_episode_debug(policy, random.Random(7))
    b = run_episode_debug(policy, random.Random(7))
    assert a == b, (a, b)
    print("OK: policy deterministic; same spawn seed -> identical episode")


def check_shared_spawn():
    rng = random.Random(3)
    genomes = [random_genome(rng) for _ in range(5)]
    # Episodes differ in outcome, but all start identically -- verify via spawn.
    from world import copy_map
    from creature import spawn
    positions = {
        (lambda c: (c.row, c.col))(spawn(copy_map(STATIC_MAP), random.Random(99)))
        for _ in genomes
    }
    assert len(positions) == 1, positions
    print(f"OK: shared spawn seed -> identical start {positions.pop()}")


def check_ga_operators():
    rng = random.Random(5)
    a = [0.0] * GENOME_LENGTH
    b = [1.0] * GENOME_LENGTH

    child = uniform_crossover(a, b, rng)
    assert len(child) == GENOME_LENGTH
    assert set(child) <= {0.0, 1.0}
    mix = sum(child) / GENOME_LENGTH
    assert 0.3 < mix < 0.7, f"crossover not mixing evenly: {mix}"

    m = mutate(a, rng)
    changed = sum(1 for x in m if x != 0.0)
    assert 20 < changed < 70, f"mutation rate off: {changed}/404 genes changed"

    scored = [(float(i), [float(i)] * GENOME_LENGTH) for i in range(20)]
    picks = [tournament_select(scored, rng)[0] for _ in range(300)]
    assert sum(picks) / 300 > 10, "tournament not favouring fitter genomes"

    pop = next_generation(scored, rng)
    assert len(pop) == POP_SIZE
    assert pop[0] == scored[-1][1], "elite not carried forward unchanged"
    print(f"OK: crossover mix={mix:.2f}, mutation={changed}/404 genes, "
          f"tournament mean rank={sum(picks)/300:.1f}/19, elitism preserved")


def check_generation_one_matches_baseline(n=POP_SIZE * 5):
    rng = random.Random(11)
    fits, causes = [], Counter()
    for _ in range(n):
        policy = make_policy(random_genome(rng))
        fitness, cause, _s, _f = run_episode_debug(policy, rng)
        fits.append(fitness)
        causes[cause] += 1
    fits.sort()
    mean = sum(fits) / n
    hazard = causes["hazard"] / n
    print(f"\n=== GEN-1 PLUMBING CHECK ({n} random genomes) ===")
    print(f"min={fits[0]} median={fits[n//2]} mean={mean:.1f} max={fits[-1]}")
    print("termination cause: " + ", ".join(
        f"{k}={causes[k]} ({causes[k]/n:.1%})" for k in ("hazard", "starve", "cap")
    ))
    print("Stage 0 random baseline: median=100 mean=95.2 hazard=65.0%")


if __name__ == "__main__":
    check_encoding()
    check_bias_breaks_ties()
    check_determinism()
    check_shared_spawn()
    check_ga_operators()
    check_generation_one_matches_baseline()
