from config import (
    MUTATION_RATE,
    MUTATION_SIGMA,
    N_ELITES,
    POP_SIZE,
    TOURNAMENT_K,
)


def tournament_select(scored, rng, k=TOURNAMENT_K):
    best = max((rng.choice(scored) for _ in range(k)), key=lambda p: p[0])
    return best[1]


def uniform_crossover(a, b, rng):
    return [a[i] if rng.random() < 0.5 else b[i] for i in range(len(a))]


def mutate(genome, rng, mutation_rate=MUTATION_RATE):
    return [
        g + rng.gauss(0, MUTATION_SIGMA) if rng.random() < mutation_rate else g
        for g in genome
    ]


def elites(scored, n=N_ELITES):
    ranked = sorted(scored, key=lambda p: p[0], reverse=True)
    return [genome for _, genome in ranked[:n]]


def next_generation(scored, rng, pop_size=POP_SIZE, n_elites=N_ELITES,
                    mutation_rate=MUTATION_RATE):
    population = list(elites(scored, n_elites))
    while len(population) < pop_size:
        parent_a = tournament_select(scored, rng)
        parent_b = tournament_select(scored, rng)
        child = uniform_crossover(parent_a, parent_b, rng)
        population.append(mutate(child, rng, mutation_rate))
    return population
