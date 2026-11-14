import argparse
import csv
import json
import random
from collections import Counter

from config import GA_SEED, MUTATION_RATE, N_GENERATIONS, POP_SIZE
from simulation import run_episode_debug
from ga import next_generation, elites
from ascii_chart import line_chart
import architectures
import paths


def evaluate_population(population, spawn_seed, arch):
    scored = []
    causes = Counter()
    foods = []
    for genome in population:
        policy = arch.make_policy(genome)
        fitness, cause, _steps, food = run_episode_debug(
            policy, random.Random(spawn_seed)
        )
        scored.append((fitness, genome))
        causes[cause] += 1
        foods.append(food)
    return scored, causes, foods


def train(arch_name="linear", n_generations=N_GENERATIONS, seed=GA_SEED,
          mutation_rate=MUTATION_RATE, log_path=None, champion_path=None,
          verbose=True):
    arch = architectures.get(arch_name)
    master = random.Random(seed)
    population = [arch.random_genome(master) for _ in range(POP_SIZE)]
    assert all(len(g) == arch.genome_length for g in population)

    best_ever = (float("-inf"), None, None)  # (fitness, genome, spawn_seed)
    history = []

    for gen in range(n_generations):
        spawn_seed = master.randrange(2 ** 31)
        scored, causes, foods = evaluate_population(population, spawn_seed, arch)

        fits = sorted((f for f, _ in scored), reverse=True)
        row = {
            "generation": gen,
            "spawn_seed": spawn_seed,
            "best": fits[0],
            "mean": sum(fits) / len(fits),
            "worst": fits[-1],
            "mean_food": sum(foods) / len(foods),
            "hazard": causes["hazard"],
            "starve": causes["starve"],
            "cap": causes["cap"],
        }
        history.append(row)

        top_genome = elites(scored, 1)[0]
        if fits[0] > best_ever[0]:
            best_ever = (fits[0], list(top_genome), spawn_seed)

        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            print(
                f"gen {gen:4d}  best={row['best']:6.0f}  mean={row['mean']:6.1f}  "
                f"worst={row['worst']:5.0f}  food={row['mean_food']:4.2f}  "
                f"hazard={causes['hazard']:3d} starve={causes['starve']:3d} "
                f"cap={causes['cap']:3d}"
            )

        next_pop = next_generation(scored, master, mutation_rate=mutation_rate)

        # Elitism check: the champion genome must survive untouched. (Note the
        # per-generation BEST FITNESS is not monotonic -- the spawn changes each
        # generation, so the elite is re-scored against a new start.)
        assert next_pop[0] == top_genome, f"elite lost at generation {gen}"
        population = next_pop

    if log_path:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    if champion_path:
        with open(champion_path, "w") as f:
            json.dump(
                {
                    "arch": arch_name,
                    "fitness": best_ever[0],
                    "spawn_seed": best_ever[2],
                    "seed": seed,
                    "mutation_rate": mutation_rate,
                    "genome": best_ever[1],
                },
                f,
            )

    return history, best_ever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="linear", choices=list(architectures.ARCHITECTURES))
    p.add_argument("--seed", type=int, default=GA_SEED)
    p.add_argument("--generations", type=int, default=N_GENERATIONS)
    p.add_argument("--mutation-rate", type=float, default=MUTATION_RATE)
    p.add_argument("--champion", default=None)
    p.add_argument("--log", default=None)
    args = p.parse_args()

    champion_path = paths.champion(args.champion or args.arch)
    log_path = paths.log(args.log or args.arch)

    history, best_ever = train(
        arch_name=args.arch, n_generations=args.generations, seed=args.seed,
        mutation_rate=args.mutation_rate, log_path=log_path,
        champion_path=champion_path,
    )

    print(f"\nwrote {log_path} and {champion_path}")
    print(f"best-ever training fitness: {best_ever[0]:.0f} "
          f"(spawn_seed={best_ever[2]}) -- spawn-conditional, verify with evaluate.py")
    print()
    print(line_chart(
        {"best": [r["best"] for r in history], "mean": [r["mean"] for r in history]},
        title=f"fitness per generation ({args.arch})",
    ))


if __name__ == "__main__":
    main()
