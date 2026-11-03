import json
import statistics as st
import time

import architectures
import ga
from analyze_champion import profile
from train import train as train_ga
import dqn_numpy as dqnp
import train_dqn_numpy

SEEDS = [1, 2, 3]


def mutation_sigma_sweep():
    original_sigma = ga.MUTATION_SIGMA
    values = [0.1, 0.2, 0.4]
    results = {}
    try:
        for sigma in values:
            ga.MUTATION_SIGMA = sigma
            rows = []
            for seed in SEEDS:
                _hist, best = train_ga(arch_name="linear", seed=seed, verbose=False)
                policy = architectures.get("linear").make_policy(best[1])
                p = profile(policy, n=1000)
                rows.append(p["mean_fitness"])
            results[sigma] = rows
            print(f"MUTATION_SIGMA={sigma}: " +
                  "  ".join(f"seed{s}={f:.1f}" for s, f in zip(SEEDS, rows)),
                  flush=True)
    finally:
        ga.MUTATION_SIGMA = original_sigma
    return results


def grad_clip_sweep():
    values = [0.5, 1.0, 2.0]
    results = {}
    for clip in values:
        rows = []
        for seed in SEEDS:
            net = train_dqn_numpy.train(seed=seed, lr=1e-2, grad_clip=clip)
            genome = dqnp.to_genome(net)
            p = profile(architectures.get("mlp").make_policy(genome), n=1000)
            rows.append(p["mean_fitness"])
        results[clip] = rows
        print(f"GRAD_CLIP={clip}: " +
              "  ".join(f"seed{s}={f:.1f}" for s, f in zip(SEEDS, rows)), flush=True)
    return results


def target_sync_sweep():
    values = [500, 1000, 2000]
    results = {}
    for sync in values:
        rows = []
        for seed in SEEDS:
            net = train_dqn_numpy.train(seed=seed, lr=1e-2, target_sync_steps=sync)
            genome = dqnp.to_genome(net)
            p = profile(architectures.get("mlp").make_policy(genome), n=1000)
            rows.append(p["mean_fitness"])
        results[sync] = rows
        print(f"TARGET_SYNC_STEPS={sync}: " +
              "  ".join(f"seed{s}={f:.1f}" for s, f in zip(SEEDS, rows)), flush=True)
    return results


if __name__ == "__main__":
    t0 = time.time()
    print("=== MUTATION_SIGMA (GA linear, 3 seeds) ===")
    sigma_results = mutation_sigma_sweep()

    print("\n=== GRAD_CLIP (DQN numpy, 3 seeds, lr=1e-2, 500k steps) ===")
    clip_results = grad_clip_sweep()

    print("\n=== TARGET_SYNC_STEPS (DQN numpy, 3 seeds, lr=1e-2, 500k steps) ===")
    sync_results = target_sync_sweep()

    print("\n=== SUMMARY ===")
    for label, results in (("MUTATION_SIGMA", sigma_results),
                            ("GRAD_CLIP", clip_results),
                            ("TARGET_SYNC_STEPS", sync_results)):
        for value, fits in results.items():
            print(f"{label}={value}: mean={st.mean(fits):.1f} "
                  f"stdev={st.stdev(fits) if len(fits) > 1 else 0.0:.1f}")

    json.dump(
        {"mutation_sigma": sigma_results, "grad_clip": clip_results,
         "target_sync_steps": sync_results},
        open("data/config_ablation_results.json", "w"), indent=2,
    )
    print(f"\nwrote data/config_ablation_results.json  ({time.time()-t0:.0f}s total)")
