import random
from collections import Counter

from config import HIDDEN_SIZE, MLP_GENOME_LENGTH, N_INPUTS
from actions import N_ACTIONS
from world import STATIC_MAP, copy_map
from creature import spawn
from perception import perceive
from simulation import step, run_episode_debug
import mlp_policy
import linear_policy
import architectures


def check_genome_length():
    expected = N_INPUTS * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * N_ACTIONS + N_ACTIONS
    assert expected == 1684, expected
    assert MLP_GENOME_LENGTH == 1684, MLP_GENOME_LENGTH
    g = mlp_policy.random_genome(random.Random(0))
    assert len(g) == 1684, len(g)
    print(f"OK: genome length = {N_INPUTS}*{HIDDEN_SIZE} + {HIDDEN_SIZE} + "
          f"{HIDDEN_SIZE}*{N_ACTIONS} + {N_ACTIONS} = 1684")


def check_saturation(n=300):
    rng = random.Random(1)
    obs_samples = [perceive(STATIC_MAP, (r, c))
                   for r, c in [(15, 15), (0, 0), (29, 29), (5, 20), (14, 4)]]
    for label, w1_range in (("scaled (+/-0.4)", 0.4), ("unscaled (+/-1.0)", 1.0)):
        sat = total = 0
        pre_sq = 0.0
        for _ in range(n):
            genome = [rng.uniform(-w1_range, w1_range) for _ in range(N_INPUTS * HIDDEN_SIZE)]
            genome += [rng.uniform(-1, 1) for _ in range(MLP_GENOME_LENGTH - len(genome))]
            for obs in obs_samples:
                hidden, _ = mlp_policy.forward(genome, obs)
                raw, _ = mlp_policy.forward(genome, obs, activation=lambda x: x)
                sat += sum(1 for h in hidden if abs(h) > 0.99)
                pre_sq += sum(x * x for x in raw)
                total += len(hidden)
        frac = sat / total
        print(f"  {label:20s} saturated |h|>0.99: {frac:6.1%}   "
              f"pre-activation std ~{(pre_sq/total) ** 0.5:.2f}")
        if w1_range == 0.4:
            assert frac < 0.5, f"layer-1 init scaling failed: {frac:.1%} saturated"
    print("OK: scaled init keeps hidden units in tanh's responsive band")


def check_nonlinearity_engaged():
    rng = random.Random(2)
    positions = [(r, c) for r in range(2, 28, 3) for c in range(2, 28, 3)]
    disagreements = 0
    for _ in range(20):
        genome = mlp_policy.random_genome(rng)
        tanh_pol = mlp_policy.make_policy(genome)
        lin_pol = mlp_policy.make_policy(genome, activation=lambda x: x)
        for pos in positions:
            obs = perceive(STATIC_MAP, pos)
            if tanh_pol(obs) != lin_pol(obs):
                disagreements += 1
    total = 20 * len(positions)
    assert disagreements > 0, "tanh and identity never differ -- nonlinearity inert"
    print(f"OK: tanh vs identity differ on {disagreements}/{total} "
          f"({disagreements/total:.1%}) observations -- nonlinearity is engaged")


def check_encoding_shared():
    obs = perceive(STATIC_MAP, (7, 9))
    assert mlp_policy.encode is linear_policy.encode
    assert len(linear_policy.encode(obs)) == 24
    print("OK: MLP reuses the linear one-hot encoding (not a fork)")


def gen1_degeneracy(arch_name, n=200):
    arch = architectures.get(arch_name)
    rng = random.Random(11)
    actions_used, jam, cells_all, causes = Counter(), 0, [], Counter()
    for _ in range(n):
        pol = arch.make_policy(arch.random_genome(rng))
        grid = copy_map(STATIC_MAP)
        c = spawn(grid, rng)
        acts, seen = set(), set()
        while True:
            a = pol(perceive(grid, (c.row, c.col)))
            acts.add(a)
            seen.add((c.row, c.col))
            if step(grid, c, pol):
                break
        actions_used[len(acts)] += 1
        cells_all.append(len(seen))
        if len(seen) <= 3:
            jam += 1
        causes["hazard" if not c.alive else "starve"] += 1
    print(f"  {arch_name:7s} distinct actions/episode {dict(sorted(actions_used.items()))}"
          f"  one-action={actions_used[1]}/{n}  jammed(<=3 cells)={jam}/{n}"
          f"  mean cells={sum(cells_all)/n:.1f}  hazard={causes['hazard']/n:.1%}")


if __name__ == "__main__":
    check_genome_length()
    check_encoding_shared()
    print("\n=== SATURATION (layer-1 init) ===")
    check_saturation()
    print()
    check_nonlinearity_engaged()
    print("\n=== GEN-1 DEGENERACY (random genomes, measured not assumed) ===")
    gen1_degeneracy("linear")
    gen1_degeneracy("mlp")
