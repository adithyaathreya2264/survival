import json
import statistics as st

from analyze_champion import profile
from policies import random_policy
import architectures
import env as envmod
import paths

PROBLEM_TILES = [(29, 26), (15, 20), (24, 20)]  # GA seed-1 champion died here


def load(tag):
    d = json.load(open(paths.champion(tag)))
    return architectures.get(d.get("arch", "linear")).make_policy(d["genome"])


def problem_tile_test(policy, trials=1):
    out = []
    for tile in PROBLEM_TILES:
        e = envmod.GridEnv(envmod.return_matched())
        e.reset(spawn_at=tile)
        while True:
            obs = e.observe()
            _o, _r, done, info = e.step(policy(obs))
            if done:
                break
        out.append((tile, info["cause"], info["steps"], info["fitness"]))
    return out


if __name__ == "__main__":
    ga = json.load(open("data/linear_20seeds.json"))
    dqn_res = json.load(open("data/dqn_results.json"))   # lr=1e-3, RETRACTED
    lr2 = json.load(open("data/dqn_lr2_results.json"))   # lr=1e-2, the real result

    ga_means = sorted(r["mean_fitness"] for r in ga)
    A = [m for m in ga_means if m < 230]
    B = [m for m in ga_means if 230 <= m < 450]
    C = [m for m in ga_means if m >= 450]

    print("=" * 74)
    print("CROSS-STAGE COMPARISON  (all profiled on the same 2000 spawns)")
    print("=" * 74)
    print(f"{'method':<34}{'fitness':>18}{'food':>8}{'cells':>8}")
    print("-" * 74)

    rp = profile(random_policy, n=2000)
    print(f"{'Stage 0  random policy':<34}{rp['mean_fitness']:>10.1f}"
          f"{'':>8}{rp['mean_food']:>8.2f}{rp['mean_cells']:>8.1f}")

    for label, grp in (("A wander/starve", A), ("B partial forage", B),
                       ("C full forage", C)):
        sub = [r for r in ga if (r["mean_fitness"] in grp)]
        f = st.mean(r["mean_food"] for r in sub)
        c = st.mean(r["mean_cells"] for r in sub)
        print(f"{'Stage 1  GA linear, basin ' + label[0]:<34}"
              f"{st.mean(grp):>10.1f} +/-{st.pstdev(grp):>5.0f}"
              f"{f:>8.2f}{c:>8.1f}   ({len(grp)}/20 seeds)")

    def dqn_row(label, rows, note):
        m = [r["mean_fitness"] for r in rows]
        print(f"{label:<34}{st.mean(m):>10.1f} +/-{st.stdev(m):>5.1f}"
              f"{st.mean(r['mean_food'] for r in rows):>8.2f}"
              f"{st.mean(r['mean_cells'] for r in rows):>8.1f}   {note}")

    dqn_row("Stage 3  DQN lr=1e-2 (TUNED)", lr2, "(5 seeds)")
    print("-" * 74)
    print("  the two rows below are an UNDER-TUNED optimiser, not a property of DQN:")
    for arm in ("return_matched", "sparse"):
        dqn_row(f"  [lr=1e-3 artifact] {arm}", dqn_res[arm], "(5 seeds)")

    ga_median = st.median(ga_means)
    lr2_mean = st.mean(r["mean_fitness"] for r in lr2)
    print(f"\nTuned DQN {lr2_mean:.1f} vs GA 20-seed median {ga_median:.1f} -- a dead heat "
          f"on the median.\nGA's advantage is entirely its TAIL: basin C ({st.mean(C):.0f}) "
          f"in {len(C)}/20 seeds;\nbest of 5 tuned-DQN seeds was "
          f"{max(r['mean_fitness'] for r in lr2):.1f}.")

    print("\n" + "=" * 74)
    print("PROBLEM-TILE TEST -- spawns where GA seed 1 died at step 1")
    print("=" * 74)
    best_dqn = max(lr2, key=lambda r: r["mean_fitness"])
    cands = [("GA linear seed 1 (Basin C)", load("linear_mr100_s1")),
             ("GA linear seed 9 (Basin C)", load("linear_mr100_s9")),
             (f"DQN lr=1e-2 seed {best_dqn['seed']} (tuned)",
              load(f"dqn_lr2_s{best_dqn['seed']}"))]
    for name, pol in cands:
        res = problem_tile_test(pol)
        survived = sum(1 for _t, cause, _s, _f in res if cause != "hazard")
        detail = "  ".join(f"{t}:{cause}({s}st)" for t, cause, s, _f in res)
        print(f"{name:<30} survived {survived}/3   {detail}")
