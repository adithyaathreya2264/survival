import json, statistics as st, time
import train_dqn, dqn, architectures
from analyze_champion import profile

SEEDS = [1, 2, 3, 4, 5]
rows = []
print(f"{'seed':>5}{'fitness':>10}{'food':>7}{'cells':>8}{'hazard':>8}{'jam':>7}{'sec':>6}", flush=True)
for s in SEEDS:
    t0 = time.time()
    net, _h = train_dqn.train(seed=s, reward_name="return_matched",
                              total_steps=500_000, lr=1e-2, gamma=0.99,
                              log_every=10**9, verbose=False)
    p = profile(architectures.get("mlp").make_policy(dqn.to_genome(net)), n=2000)
    p["seed"] = s
    rows.append(p)
    json.dump({"arch": "mlp", "genome": dqn.to_genome(net)},
              open(f"data/champions/champion_dqn_lr2_s{s}.json", "w"))
    print(f"{s:>5}{p['mean_fitness']:>10.1f}{p['mean_food']:>7.2f}{p['mean_cells']:>8.1f}"
          f"{p['hz_all']:>8.1%}{p['jam_rate']:>7.1%}{time.time()-t0:>6.0f}", flush=True)

f = [r["mean_fitness"] for r in rows]
json.dump(rows, open("data/dqn_lr2_results.json", "w"), indent=2)
print(f"\nlr=1e-2 5 seeds: mean={st.mean(f):.1f} +/- {st.stdev(f):.1f}  "
      f"range={min(f):.1f}-{max(f):.1f}  food={st.mean(r['mean_food'] for r in rows):.2f}")
print("lr=1e-3 matrix  : mean=103.9 +/- 4.0  range=98.9-107.9  food=0.16")
print("GA basin A      : mean=175.0 +/-  15   food=1.16   | basin C: 757.6, food 9.41")
