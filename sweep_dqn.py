import json, time
import train_dqn, dqn, architectures
from analyze_champion import profile

LRS = [1e-4, 1e-3, 1e-2]
GAMMAS = [0.90, 0.99, 0.999]
STEPS = 300_000

rows = []
print(f"{'lr':>8}{'gamma':>8}{'fitness':>10}{'food':>7}{'cells':>8}{'hazard':>8}{'jam':>7}{'sec':>6}", flush=True)
for lr in LRS:
    for g in GAMMAS:
        t0 = time.time()
        net, _h = train_dqn.train(seed=1, reward_name="return_matched",
                                  total_steps=STEPS, lr=lr, gamma=g,
                                  log_every=STEPS + 1, verbose=False)
        p = profile(architectures.get("mlp").make_policy(dqn.to_genome(net)), n=1000)
        rows.append({"lr": lr, "gamma": g, **{k: p[k] for k in
                    ("mean_fitness","mean_food","mean_cells","hz_all","jam_rate")}})
        print(f"{lr:>8.0e}{g:>8.3f}{p['mean_fitness']:>10.1f}{p['mean_food']:>7.2f}"
              f"{p['mean_cells']:>8.1f}{p['hz_all']:>8.1%}{p['jam_rate']:>7.1%}"
              f"{time.time()-t0:>6.0f}", flush=True)
json.dump(rows, open("data/dqn_sweep.json","w"), indent=2)
best = max(rows, key=lambda r: r["mean_fitness"])
print(f"\nbest: lr={best['lr']:.0e} gamma={best['gamma']} -> fitness={best['mean_fitness']:.1f} food={best['mean_food']:.2f}")
print("GA basin A=175 (food 1.16) | basin C=758 (food 9.41) | random=99 (food 0.60)")
