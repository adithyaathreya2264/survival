import json, time
import train_dqn

for lr, tag in ((1e-3, "untuned"), (1e-2, "tuned")):
    t0 = time.time()
    net, hist, curves = train_dqn.train(seed=1, reward_name="return_matched",
                                        total_steps=500_000, lr=lr,
                                        log_every=500_000, verbose=False,
                                        return_curves=True)
    json.dump({"lr": lr, "history": hist, **curves},
              open(f"data/dqn_curves_{tag}.json", "w"))
    print(f"{tag} (lr={lr}): {len(curves['ep_fitness'])} episodes, "
          f"{len(curves['losses'])} loss points  ({time.time()-t0:.0f}s)", flush=True)
