import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

from config import GRID_SIZE
import env as envmod
import architectures
import paths

OUT = "data/plots"
os.makedirs(OUT, exist_ok=True)

# display codes: 0 empty, 1 food, 2 hazard, 3 creature
CMAP = ListedColormap(["white", "#2ca02c", "#d62728", "#1f77b4"])


def load_policy(tag):
    d = json.load(open(paths.champion(tag)))
    return architectures.get(d.get("arch", "linear")).make_policy(d["genome"])


def trace_episode(policy, rng=None, spawn_at=None):
    e = envmod.GridEnv(envmod.return_matched(), rng or random.Random())
    e.reset(rng or e.rng, spawn_at=spawn_at)
    frames = []

    def snapshot():
        g = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                v = e.grid[r][c]
                g[r][c] = 1 if v == 1 else (2 if v == -1 else 0)
        g[e.creature.row][e.creature.col] = 3
        return g

    frames.append((snapshot(), None))
    while True:
        obs = e.observe()
        action = policy(obs)
        _o, _r, done, info = e.step(action)
        frames.append((snapshot(), action))
        if done:
            break
    return frames, info


def trim_boring_tail(frames, keep_start=15, keep_end=5, max_len=40):
    if len(frames) <= max_len:
        return frames, None
    marker_at = keep_start
    return frames[:keep_start] + frames[-keep_end:], marker_at


def render_gif(frames, path, title, marker_at=None, fps=5):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow([f[0] for f in frames][0], cmap=CMAP, vmin=0, vmax=3)
    ax.set_xticks([]); ax.set_yticks([])
    txt = ax.set_title(title, fontsize=10)

    def update(i):
        grid, action = frames[i]
        im.set_data(grid)
        label = f"{title}\nstep {i}"
        if marker_at is not None and i == marker_at:
            label += "   [... skipping repeated steps ...]"
        txt.set_text(label)
        return im, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"wrote {path}  ({len(frames)} frames)")


def render_champion(tag, label, spawn_at=None, seed=1, out_name=None, trim=True):
    policy = load_policy(tag)
    frames, info = trace_episode(policy, random.Random(seed), spawn_at=spawn_at)
    if trim:
        trimmed, marker = trim_boring_tail(frames)
    else:
        trimmed, marker = frames, None
    fname = out_name or f"replay_{tag}.gif"
    render_gif(trimmed, f"{OUT}/{fname}",
              f"{label}  (steps={info['steps']} food={info['food']} "
              f"cause={info['cause']})", marker_at=marker)
    return info


def render_side_by_side(tag_a, label_a, tag_b, label_b, seed_a=7, seed_b=3,
                        spawn_a=None, spawn_b=None, out_name="replay_comparison.gif",
                        fps=5):
    pa, pb = load_policy(tag_a), load_policy(tag_b)
    frames_a, info_a = trace_episode(pa, random.Random(seed_a), spawn_at=spawn_a)
    frames_b, info_b = trace_episode(pb, random.Random(seed_b), spawn_at=spawn_b)
    n = max(len(frames_a), len(frames_b))
    frames_a += [frames_a[-1]] * (n - len(frames_a))
    frames_b += [frames_b[-1]] * (n - len(frames_b))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 5.2))
    ima = axa.imshow(frames_a[0][0], cmap=CMAP, vmin=0, vmax=3)
    imb = axb.imshow(frames_b[0][0], cmap=CMAP, vmin=0, vmax=3)
    for ax in (axa, axb):
        ax.set_xticks([]); ax.set_yticks([])
    ta = axa.set_title("", fontsize=9)
    tb = axb.set_title("", fontsize=9)
    fig.suptitle(f"{label_a}  vs  {label_b}", fontsize=11)

    def update(i):
        ima.set_data(frames_a[i][0])
        imb.set_data(frames_b[i][0])
        ta.set_text(f"step {min(i, info_a['steps'])} / {info_a['steps']}  "
                    f"food={info_a['food']}")
        tb.set_text(f"step {min(i, info_b['steps'])} / {info_b['steps']}  "
                    f"food={info_b['food']}")
        return ima, imb, ta, tb

    ani = animation.FuncAnimation(fig, update, frames=n, interval=200)
    fig.tight_layout()
    ani.save(f"{OUT}/{out_name}", writer="pillow", fps=fps)
    plt.close(fig)
    print(f"wrote {OUT}/{out_name}  ({n} frames, capped, "
          f"a={info_a['steps']}st/{info_a['food']}f  b={info_b['steps']}st/{info_b['food']}f)")


if __name__ == "__main__":
    print("=== GA seed 9 (Basin C) -- successful foraging (untrimmed, full run) ===")
    render_champion("linear_mr100_s9", "GA Basin C forager", seed=7,
                    out_name="replay_ga_basinC_forage.gif", trim=False)

    print("\n=== GA seed 1 (Basin C) -- failure at a problem tile ===")
    render_champion("linear_mr100_s1", "GA seed1 -- problem-tile death",
                    spawn_at=(29, 26), out_name="replay_ga_seed1_death.gif")

    print("\n=== DQN untuned (lr=1e-3) -- jamming ===")
    render_champion("dqn_return_matched_s1", "DQN untuned (lr=1e-3) -- jams",
                    seed=3, out_name="replay_dqn_untuned_jam.gif")

    print("\n=== DQN tuned (lr=1e-2) -- typical episode ===")
    render_champion("dqn_lr2_s1", "DQN tuned (lr=1e-2)", seed=3,
                    out_name="replay_dqn_tuned.gif")

    print("\n=== SIDE BY SIDE: GA Basin C forager vs DQN untuned jamming ===")
    render_side_by_side("linear_mr100_s9", "GA Basin C (forages)",
                        "dqn_return_matched_s1", "DQN lr=1e-3 (jams)",
                        seed_a=7, seed_b=3)
