import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "data/plots"
os.makedirs(OUT, exist_ok=True)


def ga_fitness_curve():
    rows = list(csv.DictReader(open("data/logs/training_log.csv")))
    gens = [int(r["generation"]) for r in rows]
    best = [float(r["best"]) for r in rows]
    mean = [float(r["mean"]) for r in rows]
    worst = [float(r["worst"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, label="best", color="#2a7", linewidth=1.2)
    ax.plot(gens, mean, label="mean", color="#27a", linewidth=1.2)
    ax.plot(gens, worst, label="worst", color="#a72", linewidth=0.8, alpha=0.6)
    for g, y in ((130, 500), (150, 440)):
        ax.axvline(x=g, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(f"gen {g} collapse", (g, y), fontsize=8, ha="center",
                    color="gray", xytext=(g + 18, y), textcoords="data",
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Linear Policy -- Fitness per Generation (single run, seed=42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/ga_fitness_curve.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/ga_fitness_curve.png")


def basin_histogram():
    rows = json.load(open("data/linear_20seeds.json"))
    values = sorted(r["mean_fitness"] for r in rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=18, color="#369", edgecolor="white")
    for x, label in ((175, "Basin A\n~175 (75%)"), (282, "Basin B\n~282 (15%)"),
                     (758, "Basin C\n~758 (10%)")):
        ax.axvline(x=x, color="#a33", linestyle=":", alpha=0.6)
        ax.annotate(label, (x, ax.get_ylim()[1] * 0.85), fontsize=8,
                    ha="center", color="#a33")
    ax.set_xlabel("Mean fitness (2000-episode eval)")
    ax.set_ylabel("Count of seeds")
    ax.set_title("GA Linear Policy -- Final Fitness Distribution (20 seeds)\n"
                 "Discrete basins, not a continuous spread")
    fig.tight_layout()
    fig.savefig(f"{OUT}/basin_histogram.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/basin_histogram.png  (values: {[round(v) for v in values]})")


def cross_stage_bar_chart():
    ga = json.load(open("data/linear_20seeds.json"))
    ga_means = sorted(r["mean_fitness"] for r in ga)
    A = [m for m in ga_means if m < 230]
    B = [m for m in ga_means if 230 <= m < 450]
    C = [m for m in ga_means if m >= 450]

    def mean(xs):
        return sum(xs) / len(xs)

    def pstdev(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    def stdev(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    lr2 = json.load(open("data/dqn_lr2_results.json"))
    lr2_fit = [r["mean_fitness"] for r in lr2]
    lr3 = json.load(open("data/dqn_results.json"))
    lr3_fit = [r["mean_fitness"] for r in lr3["return_matched"]]

    labels = ["Random", "GA basin A", "GA basin B", "GA basin C",
              "DQN tuned\n(lr=1e-2)", "DQN untuned\n(lr=1e-3, retracted)"]
    means = [99.4, mean(A), mean(B), mean(C), mean(lr2_fit), mean(lr3_fit)]
    errs = [0, pstdev(A), pstdev(B), pstdev(C), stdev(lr2_fit), stdev(lr3_fit)]
    colors = ["#888", "#369", "#369", "#369", "#2a7", "#ccc"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, m in zip(bars, means):
        ax.annotate(f"{m:.0f}", (bar.get_x() + bar.get_width() / 2, m),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(y=1360, color="#a33", linestyle=":", linewidth=1, alpha=0.7)
    ax.annotate("proven ceiling: 1360", (0.1, 1360), fontsize=8, color="#a33",
                xytext=(0, 4), textcoords="offset points")
    ax.set_ylabel("Mean fitness (2000-episode eval, same spawn set)")
    ax.set_title("Cross-Stage Comparison\n"
                 "DQN ties the GA median; GA's advantage is entirely its tail")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cross_stage_comparison.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/cross_stage_comparison.png")


def _rolling_mean(xs, window):
    out = []
    s = 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= window:
            s -= xs[i - window]
        out.append(s / min(i + 1, window))
    return out


def dqn_training_curves():
    tuned = json.load(open("data/dqn_curves_tuned.json"))
    untuned = json.load(open("data/dqn_curves_untuned.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for d, color, label in ((untuned, "#c33", "untuned lr=1e-3 (retracted)"),
                            (tuned, "#2a7", "tuned lr=1e-2")):
        losses = d["losses"]
        xs = list(range(0, len(losses), 20))
        ax1.plot(xs, [losses[i] for i in xs], color=color, alpha=0.15, linewidth=0.5)
        ax1.plot(_rolling_mean(losses, 500), color=color, label=label, linewidth=1.3)
    ax1.set_xlabel("Training batch #")
    ax1.set_ylabel("Huber loss (scaled reward units)")
    ax1.set_title("DQN Training Loss")
    ax1.legend(fontsize=8)

    for d, color, label in ((untuned, "#c33", "untuned lr=1e-3 (retracted)"),
                            (tuned, "#2a7", "tuned lr=1e-2")):
        fit = d["ep_fitness"]
        ax2.plot(fit, color=color, alpha=0.12, linewidth=0.4)
        ax2.plot(_rolling_mean(fit, 100), color=color, label=label, linewidth=1.3)
    ax2.axhline(y=175, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.annotate("GA basin A (~175)", (20, 180), fontsize=8, color="gray")
    ax2.set_xlabel("Episode #")
    ax2.set_ylabel("Episode fitness (undiscounted return, GA scale)")
    ax2.set_title("DQN Reward per Episode")
    ax2.legend(fontsize=8)

    fig.suptitle("DQN Training Dynamics -- same seed, lr is the only difference")
    fig.tight_layout()
    fig.savefig(f"{OUT}/dqn_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}/dqn_training_curves.png")


if __name__ == "__main__":
    ga_fitness_curve()
    basin_histogram()
    cross_stage_bar_chart()
    dqn_training_curves()
