import random

from config import HIDDEN_SIZE, N_INPUTS
from actions import N_ACTIONS
from world import STATIC_MAP
from perception import perceive
from linear_policy import encode
import dqn

EPS = 1e-5
TOL = 1e-6   # relative error threshold


def sample_loss(net, active, action, target):
    _a, q = dqn.forward(net, active)
    loss, _ = dqn.huber(q[action] - target)
    return loss


def numerical(net, container, idx, active, action, target):
    orig = container[idx]
    container[idx] = orig + EPS
    lp = sample_loss(net, active, action, target)
    container[idx] = orig - EPS
    lm = sample_loss(net, active, action, target)
    container[idx] = orig
    return (lp - lm) / (2 * EPS)


def run(n_params=40, seed=7):
    rng = random.Random(seed)
    net = dqn.new_net(rng)
    obs = perceive(STATIC_MAP, (12, 9))
    active = encode(obs)
    action = 2
    target = 0.7

    grads = dqn.zero_grads()
    dqn.accumulate(net, grads, active, action, target)

    for k in range(N_ACTIONS):
        if k != action:
            assert grads["b2"][k] == 0.0, f"b2 grad leaked to action {k}"
            for j in range(HIDDEN_SIZE):
                assert grads["W2"][j][k] == 0.0, f"W2 grad leaked to action {k}"
    print(f"OK: only taken action {action} receives gradient; other 3 are exactly 0")

    assert set(grads["W1"]) <= set(active), "dW1 nonzero on an inactive input"
    assert set(grads["W1"]) == set(active), "dW1 missing an active input"
    print(f"OK: dW1 is sparse -- exactly the {len(active)} active input rows")

    checks = []
    for _ in range(n_params):
        which = rng.choice(["W1", "b1", "W2", "b2"])
        if which == "W1":
            i = rng.choice(active)
            j = rng.randrange(HIDDEN_SIZE)
            cont, idx = net["W1"][i], j
            ana = grads["W1"][i][j]
        elif which == "b1":
            j = rng.randrange(HIDDEN_SIZE)
            cont, idx = net["b1"], j
            ana = grads["b1"][j]
        elif which == "W2":
            j = rng.randrange(HIDDEN_SIZE)
            cont, idx = net["W2"][j], action
            ana = grads["W2"][j][action]
        else:
            cont, idx = net["b2"], action
            ana = grads["b2"][action]
        num = numerical(net, cont, idx, active, action, target)
        denom = max(1e-8, abs(ana) + abs(num))
        checks.append((which, ana, num, abs(ana - num) / denom))

    worst = max(checks, key=lambda c: c[3])
    print(f"\nchecked {len(checks)} parameters across all four tensors")
    print(f"{'tensor':<6}{'analytic':>14}{'numerical':>14}{'rel.err':>12}")
    for c in checks[:6]:
        print(f"{c[0]:<6}{c[1]:>14.9f}{c[2]:>14.9f}{c[3]:>12.2e}")
    print(f"...\nWORST rel.err = {worst[3]:.3e}  (tensor {worst[0]}, "
          f"analytic {worst[1]:.9f} vs numerical {worst[2]:.9f})")
    assert worst[3] < TOL, f"GRADIENT CHECK FAILED: rel err {worst[3]:.2e}"
    print(f"\nOK: analytical backprop matches central differences (< {TOL:g})")

    grads2 = dqn.zero_grads()
    dqn.accumulate(net, grads2, active, action, target - 50.0)  # |e| >> delta
    j = rng.randrange(HIDDEN_SIZE)
    num = numerical(net, net["W2"][j], action, active, action, target - 50.0)
    ana = grads2["W2"][j][action]
    rel = abs(ana - num) / max(1e-8, abs(ana) + abs(num))
    assert rel < TOL, f"Huber linear branch gradient wrong: {rel:.2e}"
    print(f"OK: Huber linear branch (|e|>delta) gradient correct (rel.err {rel:.2e})")


if __name__ == "__main__":
    run()
