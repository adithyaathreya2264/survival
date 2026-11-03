import math
import random

HIDDEN = 24
N_ACTIONS = 2
N_STATE = 4

GAMMA = 0.99
LR = 5e-3
EPSILON_START = 1.0
EPSILON_FLOOR = 0.02
EPSILON_DECAY_FRACTION = 0.5
REPLAY_CAPACITY = 20_000
REPLAY_WARMUP = 1_000
BATCH_SIZE = 32
TRAIN_EVERY = 2
TARGET_SYNC_STEPS = 500
HUBER_DELTA = 1.0
GRAD_CLIP = 5.0
MAX_STEPS = 500
TOTAL_ENV_STEPS = 300_000

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # half the pole's length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02
THETA_THRESHOLD = 12 * 2 * math.pi / 360
X_THRESHOLD = 2.4


class CartPole:

    def reset(self, rng):
        self.state = [rng.uniform(-0.05, 0.05) for _ in range(4)]
        self.steps = 0
        return list(self.state)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = FORCE_MAG if action == 1 else -FORCE_MAG
        costheta, sintheta = math.cos(theta), math.sin(theta)

        temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
        thetaacc = (GRAVITY * sintheta - costheta * temp) / (
            LENGTH * (4.0 / 3.0 - MASSPOLE * costheta ** 2 / TOTAL_MASS)
        )
        xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

        x += TAU * x_dot
        x_dot += TAU * xacc
        theta += TAU * theta_dot
        theta_dot += TAU * thetaacc
        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1

        done = (
            x < -X_THRESHOLD or x > X_THRESHOLD
            or theta < -THETA_THRESHOLD or theta > THETA_THRESHOLD
            or self.steps >= MAX_STEPS
        )
        return list(self.state), 1.0, done, {}



def new_net(rng):
    w1_range = 1.0 / math.sqrt(N_STATE)
    w2_range = 1.0 / math.sqrt(HIDDEN)
    return {
        "W1": [[rng.uniform(-w1_range, w1_range) for _ in range(HIDDEN)]
               for _ in range(N_STATE)],
        "b1": [0.0] * HIDDEN,
        "W2": [[rng.uniform(-w2_range, w2_range) for _ in range(N_ACTIONS)]
               for _ in range(HIDDEN)],
        "b2": [0.0] * N_ACTIONS,
    }


def clone(net):
    return {
        "W1": [row[:] for row in net["W1"]],
        "b1": net["b1"][:],
        "W2": [row[:] for row in net["W2"]],
        "b2": net["b2"][:],
    }


def forward(net, x):
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], net["b2"]
    h = b1[:]
    for i in range(N_STATE):
        xi, row = x[i], W1[i]
        for j in range(HIDDEN):
            h[j] += xi * row[j]
    a = [math.tanh(v) for v in h]
    q = b2[:]
    for j in range(HIDDEN):
        row, av = W2[j], a[j]
        for k in range(N_ACTIONS):
            q[k] += av * row[k]
    return a, q


def huber(e, delta=HUBER_DELTA):
    if abs(e) <= delta:
        return 0.5 * e * e, e
    return delta * (abs(e) - 0.5 * delta), delta * (1.0 if e > 0 else -1.0)


def zero_grads():
    return {
        "W1": [[0.0] * HIDDEN for _ in range(N_STATE)],
        "b1": [0.0] * HIDDEN,
        "W2": [[0.0] * N_ACTIONS for _ in range(HIDDEN)],
        "b2": [0.0] * N_ACTIONS,
    }


def accumulate(net, grads, x, action, target):
    a, q = forward(net, x)
    loss, dq_taken = huber(q[action] - target)

    W2 = net["W2"]
    gW1, gb1, gW2, gb2 = grads["W1"], grads["b1"], grads["W2"], grads["b2"]
    gb2[action] += dq_taken

    dh = [0.0] * HIDDEN
    for j in range(HIDDEN):
        gW2[j][action] += a[j] * dq_taken
        da = W2[j][action] * dq_taken
        dh[j] = da * (1.0 - a[j] * a[j])
        gb1[j] += dh[j]

    for i in range(N_STATE):
        xi, row = x[i], gW1[i]
        for j in range(HIDDEN):
            row[j] += xi * dh[j]
    return loss


def apply_grads(net, grads, lr, scale=1.0, clip=None):
    if clip is not None:
        total = sum(v * v for v in grads["b2"])
        total += sum(v * v for row in grads["W2"] for v in row)
        total += sum(v * v for v in grads["b1"])
        total += sum(v * v for row in grads["W1"] for v in row)
        norm = math.sqrt(total) * scale
        if norm > clip:
            scale *= clip / norm

    step = lr * scale
    b2, W2, b1, W1 = net["b2"], net["W2"], net["b1"], net["W1"]
    for k in range(N_ACTIONS):
        b2[k] -= step * grads["b2"][k]
    for j in range(HIDDEN):
        gr, wr = grads["W2"][j], W2[j]
        for k in range(N_ACTIONS):
            wr[k] -= step * gr[k]
        b1[j] -= step * grads["b1"][j]
    for i in range(N_STATE):
        gr, wr = grads["W1"][i], W1[i]
        for j in range(HIDDEN):
            wr[j] -= step * gr[j]


def greedy_action(net, x):
    _a, q = forward(net, x)
    return 0 if q[0] >= q[1] else 1


def evaluate_greedy(net, n_episodes=20, seed=999):
    env = CartPole()
    rng = random.Random(seed)
    lengths = []
    for _ in range(n_episodes):
        x = env.reset(rng)
        done, steps = False, 0
        while not done:
            action = greedy_action(net, x)
            x, _r, done, _info = env.step(action)
            steps += 1
        lengths.append(steps)
    return sum(lengths) / len(lengths)


def train(seed=1, total_steps=TOTAL_ENV_STEPS, lr=LR, verbose=True):
    rng = random.Random(seed)
    env = CartPole()
    net = new_net(rng)
    target = clone(net)
    buffer = []
    buf_pos = 0

    decay_steps = max(1, int(total_steps * EPSILON_DECAY_FRACTION))
    x = env.reset(rng)

    for t in range(total_steps):
        eps = max(EPSILON_FLOOR,
                  EPSILON_START + (EPSILON_FLOOR - EPSILON_START) * t / decay_steps)
        if rng.random() < eps:
            action = rng.randrange(N_ACTIONS)
        else:
            action = greedy_action(net, x)

        nx, reward, done, _info = env.step(action)
        transition = (x, action, reward, nx, done)
        if len(buffer) < REPLAY_CAPACITY:
            buffer.append(transition)
        else:
            buffer[buf_pos] = transition
            buf_pos = (buf_pos + 1) % REPLAY_CAPACITY

        x = env.reset(rng) if done else nx

        if len(buffer) >= REPLAY_WARMUP and t % TRAIN_EVERY == 0:
            batch = [buffer[rng.randrange(len(buffer))] for _ in range(BATCH_SIZE)]
            grads = zero_grads()
            for (s, a, r, s2, d) in batch:
                if d:
                    y = r
                else:
                    _h, q2 = forward(target, s2)
                    y = r + GAMMA * max(q2)
                accumulate(net, grads, s, a, y)
            apply_grads(net, grads, lr, scale=1.0 / BATCH_SIZE, clip=GRAD_CLIP)

        if t % TARGET_SYNC_STEPS == 0:
            target = clone(net)

        if verbose and (t + 1) % 25_000 == 0:
            ev = evaluate_greedy(net, n_episodes=20)
            print(f"  t={t+1:>7} eps={eps:.2f} greedy_mean_steps={ev:6.1f} / {MAX_STEPS}")

    return net


if __name__ == "__main__":
    print(f"=== CartPole sanity check: hand-rolled DQN training loop, "
          f"{TOTAL_ENV_STEPS} steps ===")
    print("(random-policy baseline is ~20 steps; a correct training loop "
          f"should climb toward the {MAX_STEPS}-step cap)")
    net = train(seed=1)
    final = evaluate_greedy(net, n_episodes=100)
    print(f"\nfinal greedy mean episode length (100 eval episodes): {final:.1f} / {MAX_STEPS}")
