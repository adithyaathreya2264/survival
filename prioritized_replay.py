import random

PRIORITY_EPSILON = 1e-3
PRIORITY_ALPHA = 0.6  # priority^alpha -- 0 = uniform, 1 = fully proportional


class _Fenwick:

    def __init__(self, n):
        self.n = n
        self.tree = [0.0] * (n + 1)

    def update(self, i, delta):
        i += 1  # external 0-indexed -> internal 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def total(self):
        return self.tree[self.n] if self.n & (self.n - 1) == 0 else self._prefix(self.n - 1)

    def _prefix(self, i):
        i += 1
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def find(self, target):
        pos = 0
        bitmask = 1
        while bitmask * 2 <= self.n:
            bitmask *= 2
        remaining = target
        bm = bitmask
        while bm > 0:
            next_pos = pos + bm
            if next_pos <= self.n and self.tree[next_pos] < remaining:
                pos = next_pos
                remaining -= self.tree[next_pos]
            bm //= 2
        return min(pos, self.n - 1)  # pos is 0-indexed external position


class PrioritizedReplay:

    def __init__(self, capacity, rng=None):
        self.capacity = capacity
        self.data = [None] * capacity
        self.priorities = [0.0] * capacity
        self.tree = _Fenwick(capacity)
        self.write = 0
        self.size = 0
        self.max_priority = 1.0
        self.rng = rng or random.Random()

    def add(self, transition):
        idx = self.write
        self.data[idx] = transition
        weighted = self.max_priority ** PRIORITY_ALPHA
        self.tree.update(idx, weighted - self.priorities[idx])
        self.priorities[idx] = weighted
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        total = self.tree.total()
        indices = []
        for _ in range(batch_size):
            target = self.rng.uniform(0, total)
            idx = self.tree.find(target)
            if idx >= self.size:
                idx = self.size - 1  # guards float rounding at the tail during warmup
            indices.append(idx)
        return indices, [self.data[i] for i in indices]

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            weighted = (abs(err) + PRIORITY_EPSILON) ** PRIORITY_ALPHA
            self.tree.update(idx, weighted - self.priorities[idx])
            self.priorities[idx] = weighted
            self.max_priority = max(self.max_priority, abs(err) + PRIORITY_EPSILON)

    def __len__(self):
        return self.size
