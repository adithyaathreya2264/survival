from config import GRID_SIZE, VIEW_RADIUS, P_WALL, P_SELF


def perceive(grid, pos):
    r0, c0 = pos
    obs = []
    for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
        for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                obs.append(grid[r][c])
            else:
                obs.append(P_WALL)
    center = len(obs) // 2
    obs[center] = P_SELF
    return obs
