"""Reference Dijkstra on a 2D grid for ground-truth comparison against sweep SSSP."""

from __future__ import annotations

import heapq
import math

import torch


def dijkstra_grid(
    w: torch.Tensor,
    source: tuple[int, int],
) -> torch.Tensor:
    """Standard Dijkstra on a 4-connected grid.

    Args:
        w: (H, W) tensor (any device); cost to enter each cell. inf for obstacles.
        source: (row, col).

    Returns:
        (H, W) float32 CPU tensor of shortest distances. inf where unreachable.
    """
    w_np = w.detach().cpu().numpy()
    H, W = w_np.shape
    sr, sc = source

    d = [[math.inf] * W for _ in range(H)]
    d[sr][sc] = 0.0

    pq: list[tuple[float, int, int]] = [(0.0, sr, sc)]
    while pq:
        cur_d, i, j = heapq.heappop(pq)
        if cur_d > d[i][j]:
            continue
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                cost = float(w_np[ni, nj])
                if math.isinf(cost):
                    continue
                new_d = cur_d + cost
                if new_d < d[ni][nj]:
                    d[ni][nj] = new_d
                    heapq.heappush(pq, (new_d, ni, nj))

    return torch.tensor(d, dtype=torch.float32)
