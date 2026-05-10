"""Reference Dijkstra on 2D and 3D grids for ground-truth comparison against sweep SSSP."""

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


def dijkstra_grid_3d(
    w: torch.Tensor,
    source: tuple[int, int, int],
    via_cost: float = 1.0,
) -> torch.Tensor:
    """Standard Dijkstra on a multi-layer 4-connected grid with via edges.

    Each layer is 4-connected for in-layer wires (edge weight = w[neighbor]);
    adjacent layers connect at the same (r, c) via a via edge of weight
    `via_cost` regardless of the cell's wire cost.

    Args:
        w: (L, H, W) tensor; inf for obstacles.
        source: (layer, row, col).
        via_cost: edge weight for one via transition.

    Returns:
        (L, H, W) float32 CPU tensor of shortest distances.
    """
    w_np = w.detach().cpu().numpy()
    L, H, W = w_np.shape
    sl, sr, sc = source

    d = [[[math.inf] * W for _ in range(H)] for _ in range(L)]
    d[sl][sr][sc] = 0.0

    pq: list[tuple[float, int, int, int]] = [(0.0, sl, sr, sc)]
    while pq:
        cur_d, lyr, i, j = heapq.heappop(pq)
        if cur_d > d[lyr][i][j]:
            continue
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                cost = float(w_np[lyr, ni, nj])
                if math.isinf(cost):
                    continue
                new_d = cur_d + cost
                if new_d < d[lyr][ni][nj]:
                    d[lyr][ni][nj] = new_d
                    heapq.heappush(pq, (new_d, lyr, ni, nj))
        for dl in (-1, 1):
            nl = lyr + dl
            if 0 <= nl < L and not math.isinf(float(w_np[nl, i, j])):
                new_d = cur_d + via_cost
                if new_d < d[nl][i][j]:
                    d[nl][i][j] = new_d
                    heapq.heappush(pq, (new_d, nl, i, j))

    return torch.tensor(d, dtype=torch.float32)
