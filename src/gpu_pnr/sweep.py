"""Sweep-based SSSP on a 2D grid (4-connected), Bellman-Ford via Gauss-Seidel.

Each "iteration" runs four directional axis sweeps (H-forward, H-backward,
V-forward, V-backward). Each sweep is implemented as one cumsum + one
cummin per axis, which dispatches as a parallel scan on GPU rather than
N sequential kernel launches.

Forward-sweep derivation:
  d_new[j] = min over k<=j of (d[k] + sum w[k+1..j])
           = cw[j] + min over k<=j of (d[k] - cw[k])
           = cw[j] + cummin(d - cw)[j]
where cw = cumsum(w) along the sweep axis (inclusive).

Obstacles (w = inf) would make cumsum NaN, so we substitute a large
finite proxy and mask polluted distances back to inf each iteration.

Proxy magnitude is bounded by float32 precision: MPS doesn't support
float64, and `(cw + cm)` in the scan can lose precision when both
operands are near `proxy * N`. For N ~ 1024, proxy ~ 1e4 keeps the
worst-case magnitude under 1e7 where float32 ULP is ~1. Larger grids
need a different obstacle-handling scheme.

Convergence: O(diameter) iterations; typically a handful for sparse
obstacles.
"""

from __future__ import annotations

import torch

INF_PROXY = 1e4


def _to_proxy(w: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isinf(w), torch.full_like(w, INF_PROXY), w)


def _sweep_forward(d: torch.Tensor, w_proxy: torch.Tensor, axis: int) -> torch.Tensor:
    cw = torch.cumsum(w_proxy, dim=axis)
    v = d - cw
    cm, _ = torch.cummin(v, dim=axis)
    return cw + cm


def _sweep_backward(d: torch.Tensor, w_proxy: torch.Tensor, axis: int) -> torch.Tensor:
    d_f = torch.flip(d, dims=[axis])
    w_f = torch.flip(w_proxy, dims=[axis])
    cw = torch.cumsum(w_f, dim=axis)
    v = d_f - cw
    cm, _ = torch.cummin(v, dim=axis)
    return torch.flip(cw + cm, dims=[axis])


def _mask_polluted(d: torch.Tensor) -> torch.Tensor:
    inf = torch.full_like(d, float("inf"))
    return torch.where(d > INF_PROXY / 2, inf, d)


def sweep_sssp(
    w: torch.Tensor,
    source: tuple[int, int],
    max_iters: int = 50,
) -> tuple[torch.Tensor, int]:
    """Compute shortest-path distances on a 2D grid via alternating axis sweeps.

    Args:
        w: (H, W) tensor, cost to enter each cell. Use float('inf') for obstacles.
        source: (row, col) of the source cell.
        max_iters: cap on outer-loop iterations.

    Returns:
        (d, iters) where d is the (H, W) distance tensor and iters is the
        number of outer iterations executed before convergence.
    """
    d = torch.full_like(w, float("inf"))
    sr, sc = source
    d[sr, sc] = 0.0

    w_proxy = _to_proxy(w)

    for it in range(max_iters):
        d_prev = d.clone()
        d = _sweep_forward(d, w_proxy, axis=1)
        d = _sweep_backward(d, w_proxy, axis=1)
        d = _sweep_forward(d, w_proxy, axis=0)
        d = _sweep_backward(d, w_proxy, axis=0)
        d = _mask_polluted(d)
        if torch.equal(d, d_prev):
            return d, it + 1

    return d, max_iters


def backtrace(
    d: torch.Tensor,
    w: torch.Tensor,
    source: tuple[int, int],
    sink: tuple[int, int],
    atol: float = 1e-5,
) -> list[tuple[int, int]] | None:
    """Reconstruct a shortest path from source to sink given the distance map.

    Walks backward from sink: at each step, pick a 4-neighbor n with
    d[n] + w[current] ~= d[current].
    """
    sr, sc = source
    si, sj = sink
    H, W = d.shape

    if not torch.isfinite(d[si, sj]):
        return None

    path: list[tuple[int, int]] = [(si, sj)]
    cur_i, cur_j = si, sj

    while (cur_i, cur_j) != (sr, sc):
        target = (d[cur_i, cur_j] - w[cur_i, cur_j]).item()
        moved = False
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = cur_i + di, cur_j + dj
            if 0 <= ni < H and 0 <= nj < W and torch.isfinite(d[ni, nj]):
                if abs(d[ni, nj].item() - target) <= atol:
                    path.append((ni, nj))
                    cur_i, cur_j = ni, nj
                    moved = True
                    break
        if not moved:
            return None

    path.reverse()
    return path
