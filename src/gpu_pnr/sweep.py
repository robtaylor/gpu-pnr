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

from collections.abc import Callable

import torch

INF_PROXY = 1e4


def _converge_or_max(
    d: torch.Tensor,
    body: Callable[[torch.Tensor], torch.Tensor],
    max_iters: int,
    check_every: int,
) -> tuple[torch.Tensor, int]:
    """Iterate `body(d)` until fixed point or `max_iters`, checking every K.

    Reuses `d_check`'s storage via in-place `copy_` instead of cloning each
    check, since `d` itself is reassigned to a fresh tensor every iteration
    (the sweep helpers return new tensors) -- only `d_check` needs persistence.
    """
    d_check = d.clone()
    for it in range(max_iters):
        d = body(d)
        if (it + 1) % check_every == 0:
            if torch.equal(d, d_check):
                return d, it + 1
            d_check.copy_(d)
    return d, max_iters


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
    max_iters: int = 200,
    check_every: int = 8,
) -> tuple[torch.Tensor, int]:
    """Compute shortest-path distances on a 2D grid via alternating axis sweeps.

    Convergence is checked every `check_every` iterations rather than every
    iteration -- the per-iter `torch.equal` forces a CPU<->GPU sync that
    serialises the GPU pipeline. Checking every K iters lets K iterations
    run async between syncs.

    Args:
        w: (H, W) tensor, cost to enter each cell. Use float('inf') for obstacles.
        source: (row, col) of the source cell.
        max_iters: cap on outer-loop iterations.
        check_every: how often (in iterations) to test for convergence.

    Returns:
        (d, iters) where d is the (H, W) distance tensor and iters is the
        number of outer iterations executed.
    """
    d = torch.full_like(w, float("inf"))
    sr, sc = source
    d[sr, sc] = 0.0
    w_proxy = _to_proxy(w)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, w_proxy, axis=1)
        d = _sweep_backward(d, w_proxy, axis=1)
        d = _sweep_forward(d, w_proxy, axis=0)
        d = _sweep_backward(d, w_proxy, axis=0)
        return _mask_polluted(d)

    return _converge_or_max(d, step, max_iters, check_every)


def sweep_sssp_multi(
    w: torch.Tensor,
    sources: list[tuple[int, int]],
    max_iters: int = 200,
    check_every: int = 8,
) -> tuple[torch.Tensor, int]:
    """Compute K shortest-path distance maps from K sources concurrently.

    Generalises `sweep_sssp` to a batch dim (K). Each source gets its own
    distance map; sources in the same call do NOT see each other's wires
    (no obstacle update between them). The intended use is throughput:
    one multi-sweep replaces K sequential sweeps for batched routing.

    Args:
        w: (H, W) tensor of cell-entry costs.
        sources: list of K (row, col) source pins.
        max_iters, check_every: as in `sweep_sssp`.

    Returns:
        (d, iters) where d is (K, H, W).
    """
    K = len(sources)
    H, W = w.shape
    d = torch.full((K, H, W), float("inf"), device=w.device, dtype=w.dtype)
    for k, (sr, sc) in enumerate(sources):
        d[k, sr, sc] = 0.0
    w_proxy = _to_proxy(w).unsqueeze(0)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, w_proxy, axis=2)
        d = _sweep_backward(d, w_proxy, axis=2)
        d = _sweep_forward(d, w_proxy, axis=1)
        d = _sweep_backward(d, w_proxy, axis=1)
        return _mask_polluted(d)

    return _converge_or_max(d, step, max_iters, check_every)


def sweep_sssp_3d(
    w: torch.Tensor,
    source: tuple[int, int, int],
    via_cost: float = 1.0,
    max_iters: int = 200,
    check_every: int = 8,
) -> tuple[torch.Tensor, int]:
    """Compute shortest-path distances on a multi-layer grid via sweep iteration.

    Each layer is 4-connected for horizontal wires; adjacent layers connect at
    the same (r, c) via an edge of weight `via_cost` (a via). Within a layer,
    obstacles are float('inf') in `w`. Vias are unobstructed and have constant
    cost regardless of (r, c) -- a deliberate simplification of real ASIC
    via cells (which can be DRC-blocked).

    Edge model: arrival at (l, r, c) horizontally pays w[l, r, c]; arrival via
    a via pays only `via_cost` (the destination cell's w is not also charged).

    Per iteration:
        1. Four intra-layer sweeps (axis=2 fwd/bwd, axis=1 fwd/bwd).
           Vectorised over L: every layer is scanned in parallel.
        2. Mask INF_PROXY pollution back to inf.
        3. Sequential per-layer min relaxation along axis=0 (up then down).
           Each step is `d[l] = min(d[l], d[l-1] + via_cost)` followed by an
           obstacle re-mask so via paths neither land on nor chain through
           blocked cells. A naive cumsum-cummin scan along axis=0 would let
           vias "pass through" intermediate obstacles by adding via_cost*|dl|
           regardless of whether those cells exist; the sequential form costs
           2(L-1) min/where ops per iter and is correct under obstacles.

    Args:
        w: (L, H, W) tensor, cost to enter each cell. inf for obstacles.
        source: (layer, row, col).
        via_cost: edge weight for one via transition between adjacent layers.
        max_iters, check_every: as in `sweep_sssp`.

    Returns:
        (d, iters) where d is the (L, H, W) distance tensor.
    """
    L = w.shape[0]
    d = torch.full_like(w, float("inf"))
    sl, sr, sc = source
    d[sl, sr, sc] = 0.0
    w_proxy = _to_proxy(w)
    obstacle_mask = torch.isinf(w)
    inf_scalar = float("inf")

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, w_proxy, axis=2)
        d = _sweep_backward(d, w_proxy, axis=2)
        d = _sweep_forward(d, w_proxy, axis=1)
        d = _sweep_backward(d, w_proxy, axis=1)
        d = _mask_polluted(d)
        for lyr in range(1, L):
            d[lyr] = torch.minimum(d[lyr], d[lyr - 1] + via_cost)
            d[lyr] = torch.where(obstacle_mask[lyr], inf_scalar, d[lyr])
        for lyr in range(L - 2, -1, -1):
            d[lyr] = torch.minimum(d[lyr], d[lyr + 1] + via_cost)
            d[lyr] = torch.where(obstacle_mask[lyr], inf_scalar, d[lyr])
        return d

    return _converge_or_max(d, step, max_iters, check_every)


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


def backtrace_3d(
    d: torch.Tensor,
    w: torch.Tensor,
    source: tuple[int, int, int],
    sink: tuple[int, int, int],
    via_cost: float = 1.0,
    atol: float = 1e-5,
) -> list[tuple[int, int, int]] | None:
    """Reconstruct a shortest 3D path from source to sink.

    At each step, prefer in-layer 4-neighbors (predecessor distance must equal
    d[cur] - w[cur]); fall back to cross-layer via neighbors at the same (r, c)
    on the layer above or below (predecessor distance = d[cur] - via_cost).
    """
    sl, sr, sc = source
    tl, ti, tj = sink
    L, H, W = d.shape

    if not torch.isfinite(d[tl, ti, tj]):
        return None

    path: list[tuple[int, int, int]] = [(tl, ti, tj)]
    cur_l, cur_i, cur_j = tl, ti, tj

    while (cur_l, cur_i, cur_j) != (sl, sr, sc):
        in_layer_target = (d[cur_l, cur_i, cur_j] - w[cur_l, cur_i, cur_j]).item()
        via_target = (d[cur_l, cur_i, cur_j] - via_cost).item()
        moved = False
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = cur_i + di, cur_j + dj
            if 0 <= ni < H and 0 <= nj < W and torch.isfinite(d[cur_l, ni, nj]):
                if abs(d[cur_l, ni, nj].item() - in_layer_target) <= atol:
                    path.append((cur_l, ni, nj))
                    cur_i, cur_j = ni, nj
                    moved = True
                    break
        if moved:
            continue
        for dl in (-1, 1):
            nl = cur_l + dl
            if 0 <= nl < L and torch.isfinite(d[nl, cur_i, cur_j]):
                if abs(d[nl, cur_i, cur_j].item() - via_target) <= atol:
                    path.append((nl, cur_i, cur_j))
                    cur_l = nl
                    moved = True
                    break
        if not moved:
            return None

    path.reverse()
    return path
