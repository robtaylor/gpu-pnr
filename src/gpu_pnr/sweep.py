"""Sweep-based SSSP on a 2D grid (4-connected), Bellman-Ford via Gauss-Seidel.

Each "iteration" runs four directional axis sweeps (H-forward, H-backward,
V-forward, V-backward). Each sweep is implemented as a segmented cumsum +
segmented cummin per axis, which dispatches as a parallel scan on GPU
rather than N sequential kernel launches.

Forward-sweep derivation (within a segment, i.e., a maximal run of
non-obstacle cells along the axis):
  d_new[j] = min over k<=j of (d[k] + sum w[k+1..j])
           = seg_cw[j] + min over k<=j in same segment of (d[k] - seg_cw[k])
where seg_cw[j] = cumsum of w from the current segment's start to j.

Obstacles are handled with a segmented scan, not a finite proxy:
  - cumsum(w_clean) where w_clean treats obstacles as 0; magnitudes stay
    proportional to real path weight (no INF_PROXY * N inflation).
  - seg_cw[j] = cw[j] - cw_at_most_recent_obstacle[j] (the latter via
    cummax of cw masked at obstacle positions).
  - cummin's input is offset by seg_id * SEG_BARRIER, where seg_id is the
    cumulative obstacle count along the axis. Earlier segments have a
    smaller offset subtracted, so their values are larger; cummin
    therefore can never pick across a segment boundary. The offset is
    subtracted back exactly to recover segment-restricted minima.

Float32 precision budget: max(|seg_cw|) is bounded by per-segment path
weight (small); max(|seg_id * SEG_BARRIER|) is the new dominant term.
With SEG_BARRIER=2e4 and max ~200 obstacles per row (5% density at 4096),
worst-case magnitude is ~4e6; float32 ULP ~0.25, leaving comfortable
headroom for unit-weight distances on grids well past 4096^2.

Convergence: O(diameter) iterations; typically a handful for sparse
obstacles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

SEG_BARRIER = 2e4
MAX_LEGIT_DISTANCE = SEG_BARRIER / 2


@dataclass(frozen=True)
class _ScanState:
    """Loop-invariant per-(axis, direction) state for the masked scan.

    seg_cw[j] = cumsum of finite-only weight from the current segment's start
    through j. seg_id_barrier[j] = seg_id[j] * SEG_BARRIER, the offset that
    keeps cummin from picking across segment boundaries. obstacle_mask is the
    same orientation as seg_cw / seg_id_barrier (flipped along axis for the
    backward direction).
    """

    seg_cw: torch.Tensor
    seg_id_barrier: torch.Tensor
    obstacle_mask: torch.Tensor


def _obstacle_mask(w: torch.Tensor) -> torch.Tensor:
    return torch.isinf(w)


def _precompute_scan(
    w: torch.Tensor, obstacle_mask: torch.Tensor, axis: int
) -> _ScanState:
    """Compute the parts of the segmented scan that depend only on (w, mask).

    Hoisting these out of the convergence loop matters: they're recomputed
    O(diameter) times otherwise, but they don't change as `d` evolves. With
    the hoist, the per-iter inner sweep collapses to one cummin and a few
    arithmetic ops on (d - seg_cw - seg_id_barrier).
    """
    w_clean = torch.where(obstacle_mask, 0.0, w)
    cw = torch.cumsum(w_clean, dim=axis)
    seg_id = torch.cumsum(obstacle_mask.to(w.dtype), dim=axis)
    cw_at_obs = torch.where(obstacle_mask, cw, 0.0)
    cw_recent_obs, _ = torch.cummax(cw_at_obs, dim=axis)
    return _ScanState(
        seg_cw=cw - cw_recent_obs,
        seg_id_barrier=seg_id * SEG_BARRIER,
        obstacle_mask=obstacle_mask,
    )


def _precompute_axis(
    w: torch.Tensor, obstacle_mask: torch.Tensor, axis: int
) -> tuple[_ScanState, _ScanState]:
    """Forward + backward state for one axis. Backward state is precomputed on
    the flipped (w, mask) so the per-iter backward sweep just flips `d`."""
    fwd = _precompute_scan(w, obstacle_mask, axis)
    w_f = torch.flip(w, dims=[axis])
    obstacle_mask_f = torch.flip(obstacle_mask, dims=[axis])
    bwd = _precompute_scan(w_f, obstacle_mask_f, axis)
    return fwd, bwd


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


def _sweep_forward(
    d: torch.Tensor, state: _ScanState, axis: int
) -> torch.Tensor:
    """Forward axis sweep using the precomputed segmented-scan state.

    When every cell in the current segment is unreachable (d=inf), v is inf
    there, so cummin propagates the prior segment's running min forward; the
    reconstruction shifts that value by (S-S')*SEG_BARRIER, producing a
    finite-but-large polluted distance instead of inf. The polluted-mask
    step at the end (d > MAX_LEGIT_DISTANCE) returns those to inf -- legit
    distances are bounded by the segment's finite path weight, well under
    MAX_LEGIT_DISTANCE for grids within the precision budget documented at
    the top of this module.
    """
    inf_scalar = float("inf")
    v = d - state.seg_cw - state.seg_id_barrier
    v = torch.where(state.obstacle_mask, inf_scalar, v)
    cm, _ = torch.cummin(v, dim=axis)
    d_new = state.seg_cw + cm + state.seg_id_barrier
    polluted = d_new > MAX_LEGIT_DISTANCE
    return torch.where(state.obstacle_mask | polluted, inf_scalar, d_new)


def _sweep_backward(
    d: torch.Tensor, state: _ScanState, axis: int
) -> torch.Tensor:
    """Backward axis sweep. `state` must be the *flipped*-direction state
    produced by `_precompute_axis`; the polluted-mask is applied inside
    `_sweep_forward`."""
    d_f = torch.flip(d, dims=[axis])
    return torch.flip(_sweep_forward(d_f, state, axis), dims=[axis])


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
    obstacle_mask = _obstacle_mask(w)
    fwd_h, bwd_h = _precompute_axis(w, obstacle_mask, axis=1)
    fwd_v, bwd_v = _precompute_axis(w, obstacle_mask, axis=0)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=1)
        d = _sweep_backward(d, bwd_h, axis=1)
        d = _sweep_forward(d, fwd_v, axis=0)
        return _sweep_backward(d, bwd_v, axis=0)

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
    w_b = w.unsqueeze(0)
    obstacle_mask_b = _obstacle_mask(w).unsqueeze(0)
    fwd_h, bwd_h = _precompute_axis(w_b, obstacle_mask_b, axis=2)
    fwd_v, bwd_v = _precompute_axis(w_b, obstacle_mask_b, axis=1)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=2)
        d = _sweep_backward(d, bwd_h, axis=2)
        d = _sweep_forward(d, fwd_v, axis=1)
        return _sweep_backward(d, bwd_v, axis=1)

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
    obstacle_mask = _obstacle_mask(w)
    inf_scalar = float("inf")
    fwd_h, bwd_h = _precompute_axis(w, obstacle_mask, axis=2)
    fwd_v, bwd_v = _precompute_axis(w, obstacle_mask, axis=1)

    def step(d: torch.Tensor) -> torch.Tensor:
        d = _sweep_forward(d, fwd_h, axis=2)
        d = _sweep_backward(d, bwd_h, axis=2)
        d = _sweep_forward(d, fwd_v, axis=1)
        d = _sweep_backward(d, bwd_v, axis=1)
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
