"""Correctness tests for the 3D (multi-layer + via) sweep SSSP."""

from __future__ import annotations

import math

import torch

from gpu_pnr.baseline import dijkstra_grid, dijkstra_grid_3d
from gpu_pnr.sweep import (
    backtrace_3d,
    sweep_sssp,
    sweep_sssp_3d,
)


def _assert_distances_match(
    d_sweep: torch.Tensor, d_ref: torch.Tensor, atol: float = 5e-3
) -> None:
    """Compare sweep result against reference. Default atol of 5e-3 absorbs
    float32 sum-order drift between cumsum-based scans and Dijkstra's
    edge-by-edge accumulation; for ~50-cell paths through random weights of
    O(1) magnitude, drift can reach a few ULPs * sqrt(N), empirically ~1e-3."""
    d_sweep_cpu = d_sweep.detach().cpu()
    d_ref_cpu = d_ref.detach().cpu()
    finite_mask = torch.isfinite(d_ref_cpu)
    inf_mask = ~finite_mask
    assert torch.allclose(
        d_sweep_cpu[finite_mask], d_ref_cpu[finite_mask], atol=atol
    ), f"finite mismatch:\nsweep={d_sweep_cpu}\nref={d_ref_cpu}"
    assert torch.all(torch.isinf(d_sweep_cpu[inf_mask])), "sweep finite where ref infinite"


def test_single_layer_3d_matches_2d():
    """L=1 with any via_cost must equal the existing 2D sweep result."""
    torch.manual_seed(0)
    H, W = 16, 16
    w2d = torch.rand(H, W) + 0.1
    w2d[5, 3:13] = math.inf
    w3d = w2d.unsqueeze(0)
    source_2d = (0, 0)
    source_3d = (0, 0, 0)
    d_2d, _ = sweep_sssp(w2d, source_2d)
    d_3d, _ = sweep_sssp_3d(w3d, source_3d, via_cost=5.0)
    _assert_distances_match(d_3d[0], d_2d)


def test_two_layers_zero_via_collapses_to_2d_min():
    """via_cost=0 means any layer's d at (r,c) >= the per-(r,c) min over both layers' 2D solutions."""
    torch.manual_seed(1)
    H, W = 12, 12
    w_layer0 = torch.rand(H, W) + 0.1
    w_layer1 = torch.rand(H, W) + 0.1
    w3d = torch.stack([w_layer0, w_layer1], dim=0)
    d3d, _ = sweep_sssp_3d(w3d, (0, 0, 0), via_cost=0.0)
    d_ref = dijkstra_grid_3d(w3d, (0, 0, 0), via_cost=0.0)
    _assert_distances_match(d3d, d_ref)
    finite = torch.isfinite(d3d.cpu())
    assert torch.allclose(d3d.cpu()[0][finite[0] & finite[1]], d3d.cpu()[1][finite[0] & finite[1]], atol=1e-4)


def test_high_via_keeps_path_on_source_layer():
    """When via_cost is huge and source layer has a clear path, distances on
    the other layer should be (source_layer_distance + 2*via_cost) since the
    only way to reach them is one via down and one via back up... actually no,
    with one via you stay on the other layer. So d[other] = d[same] + via_cost
    at minimum (single via from any reachable source-layer cell)."""
    H, W = 8, 8
    w3d = torch.ones(2, H, W)
    via_cost = 100.0
    d, _ = sweep_sssp_3d(w3d, (0, 0, 0), via_cost=via_cost)
    d_layer0 = d[0].cpu()
    d_layer1 = d[1].cpu()
    diff = d_layer1 - d_layer0
    finite = torch.isfinite(d_layer0) & torch.isfinite(d_layer1)
    assert (diff[finite] >= via_cost - 1e-4).all(), (
        "layer-1 distances must each cost at least one via more than layer-0"
    )
    assert (diff[finite] <= via_cost + 1e-4).all(), (
        "with via_cost dominating, the cheapest layer-1 path is exactly one via"
    )


def test_obstacle_detour_via_other_layer():
    """Layer 0 has an obstacle wall; layer 1 is open; via_cost is small.
    A sink past the wall on layer 0 must route via layer 1."""
    H, W = 10, 10
    w0 = torch.ones(H, W)
    w0[5, :] = math.inf
    w1 = torch.ones(H, W)
    w3d = torch.stack([w0, w1], dim=0)
    source = (0, 0, 0)
    sink = (0, H - 1, W - 1)
    d, _ = sweep_sssp_3d(w3d, source, via_cost=1.0)
    d_ref = dijkstra_grid_3d(w3d, source, via_cost=1.0)
    _assert_distances_match(d, d_ref)
    path = backtrace_3d(d.cpu(), w3d.cpu(), source, sink, via_cost=1.0)
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    layers_used = {p[0] for p in path}
    assert 1 in layers_used, "path must use layer 1 to get past the wall"


def test_random_3d_matches_dijkstra():
    torch.manual_seed(2)
    L, H, W = 3, 14, 14
    w = torch.rand(L, H, W) + 0.1
    w[0, 6, 2:12] = math.inf
    w[2, 3:11, 7] = math.inf
    source = (0, 0, 0)
    via_cost = 0.7
    d_sweep, _ = sweep_sssp_3d(w, source, via_cost=via_cost)
    d_ref = dijkstra_grid_3d(w, source, via_cost=via_cost)
    _assert_distances_match(d_sweep, d_ref)


def test_backtrace_path_validity_with_vias():
    torch.manual_seed(3)
    L, H, W = 3, 10, 10
    w = torch.rand(L, H, W) + 0.1
    w[1, 4, 1:9] = math.inf
    source = (0, 0, 0)
    sink = (2, 9, 9)
    via_cost = 0.5
    d, _ = sweep_sssp_3d(w, source, via_cost=via_cost)
    path = backtrace_3d(d.cpu(), w.cpu(), source, sink, via_cost=via_cost)
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    for (l1, i1, j1), (l2, i2, j2) in zip(path, path[1:]):
        in_layer = l1 == l2 and abs(i1 - i2) + abs(j1 - j2) == 1
        via = abs(l1 - l2) == 1 and i1 == i2 and j1 == j2
        assert in_layer or via, f"non-adjacent step {(l1, i1, j1)} -> {(l2, i2, j2)}"
    for lyr, i, j in path:
        assert not math.isinf(float(w[lyr, i, j])), "path goes through obstacle"


def test_unreachable_isolated_layer():
    """Source on layer 0 is fully walled off in layer 0; via_cost is finite but
    layer-0 obstacles force any path off layer 0 immediately."""
    H, W = 6, 6
    w0 = torch.ones(H, W)
    w0[1, :] = math.inf
    w1 = torch.ones(H, W)
    w3d = torch.stack([w0, w1], dim=0)
    source = (0, 0, 0)
    sink = (0, H - 1, W - 1)
    d, _ = sweep_sssp_3d(w3d, source, via_cost=1.0)
    assert torch.isfinite(d[sink])
    d_ref = dijkstra_grid_3d(w3d, source, via_cost=1.0)
    _assert_distances_match(d, d_ref)


def test_mps_matches_cpu_3d():
    if not torch.backends.mps.is_available():
        return
    torch.manual_seed(4)
    L, H, W = 3, 32, 32
    w_cpu = torch.rand(L, H, W) + 0.1
    w_cpu[1, 10, 5:25] = math.inf
    source = (0, 0, 0)
    via_cost = 0.8
    d_cpu, _ = sweep_sssp_3d(w_cpu, source, via_cost=via_cost)
    d_mps, _ = sweep_sssp_3d(w_cpu.to("mps"), source, via_cost=via_cost)
    finite = torch.isfinite(d_cpu)
    inf = ~finite
    d_mps_cpu = d_mps.cpu()
    assert torch.allclose(d_mps_cpu[finite], d_cpu[finite], atol=5e-2), (
        "MPS and CPU 3D sweep disagree beyond float32 sum-order drift"
    )
    assert torch.all(torch.isinf(d_mps_cpu[inf]))


def test_dijkstra_3d_collapses_to_2d_when_one_layer():
    """Sanity: dijkstra_grid_3d on (1, H, W) must equal dijkstra_grid on (H, W)."""
    torch.manual_seed(5)
    H, W = 10, 10
    w2d = torch.rand(H, W) + 0.1
    w2d[3, 1:8] = math.inf
    d_2d = dijkstra_grid(w2d, (0, 0))
    d_3d = dijkstra_grid_3d(w2d.unsqueeze(0), (0, 0, 0), via_cost=99.0)
    _assert_distances_match(d_3d[0], d_2d)
