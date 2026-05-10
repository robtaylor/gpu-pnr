"""Correctness tests: sweep SSSP must agree with Dijkstra on small grids."""

from __future__ import annotations

import math

import torch

from gpu_pnr.baseline import dijkstra_grid
from gpu_pnr.sweep import backtrace, sweep_sssp


def _assert_distances_match(d_sweep: torch.Tensor, d_ref: torch.Tensor) -> None:
    d_sweep_cpu = d_sweep.detach().cpu()
    d_ref_cpu = d_ref.detach().cpu()
    finite_mask = torch.isfinite(d_ref_cpu)
    inf_mask = ~finite_mask

    assert torch.allclose(
        d_sweep_cpu[finite_mask], d_ref_cpu[finite_mask], atol=1e-4
    ), f"finite-distance mismatch:\nsweep={d_sweep_cpu}\nref={d_ref_cpu}"
    assert torch.all(torch.isinf(d_sweep_cpu[inf_mask])), (
        "sweep finite where ref infinite"
    )


def test_open_grid_unit_weights():
    H, W = 8, 8
    w = torch.ones(H, W)
    source = (0, 0)
    d_sweep, _ = sweep_sssp(w, source)
    d_ref = dijkstra_grid(w, source)
    _assert_distances_match(d_sweep, d_ref)


def test_open_grid_random_weights():
    torch.manual_seed(0)
    H, W = 16, 16
    w = torch.rand(H, W) + 0.1
    source = (3, 5)
    d_sweep, _ = sweep_sssp(w, source)
    d_ref = dijkstra_grid(w, source)
    _assert_distances_match(d_sweep, d_ref)


def test_grid_with_obstacles():
    H, W = 12, 12
    w = torch.ones(H, W)
    w[5, 0:10] = math.inf
    w[7, 2:12] = math.inf
    source = (0, 0)
    d_sweep, _ = sweep_sssp(w, source)
    d_ref = dijkstra_grid(w, source)
    _assert_distances_match(d_sweep, d_ref)


def test_unreachable_region():
    H, W = 8, 8
    w = torch.ones(H, W)
    w[4, :] = math.inf
    source = (0, 0)
    d_sweep, _ = sweep_sssp(w, source)
    d_ref = dijkstra_grid(w, source)
    _assert_distances_match(d_sweep, d_ref)
    assert torch.isinf(d_sweep[7, 7])


def test_backtrace_valid_path():
    H, W = 10, 10
    w = torch.ones(H, W)
    w[3, 0:7] = math.inf
    source = (0, 0)
    sink = (9, 9)
    d_sweep, _ = sweep_sssp(w, source)
    path = backtrace(d_sweep, w, source, sink)
    assert path is not None
    assert path[0] == source
    assert path[-1] == sink
    for (i1, j1), (i2, j2) in zip(path, path[1:]):
        assert abs(i1 - i2) + abs(j1 - j2) == 1, "non-adjacent step"
    assert all(not math.isinf(float(w[i, j])) for (i, j) in path), (
        "path goes through obstacle"
    )


def test_multi_source_matches_per_source():
    torch.manual_seed(0)
    H, W = 24, 24
    w = torch.rand(H, W) + 0.1
    w[10, 4:18] = math.inf
    sources = [(0, 0), (H - 1, W - 1), (5, 12), (15, 5)]

    from gpu_pnr.sweep import sweep_sssp_multi

    d_multi, _ = sweep_sssp_multi(w, sources)
    for k, src in enumerate(sources):
        d_single, _ = sweep_sssp(w, src)
        finite_mask = torch.isfinite(d_single)
        diff = (d_multi[k][finite_mask] - d_single[finite_mask]).abs()
        assert diff.max().item() < 5e-2, (
            f"source {k}={src}: max diff {diff.max().item()}"
        )
        inf_mask = ~finite_mask
        assert torch.all(torch.isinf(d_multi[k][inf_mask])), (
            f"source {k}: multi finite where single is inf"
        )


def test_mps_matches_cpu_when_available():
    if not torch.backends.mps.is_available():
        return
    torch.manual_seed(1)
    H, W = 32, 32
    w_cpu = torch.rand(H, W) + 0.1
    w_cpu[10, 5:25] = math.inf
    source = (0, 0)
    d_cpu, _ = sweep_sssp(w_cpu, source)
    w_mps = w_cpu.to("mps")
    d_mps, _ = sweep_sssp(w_mps, source)
    finite_mask = torch.isfinite(d_cpu)
    inf_mask = ~finite_mask
    d_mps_cpu = d_mps.cpu()
    assert torch.allclose(
        d_mps_cpu[finite_mask], d_cpu[finite_mask], atol=5e-2
    ), "MPS and CPU sweep disagree beyond float32 sum-order drift"
    assert torch.all(torch.isinf(d_mps_cpu[inf_mask]))
