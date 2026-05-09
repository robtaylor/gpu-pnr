"""Phase 1 demo: synthetic obstacle grid, sweep SSSP on MPS vs Dijkstra on CPU.

Reports wall-clock for each, route lengths, and asserts the routes have
matching cost.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from gpu_pnr.baseline import dijkstra_grid
from gpu_pnr.sweep import backtrace, sweep_sssp


def make_grid(H: int, W: int, obstacle_frac: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.ones(H, W)
    mask = torch.rand(H, W, generator=g) < obstacle_frac
    w[mask] = math.inf
    return w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--obstacles", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    H = W = args.size
    print(f"Grid: {H}x{W}, obstacle_frac={args.obstacles}, device={device}")

    w_cpu = make_grid(H, W, args.obstacles, args.seed)
    source = (0, 0)
    sink = (H - 1, W - 1)

    while not math.isfinite(float(w_cpu[source])):
        w_cpu[source] = 1.0
    while not math.isfinite(float(w_cpu[sink])):
        w_cpu[sink] = 1.0

    t0 = time.perf_counter()
    d_ref = dijkstra_grid(w_cpu, source)
    t_ref = time.perf_counter() - t0

    w_dev = w_cpu.to(device)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    d_sweep, iters = sweep_sssp(w_dev, source)
    if device == "mps":
        torch.mps.synchronize()
    t_sweep = time.perf_counter() - t0

    cost_ref = float(d_ref[sink])
    cost_sweep = float(d_sweep[sink].cpu())

    path_sweep = backtrace(d_sweep.cpu(), w_cpu, source, sink)
    path_len = len(path_sweep) if path_sweep else 0

    print(f"Dijkstra (CPU):    cost={cost_ref:>8.3f}   time={t_ref*1000:>7.2f} ms")
    print(
        f"Sweep ({device:>3}):       cost={cost_sweep:>8.3f}   time={t_sweep*1000:>7.2f} ms   iters={iters}"
    )
    print(f"Sweep backtrace path length: {path_len} cells")

    if math.isfinite(cost_ref):
        assert abs(cost_ref - cost_sweep) < 1e-3, (
            f"cost mismatch: ref={cost_ref}  sweep={cost_sweep}"
        )
        print("OK: sweep cost matches Dijkstra")
    else:
        assert math.isinf(cost_sweep)
        print("Sink unreachable (consistent across both)")


if __name__ == "__main__":
    main()
