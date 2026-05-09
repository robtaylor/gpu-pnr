"""Scaling benchmark: sweep SSSP across a range of grid sizes.

Reports wall-clock per size, per-iter time, throughput (cells/ms), and
the CPU Dijkstra baseline where it's still tractable.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from gpu_pnr.baseline import dijkstra_grid
from gpu_pnr.sweep import sweep_sssp


DEFAULT_SIZES = (256, 512, 1024, 2048, 4096, 8192)
BASELINE_MAX_SIZE = 2048


def make_grid(H: int, W: int, obstacle_frac: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.ones(H, W)
    mask = torch.rand(H, W, generator=g) < obstacle_frac
    w[mask] = math.inf
    w[0, 0] = 1.0
    w[H - 1, W - 1] = 1.0
    return w


def time_sweep(w_dev: torch.Tensor, source: tuple[int, int], device: str) -> tuple[float, int, float]:
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    d, iters = sweep_sssp(w_dev, source)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    sink = (w_dev.shape[0] - 1, w_dev.shape[1] - 1)
    cost = float(d[sink].cpu())
    return elapsed, iters, cost


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    p.add_argument("--obstacles", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--baseline-max", type=int, default=BASELINE_MAX_SIZE)
    p.add_argument("--warmup", action="store_true", default=True)
    args = p.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}, obstacle_frac={args.obstacles}")
    print()

    if args.warmup:
        w_warmup = make_grid(64, 64, args.obstacles, args.seed).to(device)
        _ = sweep_sssp(w_warmup, (0, 0))
        if device == "mps":
            torch.mps.synchronize()

    print(
        f"{'size':>6}  {'cells':>10}  {'sweep_ms':>10}  {'iters':>6}  "
        f"{'ms/iter':>8}  {'Mcells/s':>9}  {'dijkstra_ms':>12}  {'speedup':>8}  cost"
    )
    print("-" * 100)

    for n in args.sizes:
        w_cpu = make_grid(n, n, args.obstacles, args.seed)
        w_dev = w_cpu.to(device)
        sweep_ms, iters, cost_sweep = time_sweep(w_dev, (0, 0), device)
        sweep_ms *= 1000.0
        cells = n * n
        ms_per_iter = sweep_ms / max(iters, 1)
        throughput = cells / sweep_ms / 1e3

        if n <= args.baseline_max:
            t0 = time.perf_counter()
            d_ref = dijkstra_grid(w_cpu, (0, 0))
            dijk_ms = (time.perf_counter() - t0) * 1000.0
            cost_ref = float(d_ref[n - 1, n - 1])
            speedup = dijk_ms / sweep_ms if sweep_ms > 0 else float("inf")
            agree = "OK" if (
                (math.isfinite(cost_ref) and abs(cost_ref - cost_sweep) < 1e-2)
                or (math.isinf(cost_ref) and math.isinf(cost_sweep))
            ) else f"DIFF ref={cost_ref} sweep={cost_sweep}"
            dijk_str = f"{dijk_ms:>12.1f}"
            speedup_str = f"{speedup:>7.2f}x"
        else:
            agree = ""
            dijk_str = f"{'skipped':>12}"
            speedup_str = f"{'-':>8}"

        cost_str = f"{cost_sweep:.0f}" if math.isfinite(cost_sweep) else "inf"
        print(
            f"{n:>6}  {cells:>10}  {sweep_ms:>10.1f}  {iters:>6}  "
            f"{ms_per_iter:>8.2f}  {throughput:>9.1f}  {dijk_str}  {speedup_str}  {cost_str} {agree}"
        )


if __name__ == "__main__":
    main()
