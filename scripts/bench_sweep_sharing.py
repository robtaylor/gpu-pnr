"""Compare K sequential sweep_sssp() calls vs one sweep_sssp_multi(K).

Same grid, same K random source positions. Reports wall-clock for each
and the speedup. Demonstrates the GPU throughput potential of
sweep-sharing.
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from gpu_pnr.sweep import sweep_sssp, sweep_sssp_multi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print(f"Grid: {args.size}x{args.size}  device={device}")
    print()

    w_warm = torch.ones(64, 64, device=device)
    _ = sweep_sssp(w_warm, (0, 0))
    _ = sweep_sssp_multi(w_warm, [(0, 0), (10, 10)])
    if device == "mps":
        torch.mps.synchronize()

    w = torch.ones(args.size, args.size, device=device)
    rng = random.Random(args.seed)
    all_sources = [
        (rng.randrange(args.size), rng.randrange(args.size))
        for _ in range(max(args.ks))
    ]

    print(
        f"{'K':>4}  {'sequential_ms':>14}  {'multi_ms':>10}  "
        f"{'speedup':>8}  {'ms/source_seq':>14}  {'ms/source_mul':>14}"
    )
    print("-" * 80)
    for K in args.ks:
        sources = all_sources[:K]

        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for src in sources:
            _ = sweep_sssp(w, src)
        if device == "mps":
            torch.mps.synchronize()
        t_seq = (time.perf_counter() - t0) * 1000.0

        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = sweep_sssp_multi(w, sources)
        if device == "mps":
            torch.mps.synchronize()
        t_mul = (time.perf_counter() - t0) * 1000.0

        speedup = t_seq / t_mul if t_mul > 0 else float("inf")
        print(
            f"{K:>4}  {t_seq:>14.1f}  {t_mul:>10.1f}  "
            f"{speedup:>7.2f}x  {t_seq/K:>14.2f}  {t_mul/K:>14.2f}"
        )


if __name__ == "__main__":
    main()
