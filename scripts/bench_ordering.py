"""Compare net-ordering strategies on the same workload.

Reports success rate, total wirelength of routed nets, and wall-clock
across strategies. Same grid + nets each time, only the order differs.
"""

from __future__ import annotations

import argparse
import time

import torch

from gpu_pnr.ordering import order_nets
from gpu_pnr.router import route_nets

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from demo_multinet import make_grid, make_nets  # noqa: E402


STRATEGIES = ("identity", "hpwl_asc", "hpwl_desc")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--nets", type=int, nargs="+", default=[10, 20, 30, 50, 80])
    p.add_argument("--obstacles", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-reserve", action="store_true")
    args = p.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    reserve = not args.no_reserve

    print(f"Grid: {args.size}x{args.size}  obstacles={args.obstacles}  "
          f"device={device}  reserve_pins={reserve}")
    print()
    print(
        f"{'n_nets':>7}  {'strategy':>10}  {'routed':>10}  {'wl':>8}  "
        f"{'time_ms':>9}"
    )
    print("-" * 60)

    for n in args.nets:
        w_cpu = make_grid(args.size, args.size, args.obstacles, args.seed)
        nets = make_nets(w_cpu, n, args.seed)
        w_dev = w_cpu.to(device)

        for strategy in STRATEGIES:
            ordered = order_nets(nets, strategy)
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            results = route_nets(w_dev, ordered, reserve_pins=reserve)
            if device == "mps":
                torch.mps.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000.0
            routed = sum(1 for r in results if r.routed)
            wl = sum(r.length for r in results if r.routed)
            print(
                f"{n:>7}  {strategy:>10}  {routed:>4}/{len(ordered):<4}  "
                f"{wl:>8}  {elapsed:>9.1f}"
            )
        print()


if __name__ == "__main__":
    main()
